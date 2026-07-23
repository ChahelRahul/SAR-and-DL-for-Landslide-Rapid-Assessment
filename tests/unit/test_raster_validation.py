from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.acquisition.local_raster import RasterValidationError, read_sentinel1_stack
from app.config import AppConfig, EXPECTED_BAND_ORDER


def write_raster(path: Path, *, width=64, height=64, orbit="ASCENDING", resolution=10, nodata=None, data=None):
    if data is None:
        data = np.zeros((4, height, width), dtype="float32")
        data[0] = -15; data[1] = -20; data[2] = 1; data[3] = -1
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4,
                       dtype="float32", crs="EPSG:32630", transform=from_origin(500000, 1000, resolution, resolution), nodata=nodata) as dst:
        dst.write(data)
        for i, name in enumerate(EXPECTED_BAND_ORDER, 1): dst.set_band_description(i, name)
        dst.update_tags(input_band_order=",".join(EXPECTED_BAND_ORDER), orbit=orbit)


def test_valid_raster_writes_validation_metadata(tmp_path):
    path=tmp_path/'valid.tif'; write_raster(path)
    result=read_sentinel1_stack(path, config=AppConfig())
    assert result.metadata['validation']['status']=='passed'
    assert result.metadata['validation']['checks']['nodata_fill']=='none'


def test_small_raster_is_rejected(tmp_path):
    path=tmp_path/'small.tif'; write_raster(path, width=32, height=64)
    with pytest.raises(RasterValidationError, match='minimum is 64x64'):
        read_sentinel1_stack(path, config=AppConfig())


def test_wrong_orbit_is_rejected(tmp_path):
    path=tmp_path/'orbit.tif'; write_raster(path, orbit='DESCENDING')
    with pytest.raises(RasterValidationError, match='selected model orbit'):
        read_sentinel1_stack(path, config=AppConfig())


def test_resolution_is_rejected(tmp_path):
    path=tmp_path/'resolution.tif'; write_raster(path, resolution=30)
    with pytest.raises(RasterValidationError, match='pixel resolution'):
        read_sentinel1_stack(path, config=AppConfig())


def test_sparse_nodata_is_filled_and_reported(tmp_path):
    data=np.zeros((4,64,64),dtype='float32'); data[0]=-15; data[1]=-20; data[2]=1; data[3]=-1
    data[:,0,0]=-9999
    path=tmp_path/'nodata.tif'; write_raster(path,nodata=-9999,data=data)
    result=read_sentinel1_stack(path, config=AppConfig())
    check=result.metadata['validation']['checks']
    assert check['nodata_fraction'] > 0
    assert check['nodata_fill']=='per-band median'
    assert np.isfinite(result.data).all()


def test_out_of_range_values_are_rejected(tmp_path):
    data=np.full((4,64,64),100,dtype='float32')
    path=tmp_path/'range.tif'; write_raster(path,data=data)
    with pytest.raises(RasterValidationError, match='training-preprocessing range'):
        read_sentinel1_stack(path, config=AppConfig())
