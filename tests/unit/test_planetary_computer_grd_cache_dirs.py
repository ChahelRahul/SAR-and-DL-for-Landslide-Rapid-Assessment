from pathlib import Path

from app.acquisition import planetary_computer as pc


def test_build_dem_creates_cache_directory_before_write(tmp_path, monkeypatch):
    created = []

    class DummyCatalogClient:
        @classmethod
        def open(cls, *args, **kwargs):
            class C:
                def search(self, **kwargs):
                    class S:
                        def items(self):
                            return []
                    return S()
            return C()

    class DummyCatalog:
        Client = DummyCatalogClient

    class DummyPC:
        @staticmethod
        def sign_inplace(x):
            return x

    def fake_deps():
        import rasterio
        from shapely.geometry import shape
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject, transform_geom
        return None, DummyPC, DummyCatalog, rasterio, None, (lambda g: shape(g).bounds), from_origin, Resampling, reproject, transform_geom

    monkeypatch.setattr(pc, "_grd_deps", fake_deps)
    cache = tmp_path / "nested" / "dem"
    roi = {"type": "Polygon", "coordinates": [[[0,0],[0.01,0],[0.01,0.01],[0,0.01],[0,0]]], "bbox": [0,0,0.01,0.01]}

    class Imagery:
        scale_m = 1000
    class Config:
        imagery = Imagery()

    try:
        pc._build_dem(roi_geojson=roi, config=Config(), cache_dir=cache)
    except ValueError as exc:
        assert "No Copernicus DEM" in str(exc)
    assert cache.is_dir()


def test_rtc_scene_creates_output_parent(tmp_path, monkeypatch):
    class DummyRIO:
        @staticmethod
        def to_raster(path, **kwargs):
            Path(path).write_bytes(b"x")

    class DummyRTC:
        rio = DummyRIO()
        def squeeze(self, drop=True):
            return self

    class DummyProduct:
        def __init__(self, *args, **kwargs):
            pass

    class DummySarsen:
        Sentinel1SarProduct = DummyProduct
        @staticmethod
        def terrain_correction(*args, **kwargs):
            return DummyRTC()

    monkeypatch.setattr(pc, "_grd_deps", lambda: (None, None, None, None, DummySarsen, None, None, None, None, None))
    out = tmp_path / "nested" / "rtc" / "scene.tif"
    got = pc._rtc_scene(tmp_path / "product.safe", "VV", tmp_path / "dem.tif", out)
    assert got == out
    assert out.is_file()
