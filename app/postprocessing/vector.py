from __future__ import annotations
from pathlib import Path
import numpy as np
from app.config import MODEL_VERSION


def write_detection_vectors(path: str | Path, mask: np.ndarray, *, transform, crs, orbit: str, relative_orbit: int | None, weights_sha256: str) -> Path | None:
    try:
        import geopandas as gpd
        from rasterio.features import shapes
        from shapely.geometry import shape
    except ImportError as exc:
        raise RuntimeError("Vector output requires the 'geo' extra") from exc
    records = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask == 1, transform=transform):
        if value == 1:
            records.append({"geometry": shape(geometry), "model_ver": MODEL_VERSION,
                            "orbit": orbit, "rel_orbit": relative_orbit,
                            "weights_sha256": weights_sha256})
    if not records:
        return None
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, crs=crs).to_file(path)
    return path
