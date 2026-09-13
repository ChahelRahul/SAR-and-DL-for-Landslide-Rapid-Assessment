"""Regenerate compact project-authored regression rasters.

The pixel values are deterministic synthetic Sentinel-1-like dB values. Public
earthquake locations/dates only anchor the fixture geographies; no third-party
imagery pixels are redistributed.
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parent
BANDS = ("postVV", "postVH", "diffVV", "diffVH")
CASES = [
    {
        "id": "haiti-2021-ascending",
        "orbit": "ASCENDING",
        "event_date": "2021-08-14",
        "event_lon": -73.482,
        "event_lat": 18.434,
        "epsg": 32618,
        "seed": 17,
        "public_event_reference": "USGS M7.2 Nippes, Haiti earthquake, 2021-08-14",
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us6000f65h",
        "fixture_license": "MIT",
        "pixel_data": "project-authored synthetic Sentinel-1-like values; no third-party imagery pixels redistributed",
    },
    {
        "id": "sumatra-2022-descending",
        "orbit": "DESCENDING",
        "event_date": "2022-02-25",
        "event_lon": 100.101,
        "event_lat": 0.219,
        "epsg": 32647,
        "seed": 29,
        "public_event_reference": "USGS M6.1 65 km NNW of Bukittinggi, Indonesia, 2022-02-25",
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us6000gzyg",
        "fixture_license": "MIT",
        "pixel_data": "project-authored synthetic Sentinel-1-like values; no third-party imagery pixels redistributed",
    },
]

for case in CASES:
    rng = np.random.default_rng(case["seed"])
    x, y = transform("EPSG:4326", f"EPSG:{case['epsg']}", [case["event_lon"]], [case["event_lat"]])
    width, height, res = 96, 80, 10.0
    left = x[0] - width * res / 2
    top = y[0] + height * res / 2
    aff = from_origin(left, top, res, res)
    yy, xx = np.mgrid[0:height, 0:width]
    bump = np.exp(-(((xx - 55) / 16) ** 2 + ((yy - 35) / 13) ** 2))
    stripe = np.sin(xx / 11.0) * np.cos(yy / 9.0)
    post_vv = -16.0 + 2.8 * bump + 0.45 * stripe + rng.normal(0, 0.08, (height, width))
    post_vh = -23.0 + 3.4 * bump + 0.35 * stripe + rng.normal(0, 0.08, (height, width))
    diff_vv = -0.8 + 5.1 * bump + 0.15 * stripe + rng.normal(0, 0.05, (height, width))
    diff_vh = -0.5 + 6.0 * bump + 0.18 * stripe + rng.normal(0, 0.05, (height, width))
    data = np.stack([post_vv, post_vh, diff_vv, diff_vh]).astype("float32")

    # Stable small NoData corner exercises mask/fill handling without affecting the ROI core.
    data[:, :2, :3] = -9999.0
    tif = ROOT / f"{case['id']}.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", width=width, height=height, count=4,
        dtype="float32", crs=f"EPSG:{case['epsg']}", transform=aff,
        nodata=-9999.0, compress="deflate",
    ) as dst:
        dst.write(data)
        for i, name in enumerate(BANDS, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(
            source="synthetic-regression-fixture",
            orbit=case["orbit"],
            relative_orbit="",
            event_date=case["event_date"],
            input_band_order=",".join(BANDS),
            fixture_license="MIT",
        )

    # 600 x 500 m public test ROI centred at the public event location.
    roi_left, roi_right = x[0] - 300, x[0] + 300
    roi_bottom, roi_top = y[0] - 250, y[0] + 250
    lons, lats = transform(f"EPSG:{case['epsg']}", "EPSG:4326",
                           [roi_left, roi_right, roi_right, roi_left, roi_left],
                           [roi_bottom, roi_bottom, roi_top, roi_top, roi_bottom])
    roi = {"type":"Polygon", "coordinates":[[[a,b] for a,b in zip(lons,lats)]]}
    (ROOT / f"{case['id']}.roi.geojson").write_text(json.dumps(roi, indent=2)+"\n")

(ROOT / "source-manifest.json").write_text(json.dumps({"fixtures": CASES}, indent=2)+"\n")
