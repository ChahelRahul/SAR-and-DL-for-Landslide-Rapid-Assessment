from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import ceil
from typing import Any, Iterable

SENTINEL1_EARLIEST_DATE = date(2014, 10, 3)


class RoiValidationError(ValueError):
    """Raised when a submitted ROI cannot be processed safely."""


@dataclass(frozen=True, slots=True)
class RoiReport:
    geometry_type: str
    area_km2: float
    bbox_width_km: float
    bbox_height_km: float
    vertex_count: int
    crosses_antimeridian: bool
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_event_date(event_date: date, *, today: date | None = None) -> None:
    today = today or date.today()
    if event_date < SENTINEL1_EARLIEST_DATE:
        raise RoiValidationError(
            f"event date {event_date.isoformat()} predates supported Sentinel-1 GRD coverage "
            f"({SENTINEL1_EARLIEST_DATE.isoformat()})"
        )
    if event_date > today:
        raise RoiValidationError(f"event date {event_date.isoformat()} is in the future")


def validate_roi(
    payload: dict[str, Any],
    *,
    max_area_km2: float,
    max_width_km: float,
    max_height_km: float,
    max_vertices: int,
) -> RoiReport:
    try:
        from pyproj import Geod
        from shapely.geometry import shape
        from shapely.validation import explain_validity
    except ImportError as exc:
        raise RuntimeError("ROI validation requires the 'geo' extra (shapely and pyproj)") from exc

    geometry_mapping = payload.get("geometry", payload) if payload.get("type") == "Feature" else payload
    if not isinstance(geometry_mapping, dict):
        raise RoiValidationError("ROI Feature must contain a geometry")
    geometry_type = geometry_mapping.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise RoiValidationError("ROI must be a GeoJSON Polygon or MultiPolygon")

    coordinates = geometry_mapping.get("coordinates")
    if coordinates is None:
        raise RoiValidationError("ROI geometry has no coordinates")
    points = list(_iter_positions(coordinates))
    if not points:
        raise RoiValidationError("ROI geometry contains no coordinate positions")
    for lon, lat in points:
        if not (-180.0 <= lon <= 180.0):
            raise RoiValidationError(f"longitude {lon} is outside [-180, 180]")
        if not (-90.0 <= lat <= 90.0):
            raise RoiValidationError(f"latitude {lat} is outside [-90, 90]")

    vertex_count = len(points)
    if vertex_count > max_vertices:
        raise RoiValidationError(
            f"ROI has {vertex_count:,} vertices; configured limit is {max_vertices:,}"
        )

    geometry = shape(geometry_mapping)
    if geometry.is_empty:
        raise RoiValidationError("ROI geometry is empty")
    if not geometry.is_valid:
        raise RoiValidationError(f"ROI geometry is invalid: {explain_validity(geometry)}")

    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geometry)
    area_km2 = abs(float(area_m2)) / 1_000_000.0
    if area_km2 <= 0:
        raise RoiValidationError("ROI has zero geodesic area")

    longitudes = [p[0] for p in points]
    latitudes = [p[1] for p in points]
    raw_span = max(longitudes) - min(longitudes)
    crosses_antimeridian = raw_span > 180.0
    west, east = _minimal_longitude_interval(longitudes)
    south, north = min(latitudes), max(latitudes)
    center_lat = (south + north) / 2.0
    _, _, width_m = geod.inv(west, center_lat, east, center_lat)
    _, _, height_m = geod.inv(west, south, west, north)
    bbox_width_km = abs(float(width_m)) / 1000.0
    bbox_height_km = abs(float(height_m)) / 1000.0

    violations: list[str] = []
    if area_km2 > max_area_km2:
        violations.append(f"area {area_km2:.2f} km² exceeds limit {max_area_km2:.2f} km²")
    if bbox_width_km > max_width_km:
        violations.append(
            f"bounding-box width {bbox_width_km:.2f} km exceeds limit {max_width_km:.2f} km"
        )
    if bbox_height_km > max_height_km:
        violations.append(
            f"bounding-box height {bbox_height_km:.2f} km exceeds limit {max_height_km:.2f} km"
        )
    if violations:
        raise RoiValidationError("ROI processing limit exceeded: " + "; ".join(violations))

    return RoiReport(
        geometry_type=geometry_type,
        area_km2=area_km2,
        bbox_width_km=bbox_width_km,
        bbox_height_km=bbox_height_km,
        vertex_count=vertex_count,
        crosses_antimeridian=crosses_antimeridian,
    )


def _iter_positions(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value[:2]):
            yield float(value[0]), float(value[1])
            return
        for item in value:
            yield from _iter_positions(item)


def _minimal_longitude_interval(longitudes: list[float]) -> tuple[float, float]:
    """Return west/east endpoints of the shortest longitude arc containing all points."""
    if len(longitudes) == 1:
        return longitudes[0], longitudes[0]
    normalized = sorted((lon + 360.0) % 360.0 for lon in longitudes)
    gaps: list[tuple[float, int]] = []
    for i, value in enumerate(normalized):
        nxt = normalized[(i + 1) % len(normalized)] + (360.0 if i == len(normalized) - 1 else 0.0)
        gaps.append((nxt - value, i))
    _, index = max(gaps)
    west_360 = normalized[(index + 1) % len(normalized)]
    east_360 = normalized[index]
    if east_360 < west_360:
        east_360 += 360.0
    west = ((west_360 + 180.0) % 360.0) - 180.0
    span = east_360 - west_360
    east = west + span
    # pyproj accepts longitudes beyond 180 for the short antimeridian arc.
    return west, east
