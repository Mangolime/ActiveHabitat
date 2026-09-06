"""Study-area polygon: H3 hex union plus a metric buffer (default 10 km)."""

from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
HEX_PATH = ROOT / "data" / "processed" / "moscow_grid_h3.parquet"
EXTENT_PATH = ROOT / "data" / "processed" / "extract_extent.geojson"
CRS_METRIC = "EPSG:32637"
BUFFER_M = 10_000


def hex_union_buffered(buffer_m: float = BUFFER_M):
    hexes = gpd.read_parquet(HEX_PATH)
    union = hexes.to_crs(CRS_METRIC).union_all().buffer(buffer_m)
    poly_wgs = gpd.GeoSeries([union], crs=CRS_METRIC).to_crs(4326).iloc[0]
    return poly_wgs


def overpass_bbox(poly_wgs) -> str:
    minx, miny, maxx, maxy = poly_wgs.bounds
    return f"({miny:.5f},{minx:.5f},{maxy:.5f},{maxx:.5f})"


def save_extent(poly_wgs, path: Path = EXTENT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"buffer_m": [BUFFER_M]}, geometry=[poly_wgs], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return path


def load_bbox_and_polygon(buffer_m: float = BUFFER_M):
    poly = hex_union_buffered(buffer_m)
    bbox = overpass_bbox(poly)
    out = save_extent(poly)
    print(f"   extract extent: bbox {bbox}, buffer {buffer_m:.0f} m → {out}")
    return poly, bbox
