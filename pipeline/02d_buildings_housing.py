"""Download OSM buildings, classify residential footprint, save GPKG.

Filter rules (ActiveHabitat housing filter):
  living: apartments, house, residential, …
  non-living: office, industrial, school, …
  unclear (building=yes etc.) → living
  building:use overrides form when present
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import load_bbox_and_polygon

os.makedirs("data/processed", exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "ActiveHabitat/1.0"}
OUT = Path("data/processed/buildings_housing.gpkg")

LIVING = {
    "apartments",
    "residential",
    "house",
    "detached",
    "semidetached_house",
    "terrace",
    "bungalow",
    "dormitory",
    "allotment_house",
    "cabin",
    "farm",
    "houseboat",
    "static_caravan",
    "villa",
    "maisonette",
    "ger",
    "stilt_house",
}
NON_LIVING = {
    "office",
    "industrial",
    "warehouse",
    "retail",
    "commercial",
    "service",
    "garage",
    "garages",
    "shed",
    "hangar",
    "school",
    "kindergarten",
    "university",
    "college",
    "hospital",
    "clinic",
    "public",
    "civic",
    "government",
    "church",
    "chapel",
    "cathedral",
    "mosque",
    "synagogue",
    "temple",
    "hotel",
    "train_station",
    "transportation",
    "parking",
    "roof",
    "construction",
    "ruins",
    "kiosk",
    "greenhouse",
    "sports_centre",
    "stadium",
    "grandstand",
    "pavilion",
    "fire_station",
    "police",
    "prison",
    "guardhouse",
    "bunker",
    "toilets",
    "bridge",
    "collapsed",
    "proposed",
    "farm_auxiliary",
    "barn",
    "cowshed",
    "stable",
    "slurry_tank",
    "transformer_tower",
    "container",
}


def fetch_json(query: str):
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  -> {endpoint}")
            resp = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=600
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}"
            print(f"     {last_err}")
        except Exception as exc:
            last_err = exc
            print(f"     {exc}")
    raise RuntimeError(f"Overpass failed: {last_err}")


def classify(building: str, building_use: str) -> bool | None:
    """True = living, False = non-living."""
    use = (building_use or "").strip().lower()
    if use:
        if use in LIVING or use in {"apartments", "house", "residential"}:
            return True
        if use in NON_LIVING or use in {"offices", "religious", "education", "sport"}:
            return False
    b = (building or "").strip().lower()
    if b in LIVING:
        return True
    if b in NON_LIVING:
        return False
    # unclear → living (incl. building=yes)
    return True


def way_poly(el) -> Polygon | None:
    geom = el.get("geometry") or []
    if len(geom) < 3:
        return None
    coords = [(p["lon"], p["lat"]) for p in geom]
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def tile_bboxes(south, west, north, east, n_lat=3, n_lon=3):
    tiles = []
    for i in range(n_lat):
        for j in range(n_lon):
            s = south + (north - south) * i / n_lat
            n = south + (north - south) * (i + 1) / n_lat
            w = west + (east - west) * j / n_lon
            e = west + (east - west) * (j + 1) / n_lon
            tiles.append((s, w, n, e))
    return tiles


print("0. Контур...")
poly, _ = load_bbox_and_polygon()
west, south, east, north = poly.bounds
tiles = tile_bboxes(south, west, north, east, 3, 3)
print(f"   tiles: {len(tiles)}")

rows = []
seen = set()
for ti, (s, w, n, e) in enumerate(tiles, 1):
    print(f"1.{ti}/{len(tiles)} buildings ({s:.3f},{w:.3f},{n:.3f},{e:.3f})...")
    q = f"""
[out:json][timeout:300];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out geom;
"""
    try:
        data = fetch_json(q)
    except Exception as exc:
        print(f"   SKIP tile: {exc}")
        continue
    for el in data.get("elements") or []:
        oid = (el.get("type"), el.get("id"))
        if oid in seen:
            continue
        tags = el.get("tags") or {}
        b = tags.get("building")
        if not b:
            continue
        living = classify(b, tags.get("building:use") or "")
        if living is not True:
            continue
        if el.get("type") == "way":
            geom = way_poly(el)
        else:
            # relations: skip complex multipolygons for MVP (rare for housing)
            continue
        if geom is None or geom.area <= 0:
            continue
        seen.add(oid)
        rows.append(
            {
                "osm_id": el["id"],
                "building": b,
                "geometry": geom,
            }
        )
    print(f"   keep living so far: {len(rows)}")

if not rows:
    raise SystemExit("no buildings")

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
# clip to study polygon
gdf = gdf[gdf.intersects(poly)].copy()
print(f"2. after clip: {len(gdf)}")
if OUT.exists():
    OUT.unlink()
gdf.to_file(OUT, layer="housing", driver="GPKG")
print(f"3. saved {OUT}")
