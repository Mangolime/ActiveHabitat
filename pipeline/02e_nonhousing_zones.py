"""Download large non-housing land polygons (airports, industry, …).

Used by hex housing filter: drop buildings whose centroid falls inside,
and treat hexes mostly covered by these zones as non-residential.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import osm2geojson
import pandas as pd
import requests
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import load_bbox_and_polygon

os.makedirs("data/processed", exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "ActiveHabitat/1.0"}
CRS_METRIC = "EPSG:32637"
OUT = Path("data/processed/nonhousing_zones.gpkg")

# Минимальная площадь (га), чтобы не цеплять крошечные дворы / ошибочные куски.
MIN_HA = {
    "airport": 1.0,
    "industrial": 2.0,
    "military": 1.0,
    "railway": 2.0,
    "landfill": 1.0,
    "quarry": 1.0,
    "garages": 0.5,
    "prison": 0.5,
    "stadium": 5.0,
    "port": 2.0,
}


def fetch(query: str):
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


def kind_from_tags(tags: dict) -> str | None:
    if tags.get("aeroway") == "aerodrome":
        return "airport"
    if tags.get("landuse") == "industrial":
        return "industrial"
    if tags.get("landuse") == "military" or tags.get("military") in {
        "barracks",
        "base",
        "danger_area",
        "training_area",
        "range",
    }:
        return "military"
    if tags.get("landuse") == "railway":
        return "railway"
    if tags.get("landuse") == "landfill":
        return "landfill"
    if tags.get("landuse") == "quarry":
        return "quarry"
    if tags.get("landuse") == "garages":
        return "garages"
    if tags.get("landuse") == "port" or tags.get("industrial") == "port":
        return "port"
    if tags.get("amenity") == "prison":
        return "prison"
    if tags.get("leisure") == "stadium":
        return "stadium"
    return None


print("0. Контур...")
_, bbox = load_bbox_and_polygon()

q = f"""
[out:json][timeout:300];
(
  way["aeroway"="aerodrome"]{bbox};
  relation["aeroway"="aerodrome"]{bbox};
  way["landuse"~"^(industrial|military|railway|landfill|quarry|garages|port)$"]{bbox};
  relation["landuse"~"^(industrial|military|railway|landfill|quarry|garages|port)$"]{bbox};
  way["military"~"^(barracks|base|danger_area|training_area|range)$"]{bbox};
  relation["military"~"^(barracks|base|danger_area|training_area|range)$"]{bbox};
  way["amenity"="prison"]{bbox};
  relation["amenity"="prison"]{bbox};
  way["leisure"="stadium"]{bbox};
  relation["leisure"="stadium"]{bbox};
  way["industrial"="port"]{bbox};
  relation["industrial"="port"]{bbox};
);
out geom;
"""
print("1. Overpass non-housing zones...")
data = fetch(q)
fc = osm2geojson.json2geojson(data)
rows = []
for feat in fc.get("features") or []:
    geom = feat.get("geometry")
    if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
        continue
    props = feat.get("properties") or {}
    tags = props.get("tags") or props
    kind = kind_from_tags(tags if isinstance(tags, dict) else {})
    if not kind:
        # osm2geojson sometimes flattens tags onto properties
        kind = kind_from_tags(props)
    if not kind:
        continue
    try:
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
    except Exception:
        continue
    name = props.get("name") or (tags.get("name") if isinstance(tags, dict) else None) or ""
    rows.append({"name": name, "kind": kind, "geometry": g})

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
if gdf.empty:
    raise SystemExit("no zones fetched")

gdf_m = gdf.to_crs(CRS_METRIC)
gdf["area_ha"] = (gdf_m.geometry.area / 10_000.0).round(4)
keep = []
for kind, min_ha in MIN_HA.items():
    mask = (gdf["kind"] == kind) & (gdf["area_ha"] >= min_ha)
    keep.append(mask)
gdf = gdf[pd.concat(keep, axis=1).any(axis=1)].copy()
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)

print(f"   зон после фильтра площади: {len(gdf)}")
print(gdf.groupby("kind").agg(n=("kind", "size"), ha=("area_ha", "sum")).to_string())
# sample airports
air = gdf[gdf["kind"] == "airport"][["name", "area_ha"]].sort_values("area_ha", ascending=False)
print("   аэропорты:")
print(air.head(10).to_string(index=False))

gdf.to_file(OUT, driver="GPKG")
print(f"Готово: {OUT}")
