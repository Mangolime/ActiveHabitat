"""Download active mainline / light-rail tracks (exclude industrial dead-ends)."""

import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import load_bbox_and_polygon

os.makedirs("data/processed", exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "ActiveHabitat/1.0"}
OUT = Path("data/processed/active_rails.gpkg")

# Активные пути: rail / light_rail. МЦК в Москве почти всегда railway=rail.
KEEP_RAILWAY = {"rail", "light_rail"}
# Заводские тупики, сортировочные и съезды — не «ж/д через район».
DROP_SERVICE = {"spur", "siding", "yard", "crossover"}
# usage=industrial — ветки к заводам даже без service=*.
DROP_USAGE = {"industrial", "military", "tourism"}


def fetch_json(query):
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  -> {endpoint}")
            resp = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=300
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Overpass failed: {last_err}")


def way_geom(el):
    geom = el.get("geometry") or []
    if len(geom) < 2:
        return None
    coords = [(p["lon"], p["lat"]) for p in geom]
    return LineString(coords)


def keep_way(tags: dict) -> bool:
    rw = str(tags.get("railway") or "").strip().lower()
    if rw not in KEEP_RAILWAY:
        return False
    # Lifecycle leftovers sometimes still tagged railway=rail.
    if str(tags.get("disused") or "").lower() in {"yes", "true", "1"}:
        return False
    if str(tags.get("abandoned") or "").lower() in {"yes", "true", "1"}:
        return False
    service = str(tags.get("service") or "").strip().lower()
    if service in DROP_SERVICE:
        return False
    usage = str(tags.get("usage") or "").strip().lower()
    if usage in DROP_USAGE:
        return False
    return True


print("0. Контур выгрузки: гексы + 10 км...")
_, BBOX = load_bbox_and_polygon()

print("1. Ways railway=rail|light_rail...")
# Тянем с геометрией; фильтр service/usage — локально (так проще отладить).
raw = fetch_json(
    f"""
[out:json][timeout:300];
(
  way["railway"="rail"]{BBOX};
  way["railway"="light_rail"]{BBOX};
);
out geom;
"""
)

rows = []
n_all = 0
n_drop_service = 0
n_drop_usage = 0
n_drop_life = 0
for el in raw.get("elements") or []:
    if el.get("type") != "way":
        continue
    n_all += 1
    tags = el.get("tags") or {}
    rw = str(tags.get("railway") or "").strip().lower()
    if rw not in KEEP_RAILWAY:
        continue
    if str(tags.get("disused") or "").lower() in {"yes", "true", "1"} or str(
        tags.get("abandoned") or ""
    ).lower() in {"yes", "true", "1"}:
        n_drop_life += 1
        continue
    service = str(tags.get("service") or "").strip().lower()
    if service in DROP_SERVICE:
        n_drop_service += 1
        continue
    usage = str(tags.get("usage") or "").strip().lower()
    if usage in DROP_USAGE:
        n_drop_usage += 1
        continue
    geom = way_geom(el)
    if geom is None:
        continue
    rows.append(
        {
            "osm_id": el["id"],
            "railway": rw,
            "service": service or "",
            "usage": usage or "",
            "name": tags.get("name") or "",
            "geometry": geom,
        }
    )

print(
    f"   ways rail/light_rail: {n_all}, keep: {len(rows)}, "
    f"drop service: {n_drop_service}, usage: {n_drop_usage}, lifecycle: {n_drop_life}"
)
if not rows:
    raise RuntimeError("Нет активных путей после фильтра.")

gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
# Схлопываем мульти-сегменты не нужно — режем по гексам позже.
if OUT.exists():
    OUT.unlink()
gdf.to_file(OUT, layer="rails", driver="GPKG")
print(f"2. Сохранено {OUT} ({len(gdf)} сегментов)")
print("   usage:", gdf["usage"].replace("", "(none)").value_counts().head(10).to_string())
print("   railway:", gdf["railway"].value_counts().to_string())
