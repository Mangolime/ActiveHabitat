"""Download parks, water, POI and named rivers for hex union + 10 km buffer."""

import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import osm2geojson
from shapely.geometry import Point, shape
from shapely.ops import linemerge, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import load_bbox_and_polygon

os.makedirs("data/processed", exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "ActiveHabitat/1.0"}
CRS_METRIC = "EPSG:32637"

WATER_DROP = {
    "wastewater",
    "sewage",
    "swimming_pool",
    "reflecting_pool",
    "moat",
    "ditch",
}
BASIN_DROP = {
    "oil",
    "infiltration",
    "detention",
    "wastewater",
    "sewage",
    "evaporation",
    "settling",
}
WASTE_NAME = re.compile(r"отстойник|илов(?:ые|ых|ая)?\s+карт|очистн", re.I)

# Крупнейшие лесопарки Москвы (Лосиный остров, Битцевский, Тушинский, долины
# Сетуни и Сходни) размечены только как boundary=protected_area — ни leisure, ни
# landuse, ни natural у них нет. Но этим же тегом описаны водоохранные зоны и
# охранные зоны памятников, поэтому берём только природоохранные классы.
PROTECT_CLASS_KEEP = {"1", "1a", "1b", "2", "3", "4", "5", "6", "7"}
PROTECT_TITLE_KEEP = re.compile(
    r"национальн\w*\s+парк|природн\w*[-\s]*истор\w*\s+парк|природн\w*\s+парк"
    r"|заказник|заповедник|лесопарк|памятник\s+природы|рекреационн",
    re.I,
)

NO_BIKE_NAMES = [
    "царицыно",
    "вднх",
    "ботанический",
    "патриаршие",
    "аптекарский",
    "останкино",
    "зарядье",
]
NO_DOG_NAMES = [
    "царицыно",
    "горького",
    "музеон",
    "вднх",
    "зарядье",
    "аптекарский",
    "эрмитаж",
    "баумана",
]


def fetch(query):
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  -> Запрос к {endpoint}...")
            resp = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=300
            )
            if resp.status_code == 200:
                return osm2geojson.json2geojson(resp.json())
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Не удалось выгрузить данные с Overpass: {last_err}")


def as_tags(value) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def feature_tags(feat) -> dict:
    props = feat.get("properties") or {}
    tags = as_tags(props.get("tags"))
    if tags:
        return tags
    return {k: v for k, v in props.items() if k not in {"type", "id", "tags"} and v is not None}


def row_tags(row) -> dict:
    tags = as_tags(row.get("tags") if hasattr(row, "get") else None)
    if tags:
        return tags
    keys = (
        "name",
        "natural",
        "leisure",
        "landuse",
        "water",
        "amenity",
        "fountain",
        "basin",
        "bicycle",
        "dog",
        "waterway",
        "boundary",
        "protect_class",
        "protection_title",
    )
    out = {}
    for key in keys:
        val = row[key] if key in row.index else None
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            out[key] = val
    return out


def merge_group(geoms):
    union = unary_union(list(geoms))
    merged = union if union.geom_type == "LineString" else linemerge(union)
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def merge_rivers(gdf):
    """Склеить русла в цельные реки и померить длину.

    В OSM одна река нарезана на десятки way: медиана отдельного куска — 337 м,
    нижняя четверть короче 60 м. По такому куску не отличить реку от обрубка
    дренажной канавы, поэтому сначала склеиваем (именованные — по имени, чтобы
    развилки притоков не рвали русло), а уже потом меряем.
    """
    parts = []
    named = gdf[gdf["name"].str.len() > 0]
    for name, group in named.groupby("name"):
        parts += [{"name": name, "geometry": g} for g in merge_group(group.geometry)]
    unnamed = gdf[gdf["name"].str.len() == 0]
    if len(unnamed):
        parts += [{"name": "", "geometry": g} for g in merge_group(unnamed.geometry)]
    out = gpd.GeoDataFrame(parts, crs=gdf.crs)
    out["length_m"] = out.geometry.length.round(1)
    return out


def is_dropped_water(tags: dict, name: str) -> bool:
    if tags.get("amenity") == "fountain" or tags.get("fountain") or tags.get("leisure") == "fountain":
        return True
    water = str(tags.get("water") or "")
    if water in WATER_DROP:
        return True
    if water == "basin" and str(tags.get("basin") or "") in BASIN_DROP:
        return True
    if name and WASTE_NAME.search(name):
        return True
    return False


def is_green_protected(tags: dict) -> bool:
    if tags.get("boundary") == "national_park":
        return True
    if str(tags.get("protect_class") or "").strip().lower() in PROTECT_CLASS_KEEP:
        return True
    return bool(PROTECT_TITLE_KEEP.search(str(tags.get("protection_title") or "")))


def park_kind(tags: dict) -> str:
    if tags.get("leisure") in {"park", "nature_reserve"}:
        return str(tags["leisure"])
    if tags.get("landuse") == "forest":
        return "forest"
    if tags.get("natural") == "wood":
        return "wood"
    if tags.get("boundary") in {"protected_area", "national_park"}:
        return "protected"
    return "other"


def parse_feature(row):
    tags = row_tags(row)
    name = str(tags.get("name") or "")
    is_water = tags.get("natural") == "water"
    if is_water:
        return pd.Series(
            {
                "name": name,
                "category": "water",
                "kind": "water",
                "bike_allowed": False,
                "dog_allowed": False,
                "water": str(tags.get("water") or ""),
                "amenity": str(tags.get("amenity") or ""),
                "keep": not is_dropped_water(tags, name),
            }
        )
    bike_ok = tags.get("bicycle") != "no" and not any(s in name.lower() for s in NO_BIKE_NAMES)
    dog_ok = tags.get("dog") != "no" and not any(s in name.lower() for s in NO_DOG_NAMES)
    kind = park_kind(tags)
    # охранный контур без собственного «зелёного» тега проверяем на класс охраны
    keep = True
    if kind == "protected":
        keep = is_green_protected(tags)
    return pd.Series(
        {
            "name": name,
            "category": "park",
            "kind": kind,
            "bike_allowed": bike_ok,
            "dog_allowed": dog_ok,
            "water": "",
            "amenity": str(tags.get("amenity") or ""),
            "keep": keep,
        }
    )


print("0. Контур выгрузки: гексы + 10 км...")
_, BBOX = load_bbox_and_polygon()

print("1. Выгрузка парков...")
q_parks = f"""
[out:json][timeout:300];
(
  way["leisure"~"^(park|nature_reserve)$"]{BBOX};
  relation["leisure"~"^(park|nature_reserve)$"]{BBOX};
  way["landuse"="forest"]{BBOX};
  relation["landuse"="forest"]{BBOX};
  way["natural"="wood"]{BBOX};
  relation["natural"="wood"]{BBOX};
  way["boundary"~"^(national_park|protected_area)$"]{BBOX};
  relation["boundary"~"^(national_park|protected_area)$"]{BBOX};
);
out geom;
"""
gdf_parks = gpd.GeoDataFrame.from_features(fetch(q_parks)["features"], crs="4326")
gdf_parks = gdf_parks[gdf_parks.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
print(f"   парков/лесов: {len(gdf_parks)}")

print("1b. Выгрузка водоёмов...")
q_water = f"""
[out:json][timeout:300];
(
  way["natural"="water"]{BBOX};
  relation["natural"="water"]{BBOX};
);
out geom;
"""
gdf_water = gpd.GeoDataFrame.from_features(fetch(q_water)["features"], crs="4326")
gdf_water = gdf_water[gdf_water.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
print(f"   полигонов natural=water: {len(gdf_water)}")

gdf_all = pd.concat([gdf_parks, gdf_water], ignore_index=True)
attrs = gdf_all.apply(parse_feature, axis=1)
gdf_all = pd.concat([gdf_all[["geometry"]], attrs], axis=1)
dropped = gdf_all[~gdf_all["keep"]]
gdf_parks = gdf_all[gdf_all["keep"]].drop(columns=["keep"]).copy()
print(f"   отброшено воды (фонтаны/отстойники/бассейны): {int((dropped.category == 'water').sum())}")
print(f"   отброшено охранных зон без природного статуса: {int((dropped.category == 'park').sum())}")

gdf_parks_m = gdf_parks.to_crs(CRS_METRIC)
gdf_parks["area_ha"] = (gdf_parks_m.geometry.area / 10000.0).round(4)
print(f"   слой parks+water: {len(gdf_parks)} (парков {int((gdf_parks.category=='park').sum())}, воды {int((gdf_parks.category=='water').sum())})")
print("   по типам:")
print(gdf_parks["kind"].value_counts().to_string())
gdf_parks.to_file("data/processed/parks_and_water.gpkg", driver="GPKG")

print("2. Выгрузка POI (вело + питьевые фонтанчики)...")
q_poi = f"""
[out:json][timeout:300];
(
  node["shop"="bicycle"]{BBOX};
  node["service:bicycle:repair"="yes"]{BBOX};
  node["amenity"="bicycle_parking"]{BBOX};
  node["amenity"="drinking_water"]{BBOX};
);
out center qt;
"""
poi_list = []
for feat in fetch(q_poi)["features"]:
    t = feature_tags(feat)
    if not t:
        continue
    cat = "other"
    if t.get("service:bicycle:repair") == "yes":
        cat = "bike_repair"
    elif t.get("shop") == "bicycle":
        cat = "bike_shop"
    elif t.get("amenity") == "bicycle_parking":
        cat = "bike_parking"
    elif t.get("amenity") == "drinking_water":
        cat = "water_fountain"
    if cat == "other":
        continue
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords:
        continue
    poi_list.append(
        {
            "category": cat,
            "name": t.get("name", "") or "",
            "opening_hours": t.get("opening_hours", "") or "",
            "website": t.get("website", "") or t.get("contact:website", "") or "",
            "phone": t.get("phone", "") or t.get("contact:phone", "") or "",
            "geometry": Point(coords),
        }
    )

gdf_poi = gpd.GeoDataFrame(poi_list, crs="4326")
gdf_poi.to_file("data/processed/active_pois.gpkg", driver="GPKG")
print(f"   POI: {len(gdf_poi)}")
if len(gdf_poi):
    print(gdf_poi["category"].value_counts().to_string())

print("3. Станции электричек — 02b_train_lines.py (не перезаписываю suburban_trains.gpkg)")

print("4. Выгрузка осевых линий рек...")
q_rivers = f"""
[out:json][timeout:300];
(
  way["waterway"~"river|canal"]{BBOX};
  relation["waterway"~"river|canal"]{BBOX};
);
out geom;
"""
rivers = []
for feat in fetch(q_rivers)["features"]:
    t = feature_tags(feat)
    name = t.get("name") or ""
    waterway_type = t.get("waterway")
    if name and waterway_type == "river" and "рек" not in name.lower():
        name = f"река {name}"
    geom = shape(feat["geometry"])
    if geom.geom_type not in ("LineString", "MultiLineString"):
        continue
    rivers.append({"name": name, "waterway": waterway_type or "", "geometry": geom})

gdf_rivers = None
gdf_rivers_named = None
if rivers:
    # для расстояния до воды нужны все русла: вдоль безымянной речки бежится
    # так же хорошо. Имена нужны только чтобы подписать безымянные полигоны.
    raw = gpd.GeoDataFrame(rivers, crs="4326")
    gdf_rivers = merge_rivers(raw.to_crs(CRS_METRIC)).to_crs("4326")
    gdf_rivers.to_file("data/processed/river_lines.gpkg", driver="GPKG")
    gdf_rivers_named = gdf_rivers[gdf_rivers["name"].str.len() > 0].copy()
    gdf_rivers_named.to_file("data/processed/named_rivers.gpkg", driver="GPKG")
    km = gdf_rivers["length_m"] / 1000.0
    print(
        f"   сегментов: {len(raw)} → склеенных русел: {len(gdf_rivers)}"
        f" (с названием {len(gdf_rivers_named)}); медиана {km.median():.2f} км,"
        f" короче 1 км: {int((km < 1).sum())}"
    )

print("5. Перенос названий рек на водные полигоны...")
water_mask = (gdf_parks["category"] == "water") & (gdf_parks["name"].fillna("") == "")
gdf_water_unnamed = gdf_parks[water_mask].copy()
if not gdf_water_unnamed.empty and gdf_rivers_named is not None:
    joined = gpd.sjoin(
        gdf_water_unnamed,
        gdf_rivers_named[["name", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    gdf_parks.loc[joined.index, "name"] = joined["name_right"]
    print(f"   восстановлено названий: {len(joined)}")

gdf_parks.to_file("data/processed/parks_and_water.gpkg", driver="GPKG")
print("Готово: parks_and_water.gpkg, active_pois.gpkg, river_lines.gpkg, named_rivers.gpkg")
