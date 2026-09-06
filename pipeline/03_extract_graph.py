import os
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import hex_union_buffered, save_extent

os.makedirs("data/processed", exist_ok=True)

EXTRA_TAGS = [
    "surface",
    "smoothness",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "bicycle",
    "lit",
    "maxspeed",
    "lanes",
    "footway",
    "tracktype",
    "foot",
    "segregated",
]
ox.settings.useful_tags_way = list(set(ox.settings.useful_tags_way + EXTRA_TAGS))
ox.settings.timeout = 600
ox.settings.use_cache = True

# Keep motorway/trunk (MKAD, TTK) and trails (path/track/footway).
# Drop only non-network / unfinished ways.
CUSTOM_FILTER = (
    '["highway"]["area"!~"yes"]["access"!~"private"]'
    '["highway"!~"proposed|construction|abandoned|platform|raceway|elevator"]'
)

EDGE_COLS = [
    "osmid",
    "highway",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "bicycle",
    "length",
    "surface",
    "maxspeed",
    "lanes",
    "footway",
    "tracktype",
    "name",
    "ref",
    "geometry",
]


def stringify_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        return ";".join(str(x) for x in value if x is not None and str(x) != "nan")
    text = str(value)
    return "" if text in {"nan", "None"} else text


print("1. Контур графа: объединение гексов + 10 км...")
study_area = hex_union_buffered()
save_extent(study_area)
print(f"   bounds WGS84: {study_area.bounds}")

print("2. Скачивание полного дорожного графа из OSM (несколько минут)...")
G = ox.graph_from_polygon(
    study_area,
    custom_filter=CUSTOM_FILTER,
    simplify=True,
)
print(f"   Узлов: {len(G.nodes)}, рёбер: {len(G.edges)}")

print("3. Сохранение GraphML (задел на маршруты)...")
graphml_path = "data/processed/moscow_active_graph.graphml"
ox.save_graphml(G, graphml_path)
print(f"   {graphml_path}")

print("4. Экспорт рёбер в GeoPackage для QGIS и скоринга...")
edges = ox.graph_to_gdfs(G, nodes=False).reset_index(drop=True)
keep = [c for c in EDGE_COLS if c in edges.columns]
edges = edges[keep].copy()
for col in edges.columns:
    if col == "geometry":
        continue
    edges[col] = edges[col].map(stringify_cell)

gpkg_path = "data/processed/moscow_active_edges.gpkg"
edges.to_file(gpkg_path, layer="edges", driver="GPKG")
print(f"   {gpkg_path}, рёбер: {len(edges)}")

highway_counts = Counter()
for raw in edges["highway"]:
    for part in str(raw).split(";"):
        if part:
            highway_counts[part] += 1
print("5. Теги highway (фрагмент):")
for tag in [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "track",
    "path",
    "footway",
    "cycleway",
]:
    print(f"   {tag:16} {highway_counts.get(tag, 0)}")
print("Готово. В QGIS: слой edges из moscow_active_edges.gpkg")
