"""Clip road edges to H3 r8 hexes and compute length shares per cell."""

import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

CRS_METRIC = "EPSG:32637"
BIKE_INFRA_CYCLEWAY = {
    "lane",
    "track",
    "opposite",
    "opposite_lane",
    "shared_lane",
    "share_busway",
    "separate",
    "yes",
}
HIGH_STRESS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
}
STRESS_WEIGHT = {
    "motorway": 2.0,
    "motorway_link": 2.0,
    "trunk": 1.5,
    "trunk_link": 1.5,
    "primary": 1.0,
    "primary_link": 1.0,
}
LOW_STRESS = {
    "residential",
    "living_street",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "service",
    "cycleway",
    "path",
}
RUN_FRIENDLY = {
    "footway",
    "path",
    "pedestrian",
    "living_street",
    "track",
    "residential",
}
CYCLEWAY_COLS = ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]


def tag_contains(series, tokens):
    escaped = [re.escape(t) for t in sorted(tokens, key=len, reverse=True)]
    pattern = r"(?:^|;)(?:" + "|".join(escaped) + r")(?:;|$)"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def stress_weight_of(value) -> float:
    weights = [STRESS_WEIGHT.get(part.strip(), 0.0) for part in str(value or "").split(";")]
    return max(weights) if weights else 0.0


print("1. Загрузка гексов и рёбер...")
hexes = gpd.read_parquet("data/processed/moscow_grid_h3.parquet")[["h3_index", "geometry"]]
edges = gpd.read_file("data/processed/moscow_active_edges.gpkg", layer="edges")
print(f"   гексов: {len(hexes)}, рёбер: {len(edges)}")

print("2. Классификация рёбер...")
highway = edges["highway"] if "highway" in edges.columns else pd.Series("", index=edges.index)
bicycle = edges["bicycle"] if "bicycle" in edges.columns else pd.Series("", index=edges.index)
cycleway = pd.Series("", index=edges.index)
for col in CYCLEWAY_COLS:
    if col in edges.columns:
        cycleway = cycleway.str.cat(edges[col].fillna("").astype(str), sep=";")

edges["is_bike_infra"] = (
    tag_contains(highway, {"cycleway"})
    | tag_contains(cycleway, BIKE_INFRA_CYCLEWAY)
    | tag_contains(bicycle, {"designated"})
).astype(int)
edges["is_high_stress"] = tag_contains(highway, HIGH_STRESS).astype(int)
edges["stress_w"] = highway.map(stress_weight_of).astype(float)
edges["is_low_stress"] = tag_contains(highway, LOW_STRESS).astype(int)
edges["is_run_friendly"] = tag_contains(highway, RUN_FRIENDLY).astype(int)
print(
    "   bike_infra {:,} | high_stress {:,} | low_stress {:,} | run_friendly {:,}".format(
        int(edges["is_bike_infra"].sum()),
        int(edges["is_high_stress"].sum()),
        int(edges["is_low_stress"].sum()),
        int(edges["is_run_friendly"].sum()),
    )
)

print("3. Обрезка рёбер гексами (метрические длины)...")
hex_m = hexes.to_crs(CRS_METRIC)
edge_cols = [
    "geometry",
    "is_bike_infra",
    "is_high_stress",
    "stress_w",
    "is_low_stress",
    "is_run_friendly",
]
edges_m = edges[edge_cols].to_crs(CRS_METRIC)
edges_m = edges_m[edges_m.geometry.notna() & ~edges_m.geometry.is_empty]
hex_m = hex_m[hex_m.geometry.notna() & ~hex_m.geometry.is_empty]

clipped = gpd.overlay(edges_m, hex_m, how="intersection", keep_geom_type=False)
clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
clipped["seg_m"] = clipped.geometry.length
clipped["stress_m"] = clipped["seg_m"] * clipped["stress_w"].fillna(0.0)
print(f"   сегментов после clip: {len(clipped)}")

print("4. Агрегация долей по гексу...")
parts = [
    clipped.groupby("h3_index")["seg_m"].sum().rename("road_m"),
    clipped.groupby("h3_index")["stress_m"].sum().rename("high_stress_weighted_m"),
]
for flag, name in [
    ("is_bike_infra", "bike_infra_m"),
    ("is_high_stress", "high_stress_m"),
    ("is_low_stress", "low_stress_m"),
    ("is_run_friendly", "run_friendly_m"),
]:
    parts.append(
        clipped.loc[clipped[flag] == 1].groupby("h3_index")["seg_m"].sum().rename(name)
    )
agg = pd.concat(parts, axis=1).fillna(0).reset_index()

stats = hexes[["h3_index"]].merge(agg, on="h3_index", how="left")
meter_cols = [
    "road_m",
    "bike_infra_m",
    "high_stress_m",
    "high_stress_weighted_m",
    "low_stress_m",
    "run_friendly_m",
]
for col in meter_cols:
    stats[col] = stats[col].fillna(0.0)

road = stats["road_m"].replace(0, np.nan)
stats["bike_infra_share"] = (stats["bike_infra_m"] / road).fillna(0.0)
# trunk 1.5×, motorway 2×, primary 1×; clip so the penalty stays in [0, 1]
stats["high_stress_share"] = (stats["high_stress_weighted_m"] / road).clip(upper=1.0).fillna(0.0)
stats["low_stress_share"] = (stats["low_stress_m"] / road).fillna(0.0)
stats["run_friendly_share"] = (stats["run_friendly_m"] / road).fillna(0.0)

out_path = "data/processed/hex_network_stats.parquet"
stats.to_parquet(out_path, index=False)
print(f"5. Сохранено {out_path}")
print(
    stats[meter_cols + ["bike_infra_share", "high_stress_share", "low_stress_share", "run_friendly_share"]]
    .describe()
    .round(3)
    .to_string()
)
print(f"   гексов без дорог: {int((stats['road_m'] == 0).sum())} из {len(stats)}")
