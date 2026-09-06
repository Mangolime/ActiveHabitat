"""Enrich scored hexes with nearby names, POI distances and road mix."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ROAD_MIX_CACHE = DATA / "hex_road_mix.parquet"
CRS_METRIC = "EPSG:32637"

MAGISTRAL = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
}
STREETS = {
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
}
WALK = {"footway", "pedestrian", "steps"}
TRAIL = {"path", "track"}
BIKE = {"cycleway"}
MIX_METER_COLS = ("mix_magistral", "mix_street", "mix_walk", "mix_path", "mix_bike")


def _highway_one(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    text = str(value).strip()
    if text.startswith("[") and "," in text:
        return text.strip("[]'\" ").split(",")[0].strip(" '\"")
    return text


def _bucket(highway: str) -> str | None:
    if highway in MAGISTRAL:
        return "mix_magistral"
    if highway in STREETS:
        return "mix_street"
    if highway in WALK:
        return "mix_walk"
    if highway in TRAIL:
        return "mix_path"
    if highway in BIKE:
        return "mix_bike"
    return None


def ensure_road_mix() -> pd.DataFrame:
    km_cols = [col + "_km" for col in MIX_METER_COLS]
    edges_path = DATA / "moscow_active_edges.gpkg"
    cache_stale = True
    if ROAD_MIX_CACHE.exists() and edges_path.exists():
        cache_stale = edges_path.stat().st_mtime > ROAD_MIX_CACHE.stat().st_mtime
    if ROAD_MIX_CACHE.exists() and not cache_stale:
        cached = pd.read_parquet(ROAD_MIX_CACHE)
        if all(col in cached.columns for col in km_cols):
            return cached

    print("   длины дорог: считаю из рёбер (без service, с парковыми тропами)...")
    hexes = gpd.read_parquet(DATA / "moscow_grid_h3.parquet")[["h3_index", "geometry"]]
    edges = gpd.read_file(
        DATA / "moscow_active_edges.gpkg",
        layer="edges",
        columns=["highway", "length", "geometry"],
    )
    edges["highway"] = edges["highway"].map(_highway_one)
    edges["bucket"] = edges["highway"].map(_bucket)
    edges = edges[edges["bucket"].notna()].copy()
    edges["length_m"] = pd.to_numeric(edges["length"], errors="coerce").fillna(0.0)
    hex_m = hexes.to_crs(CRS_METRIC)
    edges_m = edges.to_crs(CRS_METRIC)
    pts = edges_m.set_geometry(edges_m.geometry.centroid, crs=hex_m.crs)
    joined = gpd.sjoin(
        pts[["bucket", "length_m", "geometry"]],
        hex_m,
        how="inner",
        predicate="within",
    )
    wide = (
        joined.groupby(["h3_index", "bucket"])["length_m"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for col in MIX_METER_COLS:
        if col not in wide.columns:
            wide[col] = 0.0
        wide[col + "_km"] = (wide[col] / 1000.0).round(2)
    out = hexes[["h3_index"]].merge(wide[["h3_index"] + km_cols], how="left")
    for col in km_cols:
        out[col] = out[col].fillna(0.0)
    out.to_parquet(ROAD_MIX_CACHE, index=False)
    print(f"   записала {ROAD_MIX_CACHE}")
    return out


def nearest_dist(hex_gdf: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> np.ndarray:
    if targets.empty:
        return np.full(len(hex_gdf), np.inf)
    joined = gpd.sjoin_nearest(
        hex_gdf[["h3_index", "geometry"]],
        targets[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="dist_m",
    )
    joined = joined.drop_duplicates("h3_index")
    return (
        hex_gdf[["h3_index"]]
        .merge(joined[["h3_index", "dist_m"]], on="h3_index", how="left")["dist_m"]
        .fillna(np.inf)
        .to_numpy()
    )


def nearby_names(hex_gdf: gpd.GeoDataFrame, places: gpd.GeoDataFrame, n: int = 2) -> list[str]:
    named = places[places["name"].fillna("").str.len() > 0]
    if named.empty:
        return [""] * len(hex_gdf)
    joined = gpd.sjoin(
        hex_gdf[["h3_index", "geometry"]],
        named[["name", "geometry"]],
        how="left",
        predicate="intersects",
    )

    def join_names(series: pd.Series) -> str:
        vals = [v for v in series.dropna().unique().tolist() if v]
        return ", ".join(vals[:n])

    agg = joined.groupby("h3_index")["name"].agg(join_names)
    return hex_gdf["h3_index"].map(agg).fillna("").tolist()


def why_run(park_ha, walk_pct, stress_pct) -> str:
    park = "много парков" if park_ha >= 15 else ("есть парки" if park_ha >= 3 else "мало парков")
    net = "удобные дорожки" if walk_pct >= 45 else ("средняя сеть" if walk_pct >= 25 else "мало дорожек")
    stress = "тихо" if stress_pct <= 4 else ("есть магистрали" if stress_pct <= 12 else "шумные магистрали")
    return f"{park} · {net} · {stress}"


def why_bike(park_ha, quiet_pct, stress_pct) -> str:
    park = "много парков" if park_ha >= 25 else ("есть парки" if park_ha >= 8 else "мало парков")
    net = "спокойные улицы" if quiet_pct >= 40 else ("сеть средняя" if quiet_pct >= 20 else "мало спокойных улиц")
    stress = "тихо" if stress_pct <= 4 else ("есть магистрали" if stress_pct <= 12 else "шумные магистрали")
    return f"{park} · {net} · {stress}"
