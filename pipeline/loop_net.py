"""Per-profile routing table for loop generation, built once from the local GPKG.

No Overpass calls: `data/processed/moscow_active_edges.gpkg` already covers Moscow
plus 10 km. Weights depend only on tags, parks and proximity to high-stress roads,
so they are computed for the whole city once and cached as GeoParquet with a
covering bbox, which lets each hex read only its own neighbourhood.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CRS_METRIC = "EPSG:32637"
EDGES_GPKG = DATA / "moscow_active_edges.gpkg"
PARKS_GPKG = DATA / "parks_and_water.gpkg"

MOTORWAY = {"motorway", "motorway_link"}
TRUNK = {"trunk", "trunk_link"}
PRIMARY = {"primary", "primary_link"}
SECONDARY = {"secondary", "secondary_link"}
RUN_NICE = {"footway", "path", "pedestrian", "track", "living_street"}
BIKE_NICE = {"cycleway", "path", "living_street"}
STREET = {"residential", "unclassified", "tertiary", "tertiary_link"}
BIKE_INFRA_TOKENS = ("lane", "track", "opposite", "shared_lane", "share_busway", "separate", "yes")

# Покрытие сводим к трём классам. Если тег surface пуст, договорённость такая:
# тропы и грунтовки — грунт, всё остальное — асфальт.
SURF_HARD = {
    "asphalt", "concrete", "paved", "chipseal", "concrete:plates",
    "concrete:lanes", "metal", "wood", "rubber", "tartan", "acrylic",
}
SURF_STONE = {
    "paving_stones", "sett", "cobblestone", "unhewn_cobblestone",
    "bricks", "brick", "paving_stones:30", "grass_paver",
}
SURF_SOFT = {
    "ground", "dirt", "earth", "mud", "sand", "grass", "gravel",
    "fine_gravel", "compacted", "pebblestone", "woodchips", "unpaved",
    "clay", "snow", "ice",
}
SURF_SOFT_HIGHWAY = {"track", "path", "bridleway"}
SURF_ASPHALT, SURF_STONE_CODE, SURF_SOIL = 0, 1, 2

HIGH_STRESS_BUFFER_M = 50.0
PARK_BONUS = 0.60
NEAR_STRESS_MULT = {"run": 3.5, "bike": 3.0}

# multiplier on metres; lower is more attractive
PENALTY = {
    "run": {
        "trunk": 40.0,
        "primary": 12.0,
        "secondary": 3.0,
        "nice": 0.35,
        "cycleway": 0.55,
        "street": 1.0,
        "service": 1.3,
        "other": 1.6,
    },
    "bike": {
        "trunk": 25.0,
        "primary": 8.0,
        "secondary": 1.8,
        "bike_infra": 0.20,
        "nice": 0.55,
        "street": 0.85,
        "service": 0.85,
        "walk": 1.1,
        "no_bike": 12.0,
        "other": 1.4,
    },
}


def cache_path(profile: str) -> Path:
    return DATA / f"loop_net_{profile}.parquet"


def tag_first(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(";")
        .str[0]
        .str.strip("[]'\" ")
    )


def tag_clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).replace({"nan": "", "None": ""})


def _bike_infra(edges: pd.DataFrame) -> np.ndarray:
    hw = tag_clean(edges["highway"]) if "highway" in edges else pd.Series("", index=edges.index)
    bicycle = tag_clean(edges["bicycle"]) if "bicycle" in edges else pd.Series("", index=edges.index)
    flag = hw.str.split(";").apply(lambda parts: "cycleway" in parts)
    flag |= bicycle.eq("designated")
    cw = pd.Series("", index=edges.index)
    for col in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        if col in edges.columns:
            cw = cw.str.cat(tag_clean(edges[col]), sep=";")
    for tok in BIKE_INFRA_TOKENS:
        flag |= cw.str.contains(tok, regex=False)
    return flag.to_numpy()


def _weights(profile: str, hw1: pd.Series, bicycle: pd.Series, is_bike, in_park, near_stress, length):
    cfg = PENALTY[profile]
    n = len(hw1)
    hw = hw1.to_numpy()
    bic = bicycle.to_numpy()
    if profile == "run":
        conds = [
            np.isin(hw, list(TRUNK)),
            np.isin(hw, list(PRIMARY)),
            np.isin(hw, list(RUN_NICE)),
            hw == "cycleway",
            np.isin(hw, list(SECONDARY)),
            np.isin(hw, list(STREET)),
            hw == "service",
        ]
        vals = [
            cfg["trunk"],
            cfg["primary"],
            cfg["nice"],
            cfg["cycleway"],
            cfg["secondary"],
            cfg["street"],
            cfg["service"],
        ]
        good = np.isin(hw, list(RUN_NICE)) | (hw == "cycleway")
    else:
        walkable = np.isin(hw, ["footway", "pedestrian"]) & (bic != "no")
        conds = [
            np.isin(hw, list(TRUNK)),
            np.isin(hw, list(PRIMARY)),
            is_bike,
            (bic == "no") & np.isin(hw, ["path", "footway", "pedestrian"]),
            np.isin(hw, list(BIKE_NICE)) & (bic != "no"),
            np.isin(hw, list(STREET)) | (hw == "service"),
            walkable,
            np.isin(hw, list(SECONDARY)),
        ]
        vals = [
            cfg["trunk"],
            cfg["primary"],
            cfg["bike_infra"],
            cfg["no_bike"],
            cfg["nice"],
            cfg["street"],
            cfg["walk"],
            cfg["secondary"],
        ]
        good = is_bike | (np.isin(hw, list(BIKE_NICE)) & (bic != "no"))

    penalty = np.select(conds, vals, default=cfg["other"]).astype(float)
    penalty = np.where(in_park, penalty * PARK_BONUS, penalty)
    penalty = np.where(near_stress, penalty * NEAR_STRESS_MULT[profile], penalty)
    good = good | np.asarray(in_park, dtype=bool)
    return length * penalty, good


def build(profile: str, force: bool = False) -> Path:
    out = cache_path(profile)
    if out.exists() and not force:
        print(f"   кеш есть: {out.name}")
        return out

    print(f"   читаю {EDGES_GPKG.name} (вся Москва)...")
    edges = gpd.read_file(EDGES_GPKG, layer="edges").to_crs(CRS_METRIC)
    print(f"   рёбер: {len(edges)}")

    hw1 = tag_first(edges["highway"])
    edges = edges[~hw1.isin(MOTORWAY)].reset_index(drop=True)
    hw1 = tag_first(edges["highway"])
    bicycle = tag_clean(edges["bicycle"]) if "bicycle" in edges else pd.Series("", index=edges.index)
    print(f"   без автомагистралей: {len(edges)}")

    print("   парки...")
    parks = gpd.read_file(PARKS_GPKG)
    parks = parks[parks["category"] == "park"].to_crs(CRS_METRIC)
    in_park = np.zeros(len(edges), dtype=bool)
    hit = gpd.sjoin(edges[["geometry"]], parks[["geometry"]], predicate="intersects", how="inner")
    in_park[np.unique(hit.index.to_numpy())] = True
    print(f"   рёбер в парках: {int(in_park.sum())}")

    print(f"   буфер магистралей {HIGH_STRESS_BUFFER_M:.0f} м...")
    stress = edges[hw1.isin(TRUNK | PRIMARY)]
    near_stress = np.zeros(len(edges), dtype=bool)
    if len(stress):
        band = gpd.GeoDataFrame(
            geometry=stress.geometry.buffer(HIGH_STRESS_BUFFER_M).values, crs=CRS_METRIC
        )
        hit = gpd.sjoin(edges[["geometry"]], band, predicate="intersects", how="inner")
        near_stress[np.unique(hit.index.to_numpy())] = True
    near_stress &= ~hw1.isin(TRUNK | PRIMARY).to_numpy()
    print(f"   рёбер вдоль магистралей: {int(near_stress.sum())}")

    length = pd.to_numeric(edges["length"], errors="coerce").to_numpy()
    fallback = edges.geometry.length.to_numpy()
    length = np.where(np.isfinite(length) & (length > 0), length, fallback)

    is_bike = _bike_infra(edges)
    w, good = _weights(profile, hw1, bicycle, is_bike, in_park, near_stress, length)

    surf_tag = tag_first(edges["surface"]).str.lower() if "surface" in edges else pd.Series("", index=edges.index)
    known = surf_tag.isin(SURF_HARD | SURF_STONE | SURF_SOFT)
    surf = np.full(len(edges), SURF_ASPHALT, dtype=np.int8)
    surf = np.where(surf_tag.isin(SURF_STONE), SURF_STONE_CODE, surf)
    surf = np.where(surf_tag.isin(SURF_SOFT), SURF_SOIL, surf)
    surf = np.where(~known & hw1.isin(SURF_SOFT_HIGHWAY), SURF_SOIL, surf).astype(np.int8)
    print(f"   покрытие задано тегом: {100 * known.mean():.0f}% рёбер")

    print("   концы рёбер...")
    coords, index = shapely.get_coordinates(edges.geometry.values, return_index=True)
    first = np.searchsorted(index, np.arange(len(edges)), side="left")
    last = np.searchsorted(index, np.arange(len(edges)), side="right") - 1

    out_gdf = gpd.GeoDataFrame(
        {
            "ux": coords[first, 0],
            "uy": coords[first, 1],
            "vx": coords[last, 0],
            "vy": coords[last, 1],
            "length_m": length,
            "w": w,
            "good": good,
            "is_bike": is_bike,
            "in_park": in_park,
            "near_stress": near_stress,
            "surf": surf,
            "highway": hw1.to_numpy(),
        },
        geometry=edges.geometry.values,
        crs=CRS_METRIC,
    )
    print(f"   пишу {out.name}...")
    out_gdf.to_parquet(out, write_covering_bbox=True, index=False)
    print(f"   готово: {out}")
    return out


def load_local(profile: str, center, radius_m: float) -> gpd.GeoDataFrame:
    """Edges near a point, read straight from the cache via covering bbox."""
    path = cache_path(profile)
    bbox = (center.x - radius_m, center.y - radius_m, center.x + radius_m, center.y + radius_m)
    gdf = gpd.read_parquet(path, bbox=bbox)
    if gdf.empty:
        return gdf
    dx = np.minimum(np.abs(gdf["ux"] - center.x), np.abs(gdf["vx"] - center.x))
    dy = np.minimum(np.abs(gdf["uy"] - center.y), np.abs(gdf["vy"] - center.y))
    keep = (dx * dx + dy * dy) <= radius_m * radius_m
    return gdf[keep].reset_index(drop=True)


def node_ids(gdf: gpd.GeoDataFrame):
    """Map edge endpoints to compact integer node ids (0.1 m snapping)."""
    kx = np.round(np.concatenate([gdf["ux"].to_numpy(), gdf["vx"].to_numpy()]) * 10).astype(np.int64)
    ky = np.round(np.concatenate([gdf["uy"].to_numpy(), gdf["vy"].to_numpy()]) * 10).astype(np.int64)
    key = kx * np.int64(10**9) + ky
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(gdf)
    u, v = inv[:n], inv[n:]
    xs = np.concatenate([gdf["ux"].to_numpy(), gdf["vx"].to_numpy()])
    ys = np.concatenate([gdf["uy"].to_numpy(), gdf["vy"].to_numpy()])
    node_x = np.zeros(len(uniq))
    node_y = np.zeros(len(uniq))
    node_x[inv] = xs
    node_y[inv] = ys
    return u, v, node_x, node_y


if __name__ == "__main__":
    import os

    os.chdir(ROOT)
    import sys

    force = "--force" in sys.argv
    for prof in ("run", "bike"):
        print(f"\n{prof}:")
        build(prof, force=force)
