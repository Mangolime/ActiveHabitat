"""Enrich scored hexes with nearby names, POI distances and road mix."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ROAD_MIX_CACHE = DATA / "hex_road_mix.parquet"
RAIL_KM_CACHE = DATA / "hex_rail_km.parquet"
ACTIVE_RAILS = DATA / "active_rails.gpkg"
HOUSING_GPKG = DATA / "buildings_housing.gpkg"
HOUSING_CACHE = DATA / "hex_housing.parquet"
PARK_NEAR_CACHE = DATA / "hex_park_near.parquet"
PARKS_GPKG = DATA / "parks_and_water.gpkg"
NONHOUSING_ZONES_GPKG = DATA / "nonhousing_zones.gpkg"
CRS_METRIC = "EPSG:32637"
# Короткие уголки пути в гексе не считаем «через район».
RAIL_THROUGH_MIN_KM = 0.25
# Одна скромная секция ~500 м² (p25 apartments в Москве).
HOUSING_MIN_HA = 0.05
# Как park_nearest_min_ha для бега: цель, куда побежишь.
PARK_NEAR_MIN_HA = 10.0
MAGISTRAL_EPS_KM = 0.05

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


def ensure_rail_km() -> pd.DataFrame:
    """Длина активных ж/д путей внутри гекса (км), без spur/siding/yard/industrial."""
    if not ACTIVE_RAILS.exists():
        raise FileNotFoundError(
            f"Нет {ACTIVE_RAILS}. Сначала: python pipeline/02c_active_rails.py"
        )
    cache_stale = True
    if RAIL_KM_CACHE.exists():
        cache_stale = ACTIVE_RAILS.stat().st_mtime > RAIL_KM_CACHE.stat().st_mtime
    if RAIL_KM_CACHE.exists() and not cache_stale:
        cached = pd.read_parquet(RAIL_KM_CACHE)
        if {"h3_index", "rail_km"}.issubset(cached.columns):
            return cached

    print("   ж/д в гексе: режу active_rails по H3...")
    hexes = gpd.read_parquet(DATA / "moscow_grid_h3.parquet")[["h3_index", "geometry"]]
    rails = gpd.read_file(ACTIVE_RAILS, layer="rails")
    hex_m = hexes.to_crs(CRS_METRIC)
    rails_m = rails.to_crs(CRS_METRIC)
    clipped = gpd.overlay(rails_m, hex_m, how="intersection", keep_geom_type=False)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped["seg_m"] = clipped.geometry.length
    agg = (
        clipped.groupby("h3_index")["seg_m"]
        .sum()
        .rename("rail_m")
        .reset_index()
    )
    out = hexes[["h3_index"]].merge(agg, on="h3_index", how="left")
    out["rail_m"] = out["rail_m"].fillna(0.0)
    out["rail_km"] = (out["rail_m"] / 1000.0).round(3)
    out["has_rail"] = (out["rail_km"] >= RAIL_THROUGH_MIN_KM).astype(int)
    out.to_parquet(RAIL_KM_CACHE, index=False)
    n_hit = int((out["has_rail"] == 1).sum())
    print(f"   записала {RAIL_KM_CACHE}: гексов с ж/д ≥{RAIL_THROUGH_MIN_KM} км — {n_hit}")
    return out


def ensure_housing_ha() -> pd.DataFrame:
    """Суммарная площадь жилого footprint в гексе (га).

    Здания с центроидом внутри парка / леса / ООПТ и крупных нежилых зон
    (аэропорты, промзоны, военные, ж/д, свалки, …) не считаем жильём.
    """
    if not HOUSING_GPKG.exists():
        raise FileNotFoundError(
            f"Нет {HOUSING_GPKG}. Сначала: python pipeline/02d_buildings_housing.py"
        )
    cache_ver = "v3_exclude_park_nonhousing"
    cache_stale = True
    if HOUSING_CACHE.exists():
        cache_stale = HOUSING_GPKG.stat().st_mtime > HOUSING_CACHE.stat().st_mtime
        if PARKS_GPKG.exists():
            cache_stale = cache_stale or (
                PARKS_GPKG.stat().st_mtime > HOUSING_CACHE.stat().st_mtime
            )
        if NONHOUSING_ZONES_GPKG.exists():
            cache_stale = cache_stale or (
                NONHOUSING_ZONES_GPKG.stat().st_mtime > HOUSING_CACHE.stat().st_mtime
            )
    if HOUSING_CACHE.exists() and not cache_stale:
        cached = pd.read_parquet(HOUSING_CACHE)
        if (
            {"h3_index", "housing_ha", "has_housing"}.issubset(cached.columns)
            and cached.attrs.get("ver") == cache_ver
        ):
            return cached
        if "excl_nonhousing_v2" in cached.columns and {
            "h3_index",
            "housing_ha",
            "has_housing",
        }.issubset(cached.columns):
            return cached

    print("   жильё в гексе: без зданий в парках и нежилых зонах...")
    hexes = gpd.read_parquet(DATA / "moscow_grid_h3.parquet")[["h3_index", "geometry"]]
    buildings = gpd.read_file(HOUSING_GPKG, layer="housing")
    hex_m = hexes.to_crs(CRS_METRIC)
    bld_m = buildings.to_crs(CRS_METRIC)
    cents = bld_m.set_geometry(bld_m.geometry.centroid, crs=bld_m.crs)

    drop_ix: set = set()
    if PARKS_GPKG.exists():
        parks = gpd.read_file(PARKS_GPKG)
        if "category" in parks.columns:
            parks = parks[parks["category"] == "park"].copy()
        parks_m = parks.to_crs(CRS_METRIC)[["geometry"]].reset_index(drop=True)
        in_park = gpd.sjoin(cents, parks_m, how="inner", predicate="within")
        n_park = len(set(in_park.index.unique()))
        drop_ix |= set(in_park.index.unique())
        print(f"   отброшено зданий в парках: {n_park}")

    if NONHOUSING_ZONES_GPKG.exists():
        zones = gpd.read_file(NONHOUSING_ZONES_GPKG)
        zones_m = zones.to_crs(CRS_METRIC)[["geometry", "kind"]].reset_index(drop=True)
        in_zone = gpd.sjoin(cents, zones_m, how="inner", predicate="within")
        n_zone = len(set(in_zone.index.unique()))
        drop_ix |= set(in_zone.index.unique())
        by_kind = (
            in_zone.groupby("kind").size().sort_values(ascending=False)
            if len(in_zone)
            else pd.Series(dtype=int)
        )
        print(f"   отброшено зданий в нежилых зонах: {n_zone}")
        if len(by_kind):
            print("     по типам:", ", ".join(f"{k}={int(v)}" for k, v in by_kind.items()))
    else:
        print(
            f"   нет {NONHOUSING_ZONES_GPKG.name} — "
            "сначала python pipeline/02e_nonhousing_zones.py"
        )

    bld_m = bld_m.loc[~bld_m.index.isin(drop_ix)].copy()
    print(f"   осталось зданий: {len(bld_m)} (отброшено {len(drop_ix)})")

    clipped = gpd.overlay(bld_m, hex_m, how="intersection", keep_geom_type=False)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped["area_m2"] = clipped.geometry.area
    agg = clipped.groupby("h3_index")["area_m2"].sum().rename("housing_m2").reset_index()
    out = hexes[["h3_index"]].merge(agg, on="h3_index", how="left")
    out["housing_m2"] = out["housing_m2"].fillna(0.0)
    out["housing_ha"] = (out["housing_m2"] / 10_000.0).round(4)

    # Гекс, где парк или нежилая зона занимает ≥ половины — не «место жилья».
    out["park_share"] = 0.0
    out["nonhousing_share"] = 0.0
    out["airport_share"] = 0.0
    hex_area = hex_m.set_index("h3_index").geometry.area
    out = out.set_index("h3_index")

    if PARKS_GPKG.exists():
        parks = gpd.read_file(PARKS_GPKG)
        if "category" in parks.columns:
            parks = parks[parks["category"] == "park"].copy()
        parks_m = parks.to_crs(CRS_METRIC)
        park_clip = gpd.overlay(
            parks_m[["geometry"]], hex_m, how="intersection", keep_geom_type=False
        )
        park_clip = park_clip[park_clip.geometry.notna() & ~park_clip.geometry.is_empty]
        park_clip["park_m2"] = park_clip.geometry.area
        park_agg = park_clip.groupby("h3_index")["park_m2"].sum()
        share = (park_agg / hex_area).fillna(0.0).clip(upper=1.0)
        out["park_share"] = share.reindex(out.index).fillna(0.0)

    if NONHOUSING_ZONES_GPKG.exists():
        zones = gpd.read_file(NONHOUSING_ZONES_GPKG).to_crs(CRS_METRIC)
        z_clip = gpd.overlay(
            zones[["geometry", "kind"]], hex_m, how="intersection", keep_geom_type=False
        )
        z_clip = z_clip[z_clip.geometry.notna() & ~z_clip.geometry.is_empty]
        z_clip["z_m2"] = z_clip.geometry.area
        z_agg = z_clip.groupby("h3_index")["z_m2"].sum()
        z_share = (z_agg / hex_area).fillna(0.0).clip(upper=1.0)
        out["nonhousing_share"] = z_share.reindex(out.index).fillna(0.0)
        air = z_clip[z_clip["kind"] == "airport"]
        if len(air):
            air_agg = air.groupby("h3_index")["z_m2"].sum()
            air_share = (air_agg / hex_area).fillna(0.0).clip(upper=1.0)
            out["airport_share"] = air_share.reindex(out.index).fillna(0.0)

    out = out.reset_index()
    # Аэропорт: даже частичное покрытие гекса (≥15%) — не жильё.
    # Остальные нежилые зоны / парки — от половины площади.
    blocked = (
        (out["park_share"] >= 0.5)
        | (out["nonhousing_share"] >= 0.5)
        | (out["airport_share"] >= 0.15)
    )
    out["has_housing"] = (
        (out["housing_ha"] >= HOUSING_MIN_HA) & ~blocked
    ).astype(int)
    out["excl_park"] = 1
    out["excl_nonhousing"] = 1
    out["excl_nonhousing_v2"] = 1
    out.to_parquet(HOUSING_CACHE, index=False)
    n_hit = int(out["has_housing"].sum())
    n_parkish = int((out["park_share"] >= 0.5).sum())
    n_zoneish = int((out["nonhousing_share"] >= 0.5).sum())
    n_air = int((out["airport_share"] >= 0.15).sum())
    print(
        f"   записала {HOUSING_CACHE}: жильё={n_hit}, "
        f"≥50% парк={n_parkish}, ≥50% нежилая={n_zoneish}, ≥15% аэропорт={n_air}"
    )
    return out


def ensure_park_near() -> pd.DataFrame:
    """Парк ≥ PARK_NEAR_MIN_HA га в гексе или соседях H3 k=1."""
    import h3

    if not PARKS_GPKG.exists():
        raise FileNotFoundError(f"Нет {PARKS_GPKG}")
    cache_stale = True
    if PARK_NEAR_CACHE.exists():
        cache_stale = PARKS_GPKG.stat().st_mtime > PARK_NEAR_CACHE.stat().st_mtime
    if PARK_NEAR_CACHE.exists() and not cache_stale:
        cached = pd.read_parquet(PARK_NEAR_CACHE)
        if {"h3_index", "park_near"}.issubset(cached.columns):
            return cached

    print(f"   парк рядом: объекты ≥ {PARK_NEAR_MIN_HA} га, гекс + k=1...")
    hexes = gpd.read_parquet(DATA / "moscow_grid_h3.parquet")[["h3_index", "geometry"]]
    parks = gpd.read_file(PARKS_GPKG)
    if "category" in parks.columns:
        parks = parks[parks["category"] == "park"].copy()
    if "area_ha" not in parks.columns:
        parks = parks.to_crs(CRS_METRIC)
        parks["area_ha"] = parks.geometry.area / 10_000.0
        parks = parks.to_crs(4326)
    big = parks[parks["area_ha"] >= PARK_NEAR_MIN_HA].copy()
    if big.empty:
        out = hexes[["h3_index"]].copy()
        out["park_near"] = 0
        out.to_parquet(PARK_NEAR_CACHE, index=False)
        return out

    hit = gpd.sjoin(
        hexes,
        big[["geometry"]].reset_index(drop=True),
        how="inner",
        predicate="intersects",
    )
    direct = set(hit["h3_index"].astype(str).unique())
    near: set[str] = set()
    for hid in direct:
        try:
            near.update(str(x) for x in h3.grid_disk(hid, 1))
        except Exception:
            near.add(hid)
    out = hexes[["h3_index"]].copy()
    out["park_near"] = out["h3_index"].astype(str).isin(near).astype(int)
    out.to_parquet(PARK_NEAR_CACHE, index=False)
    print(f"   записала {PARK_NEAR_CACHE}: park_near=1 у {int(out['park_near'].sum())} гексов")
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
