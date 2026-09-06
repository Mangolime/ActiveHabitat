"""Score H3 cells for runner and cyclist profiles."""

import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_dist  # noqa: E402

CRS_METRIC = "EPSG:32637"
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "processed")

# До воды доходят по набережной, а не вплавь: целью считаем полосу вдоль берега.
WATER_BANK_M = 30.0


def km_radius(km):
    """Длина петли задаёт радиус поиска как L/4 — как в 06_sample_loops."""
    return float(km) * 1000.0 / 4.0


def by_km(mapping, km, fallback):
    if not mapping:
        return fallback
    if km in mapping:
        return mapping[km]
    return mapping.get(str(km), fallback)


def drop_train_weights(weights):
    rest = {k: v for k, v in weights.items() if k != "trains"}
    total = sum(rest.values()) or 1.0
    return {k: v / total for k, v in rest.items()}


def saturate(x, x0):
    x = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-np.clip(x, 0, None) / x0)


def decay(dist, radius):
    return np.clip(1.0 - np.asarray(dist, dtype=float) / radius, 0.0, 1.0)


def load_weights():
    path = os.path.join(os.path.dirname(__file__), "scoring_weights.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def nearest_dist(hex_gdf, targets):
    if targets.empty:
        return pd.Series(np.full(len(hex_gdf), np.inf), index=hex_gdf.index)
    joined = gpd.sjoin_nearest(
        hex_gdf[["h3_index", "geometry"]],
        targets[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="dist_m",
    )
    joined = joined.drop_duplicates("h3_index")
    return hex_gdf[["h3_index"]].merge(
        joined[["h3_index", "dist_m"]], on="h3_index", how="left"
    )["dist_m"].fillna(np.inf).to_numpy()


def dissolve(polys, minus=None):
    """Схлопнуть слой в непересекающиеся куски, при желании вычтя запретную зону.

    Зелень в OSM размечена слоями: natural=wood лежит внутри leisure=park, а тот
    — внутри boundary=protected_area. Так, у Кускова поверх лесопарка (304 га)
    нарисован ещё и безымянный лес (241 га). Площадь в радиусе считается через
    пересечение с буфером, поэтому без объединения одни и те же гектары шли бы
    в сумму дважды: по гексу над Кусковом выходило 585 га вместо 286.

    `minus` нужен велопрофилю: запрет на велосипед висит на именованном полигоне
    парка, а лес внутри него — это отдельные безымянные куски без всякого
    запрета, и 635 га Царицына возвращались бы в балл через них.
    """
    if polys.empty:
        return polys
    merged = polys.geometry.union_all()
    if minus is not None and not minus.empty:
        merged = merged.difference(minus.geometry.union_all())
    if merged.is_empty:
        return polys.iloc[0:0]
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return gpd.GeoDataFrame(geometry=parts, crs=polys.crs)


def area_in_buffer(hex_gdf, polys, radius_m):
    if polys.empty:
        return np.zeros(len(hex_gdf))
    buf = hex_gdf.copy()
    buf["geometry"] = hex_gdf.geometry.buffer(radius_m)
    src = polys[["geometry"]].reset_index(drop=True)
    inter = gpd.overlay(src, buf[["h3_index", "geometry"]], how="intersection", keep_geom_type=False)
    if inter.empty:
        return np.zeros(len(hex_gdf))
    inter["ha"] = inter.geometry.area / 10000.0
    summed = inter.groupby("h3_index")["ha"].sum()
    return hex_gdf["h3_index"].map(summed).fillna(0.0).to_numpy()


def decay_sum_points(hex_gdf, points, radius_m, weights=None):
    if points.empty:
        return np.zeros(len(hex_gdf))
    pts = points.copy()
    pts["_w"] = 1.0 if weights is None else np.asarray(weights, dtype=float)
    sindex = pts.sindex
    out = np.zeros(len(hex_gdf))
    geoms = hex_gdf.geometry.values
    for i, geom in enumerate(geoms):
        buf = geom.buffer(radius_m)
        idx = list(sindex.query(buf, predicate="intersects"))
        if not idx:
            continue
        subset = pts.iloc[idx]
        dist = subset.distance(geom).to_numpy()
        out[i] = float((decay(dist, radius_m) * subset["_w"].to_numpy()).sum())
    return out


def unique_lines_in_radius(hex_gdf, stations, radius_m):
    n_hex = len(hex_gdf)
    n_unique = np.zeros(n_hex, dtype=int)
    nearest = np.full(n_hex, np.inf)
    if stations.empty:
        return nearest, n_unique
    sindex = stations.sindex
    geoms = hex_gdf.geometry.values
    for i, geom in enumerate(geoms):
        buf = geom.buffer(radius_m)
        idx = list(sindex.query(buf, predicate="intersects"))
        if not idx:
            continue
        subset = stations.iloc[idx]
        nearest[i] = float(subset.distance(geom).min())
        lines = set()
        for raw in subset["lines"].fillna(""):
            for part in str(raw).split(";"):
                part = part.strip()
                if part:
                    lines.add(part)
        n_unique[i] = len(lines)
    return nearest, n_unique


def nearby_names(hex_gdf, places, n=3):
    named = places[places["name"].fillna("").str.len() > 0]
    if named.empty:
        return [""] * len(hex_gdf)
    joined = gpd.sjoin(
        hex_gdf[["h3_index", "geometry"]],
        named[["name", "geometry"]],
        how="left",
        predicate="intersects",
    )

    def join_names(series):
        vals = [v for v in series.dropna().unique().tolist() if v]
        return ", ".join(vals[:n])

    agg = joined.groupby("h3_index")["name"].agg(join_names)
    return hex_gdf["h3_index"].map(agg).fillna("").tolist()


def weighted_infra(hex_m, infra_m, radius_m):
    cents = hex_m.geometry.centroid
    coords = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    w = np.clip(1.0 - dist / radius_m, 0.0, 1.0)
    return w @ np.asarray(infra_m, dtype=float)


RIVER_NAME = re.compile(r"река|канал", re.I)


def ensure_area_ha(gdf):
    out = gdf.copy()
    geom_ha = out.geometry.area / 10000.0
    if "area_ha" in out.columns:
        out["area_ha"] = pd.to_numeric(out["area_ha"], errors="coerce").fillna(geom_ha)
    else:
        out["area_ha"] = geom_ha
    return out


def filter_parks(parks, min_ha):
    parks = ensure_area_ha(parks)
    return parks[parks["area_ha"] >= min_ha]


def filter_water(water, min_ha):
    water = ensure_area_ha(water)
    tag = (
        water["water"].fillna("").astype(str).str.lower()
        if "water" in water.columns
        else pd.Series("", index=water.index)
    )
    name = water["name"].fillna("").astype(str) if "name" in water.columns else pd.Series("", index=water.index)
    keep = (water["area_ha"] >= min_ha) | tag.isin(["river", "canal"]) | name.str.contains(RIVER_NAME)
    return water[keep]


print("1. Загрузка слоёв и весов...")
cfg = load_weights()
hexes = gpd.read_parquet(os.path.join(DATA, "moscow_grid_h3.parquet"))
net = pd.read_parquet(os.path.join(DATA, "hex_network_stats.parquet"))
parks = gpd.read_file(os.path.join(DATA, "parks_and_water.gpkg"))
poi = gpd.read_file(os.path.join(DATA, "active_pois.gpkg"))
trains = gpd.read_file(os.path.join(DATA, "suburban_trains.gpkg"), layer="stations")
river_path = os.path.join(DATA, "river_lines.gpkg")
rivers = gpd.read_file(river_path) if os.path.exists(river_path) else None

hexes = hexes.merge(net, on="h3_index", how="left")
for col in [
    "run_friendly_share",
    "high_stress_share",
    "low_stress_share",
    "bike_infra_share",
    "bike_infra_m",
]:
    if col not in hexes.columns:
        hexes[col] = 0.0
    hexes[col] = hexes[col].fillna(0.0)

hex_m = hexes.to_crs(CRS_METRIC)
parks_m = parks.to_crs(CRS_METRIC)
poi_m = poi.to_crs(CRS_METRIC)
trains_m = trains.to_crs(CRS_METRIC)

park_poly = parks_m[parks_m["category"] == "park"]
water_poly = parks_m[parks_m["category"] == "water"]
park_area_run = dissolve(filter_parks(park_poly, cfg["run"]["park_area_min_ha"]))
park_near_run = filter_parks(park_poly, cfg["run"]["park_nearest_min_ha"])
water_ok = filter_water(water_poly, cfg["run"]["water_min_ha"])
# вдоль реки бежится не хуже, чем вдоль пруда, поэтому русла считаем водой тоже.
# Меряем длину уже склеенного русла: отдельный way в OSM бывает и 60 м.
rivers_m = rivers.to_crs(CRS_METRIC) if rivers is not None else None
if rivers_m is not None and "length_m" in rivers_m.columns:
    rivers_m = rivers_m[rivers_m["length_m"] >= cfg["run"]["river_min_km"] * 1000.0]
if "bike_allowed" in park_poly.columns:
    allowed = park_poly["bike_allowed"].fillna(0).astype(int) == 1
    bike_parks, bike_banned = park_poly[allowed], park_poly[~allowed]
else:
    bike_parks, bike_banned = park_poly, park_poly.iloc[0:0]
bike_parks_area = dissolve(
    filter_parks(bike_parks, cfg["bike"]["park_area_min_ha"]), minus=bike_banned
)
park_near_bike = filter_parks(bike_parks, cfg["bike"]["park_nearest_min_ha"])

fountains = poi_m[poi_m["category"] == "water_fountain"]
bike_shop = poi_m[poi_m["category"] == "bike_shop"]
bike_repair = poi_m[poi_m["category"] == "bike_repair"]
bike_parking = poi_m[poi_m["category"] == "bike_parking"]
print(
    f"   парки area/nearest run: {len(park_area_run)}/{len(park_near_run)}; "
    f"вело area: {len(bike_parks_area)}; вода скоринг: {len(water_ok)} из {len(water_poly)}; "
    f"русел: {0 if rivers_m is None else len(rivers_m)}"
)

if "lines" not in trains_m.columns:
    trains_m["lines"] = ""
# Расстояние и число веток обязаны считаться по одному множеству станций.
# Иначе выходит гекс 8811aa468dfffff: 134 м до «электрички» и ноль веток —
# рядом грузовая станция Рублёво, на которую пассажирские поезда не ходят.
if "is_suburban" in trains_m.columns:
    n_before = len(trains_m)
    trains_m = trains_m[trains_m["is_suburban"].fillna(False).astype(bool)]
    print(f"   станций электрички: {len(trains_m)} из {n_before}")

hex_cents = hex_m.copy()
hex_cents["geometry"] = hex_m.geometry.centroid

# Расстояния меряем пешком по улицам, а не по прямой. Прямая врёт ровно там,
# где это дороже всего: у гекса 8811aa4da5fffff до парка 37 м по прямой и
# 4358 м пешком — между ними река без моста поблизости.
print("   граф города для расстояний...")
graph = graph_dist.CityGraph("run")
water_reach = water_ok
if rivers_m is not None and not rivers_m.empty:
    water_reach = pd.concat([water_ok[["geometry"]], rivers_m[["geometry"]]], ignore_index=True)
    water_reach = gpd.GeoDataFrame(water_reach, geometry="geometry", crs=CRS_METRIC)

barriers = water_reach
d_park = graph_dist.walking_dist(graph, hex_cents.geometry, park_near_run, barriers=barriers)
d_water = graph_dist.walking_dist(graph, hex_cents.geometry, water_reach, bank_m=WATER_BANK_M)
d_park_show, d_water_show = d_park, d_water

run_kms = list(cfg["run"].get("km") or [5])
bike_kms = list(cfg["bike"].get("km") or [15])
run_default = int(cfg["run"].get("default_km") or run_kms[0])
bike_default = int(cfg["bike"].get("default_km") or bike_kms[0])
park_ha0_run = cfg["run"].get("park_ha0_by_km") or {}
park_ha0_bike = cfg["bike"].get("park_ha0_by_km") or {}
infra0_bike = cfg["bike"].get("bike_infra_km0_by_km") or {}

print("2. Площадь зелени по радиусам...")
run_park_ha = {}
for km in run_kms:
    r = km_radius(km)
    run_park_ha[km] = area_in_buffer(hex_m, park_area_run, r)
    print(f"   бег {km} км (r={r:.0f} м): медиана {np.median(run_park_ha[km]):.0f} га")
bike_park_ha = {}
for km in bike_kms:
    r = km_radius(km)
    bike_park_ha[km] = area_in_buffer(hex_m, bike_parks_area, r)
    print(f"   вело {km} км (r={r:.0f} м): медиана {np.median(bike_park_ha[km]):.0f} га")

d_park_bike = graph_dist.walking_dist(
    graph, hex_cents.geometry, park_near_bike, barriers=barriers
)
f_run_net = hexes["run_friendly_share"].to_numpy()
f_run_stress = 1.0 - np.clip(hexes["high_stress_share"].to_numpy(), 0.0, 1.0)
f_low = hexes["low_stress_share"].to_numpy()
f_bike_stress = 1.0 - np.clip(hexes["high_stress_share"].to_numpy(), 0.0, 1.0)
w_run = cfg["run"]["weights"]
w_bike = cfg["bike"]["weights"]
w_bike_nt = drop_train_weights(w_bike)
mix = cfg["bike"]["park_mix"]

print("3. Баллы по длинам петли...")
run_scores = {}
for km in run_kms:
    r = km_radius(km)
    ha0 = float(by_km(park_ha0_run, km, cfg["run"]["park_ha0"]))
    f_park = saturate(run_park_ha[km], ha0)
    f_access = 0.5 * decay(d_park, r) + 0.5 * decay(d_water, r)
    f_amenity = saturate(decay_sum_points(hex_m, fountains, r), cfg["run"]["amenity_sat"])
    run_scores[km] = 100 * (
        w_run["park_area"] * f_park
        + w_run["park_water_access"] * f_access
        + w_run["run_friendly"] * f_run_net
        + w_run["high_stress_penalty"] * f_run_stress
        + w_run["amenities"] * f_amenity
    )

bike_scores = {}
bike_scores_nt = {}
bike_infra = {}
bike_trains = {}
for km in bike_kms:
    r = km_radius(km)
    ha0 = float(by_km(park_ha0_bike, km, cfg["bike"]["park_ha0"]))
    infra0 = float(by_km(infra0_bike, km, cfg["bike"]["bike_infra_km0"]))
    f_park = (
        mix["area"] * saturate(bike_park_ha[km], ha0)
        + mix["park_dist"] * decay(d_park_bike, r)
        + mix["water_dist"] * decay(d_water, r)
    )
    infra_m = weighted_infra(hex_m, hexes["bike_infra_m"].to_numpy(), r)
    bike_infra[km] = infra_m
    f_infra = saturate(infra_m / 1000.0, infra0)
    d_tr, n_ln = unique_lines_in_radius(hex_m, trains_m, r)
    f_tr = 0.5 * decay(d_tr, r) + 0.5 * np.clip(n_ln / cfg["bike"]["train_lines_cap"], 0.0, 1.0)
    bike_trains[km] = (d_tr, n_ln)
    f_shop = saturate(
        decay_sum_points(hex_m, bike_repair, r) + decay_sum_points(hex_m, bike_shop, r),
        cfg["bike"]["shop_repair_sat"],
    )
    f_parkng = saturate(decay_sum_points(hex_m, bike_parking, r), cfg["bike"]["parking_sat"])
    shared = (
        w_bike["bike_infra"] * f_infra
        + w_bike["low_stress"] * f_low
        + w_bike["park_water"] * f_park
        + w_bike["high_stress_penalty"] * f_bike_stress
        + w_bike["bike_shop_repair"] * f_shop
        + w_bike["bike_parking"] * f_parkng
    )
    bike_scores[km] = 100 * (shared + w_bike["trains"] * f_tr)
    bike_scores_nt[km] = 100 * (
        w_bike_nt["bike_infra"] * f_infra
        + w_bike_nt["low_stress"] * f_low
        + w_bike_nt["park_water"] * f_park
        + w_bike_nt["high_stress_penalty"] * f_bike_stress
        + w_bike_nt["bike_shop_repair"] * f_shop
        + w_bike_nt["bike_parking"] * f_parkng
    )

score_run = run_scores[run_default]
score_bike = bike_scores[bike_default]
park_ha_run = run_park_ha[run_default]
park_ha_bike = bike_park_ha[bike_default]
infra_weighted_m = bike_infra[bike_default]
d_train, n_lines = bike_trains[bike_default]
print(f"   дефолт: бег {run_default} км, вело {bike_default} км")

print("4. Сборка GeoJSON...")
out = hexes[["h3_index", "geometry"]].copy()
out["nearby_places"] = nearby_names(hexes, parks)
out["score_run"] = np.clip(np.round(score_run), 0, 100).astype(int)
out["score_bike"] = np.clip(np.round(score_bike), 0, 100).astype(int)
for km in run_kms:
    out[f"score_run_{km}"] = np.clip(np.round(run_scores[km]), 0, 100).astype(int)
    out[f"park_ha_run_{km}"] = np.round(run_park_ha[km], 2)
for km in bike_kms:
    out[f"score_bike_{km}"] = np.clip(np.round(bike_scores[km]), 0, 100).astype(int)
    out[f"score_bike_{km}_nt"] = np.clip(np.round(bike_scores_nt[km]), 0, 100).astype(int)
    out[f"park_ha_bike_{km}"] = np.round(bike_park_ha[km], 2)
    out[f"bike_infra_km_{km}"] = np.round(bike_infra[km] / 1000.0, 2)
    out[f"dist_train_m_{km}"] = np.round(np.clip(bike_trains[km][0], 0, 1e6), 0).astype(int)
    out[f"train_lines_n_{km}"] = bike_trains[km][1].astype(int)
out["park_ha_run"] = np.round(park_ha_run, 2)
out["park_ha_bike"] = np.round(park_ha_bike, 2)
out["dist_park_m"] = np.round(np.clip(d_park_show, 0, 1e6), 0).astype(int)
out["dist_water_m"] = np.round(np.clip(d_water_show, 0, 1e6), 0).astype(int)
out["dist_train_m"] = np.round(np.clip(d_train, 0, 1e6), 0).astype(int)
out["train_lines_n"] = n_lines.astype(int)
out["run_friendly_share"] = np.round(hexes["run_friendly_share"], 3)
out["high_stress_share"] = np.round(hexes["high_stress_share"], 3)
out["low_stress_share"] = np.round(hexes["low_stress_share"], 3)
out["bike_infra_km_r"] = np.round(infra_weighted_m / 1000.0, 2)

geo_path = os.path.join(DATA, "moscow_h3_scored.geojson")
out_wgs = out.to_crs("EPSG:4326")
tmp_path = geo_path + ".tmp"
out_wgs.to_file(tmp_path, driver="GeoJSON")
try:
    os.replace(tmp_path, geo_path)
except PermissionError:
    geo_path = os.path.join(DATA, "moscow_h3_scored_new.geojson")
    os.replace(tmp_path, geo_path)
    print("   старый geojson занят, записала в moscow_h3_scored_new.geojson")
print(f"   {geo_path}, гексов: {len(out)}")
cols = ["score_run", "score_bike"] + [f"score_run_{k}" for k in run_kms] + [
    f"score_bike_{k}" for k in bike_kms
] + [f"score_bike_{k}_nt" for k in bike_kms]
print(out[cols].describe().round(2).to_string())
print("\nТоп-8 бег:")
print(
    out.nlargest(8, "score_run")[
        ["h3_index", "score_run", "nearby_places", "park_ha_run", "run_friendly_share"]
    ].to_string(index=False)
)
print("\nТоп-8 вело:")
print(
    out.nlargest(8, "score_bike")[
        ["h3_index", "score_bike", "nearby_places", "bike_infra_km_r", "train_lines_n"]
    ].to_string(index=False)
)
