"""ActiveHabitat choropleth: H3 r8 hex scores, run/bike toggle, sidebar."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from llmaps import Map
from llmaps.components import Controls, Legend, Sidebar
from llmaps.expressions import compute_color_stops, feature_state_color
from llmaps.layers import FillLayer
from llmaps.sources import FileSource

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "vibe-map" / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SKILL_SCRIPTS))

from hex_enrich import (  # noqa: E402
    DATA,
    ensure_housing_ha,
    ensure_park_near,
    ensure_rail_km,
    ensure_road_mix,
    nearby_names,
    nearest_dist,
)

SOURCE_GEOJSON = DATA / "moscow_h3_scored.geojson"
_SCORED_NEW = DATA / "moscow_h3_scored_new.geojson"
if _SCORED_NEW.exists() and (
    not SOURCE_GEOJSON.exists()
    or _SCORED_NEW.stat().st_mtime > SOURCE_GEOJSON.stat().st_mtime
):
    SOURCE_GEOJSON = _SCORED_NEW
PREPARED_GEOJSON = Path(__file__).resolve().parent / "_hexes.geojson"
OUTPUT_HTML = Path(__file__).resolve().parent / "map.html"

SOURCE_ID = "hex"
LAYER_ID = "hex-fill"

# Километраж задаёт радиус поиска как L/4 — та же пропорция, что уже зашита в
# 06_sample_loops (5 км -> 1250 м, 20 км -> 5000 м).
KM_OPTIONS = {"run": [5, 10, 15], "bike": [10, 20, 30]}
KM_DEFAULT = {"run": 5, "bike": 20}
ROUTE_KM = {"run": [5, 10, 15], "bike": [10, 20, 30]}

RDYLGN_RED_LOW_7 = [
    "#d73027",
    "#fc8d59",
    "#f0b429",  # было #fee08b — жёлтый ближе к янтарному, меньше слива с OSM
    "#e8d24a",  # было #ffffbf — бледно-жёлтый читался как подложка
    "#d9ef8b",
    "#91cf60",
    "#1a9850",
]

def radius_label(km: int) -> str:
    text = f"{km / 4:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{text} км"


SIDEBAR_FIELDS = [
    "parks_near",
    "water_near",
    "park_ha_run",
    "dist_park_m",
    "dist_water_m",
    "dist_train_m",
    "train_lines_n",
    "dist_shop_m",
    "dist_repair_m",
    "low_stress_pct",
    "bike_infra_km_r",
]

FIELD_LABELS = {
    "parks_near": "Парки поблизости",
    "water_near": "Водоёмы поблизости",
    "park_ha_run": "Парки в радиусе 1,25 км",
    "park_ha_bike": "Парки в радиусе 3,75 км",
    "dist_park_m": "До парка пешком",
    "dist_water_m": "До воды пешком",
    "dist_train_m": "До электрички по прямой",
    "train_lines_n": "Веток электрички",
    "dist_shop_m": "До веломагазина по прямой",
    "dist_repair_m": "До велосервиса по прямой",
    "low_stress_pct": "Спокойные улицы",
    "bike_infra_km_r": "Велодорожки в радиусе 3,75 км",
}
for _km in KM_OPTIONS["run"]:
    FIELD_LABELS[f"park_ha_run_{_km}"] = f"Парки в радиусе {radius_label(_km)}"
for _km in KM_OPTIONS["bike"]:
    FIELD_LABELS[f"park_ha_bike_{_km}"] = f"Парки в радиусе {radius_label(_km)}"
    FIELD_LABELS[f"bike_infra_km_{_km}"] = f"Велодорожки в радиусе {radius_label(_km)}"
    FIELD_LABELS[f"dist_train_m_{_km}"] = "До электрички по прямой"
    FIELD_LABELS[f"train_lines_n_{_km}"] = "Веток электрички"

RUN_LABELS = [
    FIELD_LABELS["parks_near"],
    FIELD_LABELS["water_near"],
    FIELD_LABELS["park_ha_run"],
    FIELD_LABELS["dist_park_m"],
    FIELD_LABELS["dist_water_m"],
]
BIKE_LABELS = [
    FIELD_LABELS["parks_near"],
    FIELD_LABELS["water_near"],
    FIELD_LABELS["park_ha_bike"],
    FIELD_LABELS["dist_park_m"],
    FIELD_LABELS["dist_water_m"],
    FIELD_LABELS["dist_train_m"],
    FIELD_LABELS["train_lines_n"],
    FIELD_LABELS["dist_shop_m"],
    FIELD_LABELS["dist_repair_m"],
    FIELD_LABELS["low_stress_pct"],
    FIELD_LABELS["bike_infra_km_r"],
]


def _pct(share) -> int:
    if share is None:
        return 0
    return int(round(float(share) * 100))


def _fmt_qty(value, unit: str, ndigits: int = 1) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    scale = 10 ** ndigits
    rounded = round(number * scale) / scale
    if abs(rounded - round(rounded)) < 10 ** (-ndigits - 1):
        text = str(int(round(rounded)))
    else:
        text = f"{rounded:.{ndigits}f}".replace(".", ",")
    return f"{text} {unit}"


def _fmt_dist(meters, missing: str) -> str:
    if meters is None:
        return missing
    try:
        value = float(meters)
    except (TypeError, ValueError):
        return missing
    if value >= 999999:
        return missing
    return f"{int(round(value))} м"


def _walk_coords(coords, xs: list, ys: list) -> None:
    if isinstance(coords[0], (int, float)):
        xs.append(coords[0])
        ys.append(coords[1])
        return
    for item in coords:
        _walk_coords(item, xs, ys)


def prepare_geojson(src: Path, dest: Path):
    print("Готовлю свойства гексов...")
    with src.open(encoding="utf-8") as f:
        geojson = json.load(f)

    hexes = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    hexes = hexes[["h3_index", "geometry"]].copy()
    hex_m = hexes.to_crs("EPSG:32637")

    parks = gpd.read_file(DATA / "parks_and_water.gpkg")
    park_poly = parks[parks["category"] == "park"]
    water_poly = parks[parks["category"] == "water"]
    rivers = gpd.read_file(DATA / "named_rivers.gpkg")
    poi = gpd.read_file(DATA / "active_pois.gpkg").to_crs("EPSG:32637")
    shops = poi[poi["category"] == "bike_shop"]
    repair = poi[poi["category"] == "bike_repair"]

    park_names = nearby_names(hexes, park_poly, n=2)
    water_src = gpd.GeoDataFrame(
        pd.concat(
            [water_poly[["name", "geometry"]], rivers[["name", "geometry"]]],
            ignore_index=True,
        ),
        crs=hexes.crs,
    )
    water_names = nearby_names(hexes, water_src, n=2)
    d_shop = nearest_dist(hex_m, shops)
    d_repair = nearest_dist(hex_m, repair)
    mix = ensure_road_mix().set_index("h3_index")
    rail = ensure_rail_km().set_index("h3_index")
    try:
        housing = ensure_housing_ha().set_index("h3_index")
    except FileNotFoundError as exc:
        print(f"   ВНИМАНИЕ жильё: {exc}")
        housing = None
    park_near = ensure_park_near().set_index("h3_index")

    extras = {
        row.h3_index: {
            "parks_near": park_names[i],
            "water_near": water_names[i],
            "dist_shop_m": d_shop[i],
            "dist_repair_m": d_repair[i],
            "mix_magistral_km": float(mix.loc[row.h3_index, "mix_magistral_km"]) if row.h3_index in mix.index else 0.0,
            "mix_street_km": float(mix.loc[row.h3_index, "mix_street_km"]) if row.h3_index in mix.index else 0.0,
            "mix_walk_km": float(mix.loc[row.h3_index, "mix_walk_km"]) if row.h3_index in mix.index else 0.0,
            "mix_path_km": float(mix.loc[row.h3_index, "mix_path_km"]) if row.h3_index in mix.index else 0.0,
            "mix_bike_km": float(mix.loc[row.h3_index, "mix_bike_km"]) if row.h3_index in mix.index else 0.0,
            "rail_km": float(rail.loc[row.h3_index, "rail_km"]) if row.h3_index in rail.index else 0.0,
            "housing_ha": float(housing.loc[row.h3_index, "housing_ha"]) if housing is not None and row.h3_index in housing.index else 0.0,
            "has_housing": int(housing.loc[row.h3_index, "has_housing"]) if housing is not None and row.h3_index in housing.index else 0,
            "park_near": int(park_near.loc[row.h3_index, "park_near"]) if row.h3_index in park_near.index else 0,
        }
        for i, row in enumerate(hexes.itertuples(index=False))
    }

    score_lists: dict[str, list[float]] = {}
    xs: list[float] = []
    ys: list[float] = []
    for feat in geojson["features"]:
        p = feat["properties"]
        h3 = p["h3_index"]
        extra = extras[h3]
        run_pct = _pct(p.get("run_friendly_share"))
        low_pct = _pct(p.get("low_stress_share"))
        high_pct = _pct(p.get("high_stress_share"))
        props = {
            "h3_index": h3,
            "score_run": int(p["score_run"]),
            "score_bike": int(p["score_bike"]),
            "parks_near": extra["parks_near"],
            "water_near": extra["water_near"],
            "park_ha_run": _fmt_qty(p.get("park_ha_run"), "га"),
            "park_ha_bike": _fmt_qty(p.get("park_ha_bike"), "га"),
            "dist_park_m": _fmt_dist(p.get("dist_park_m"), "нет рядом"),
            "dist_water_m": _fmt_dist(p.get("dist_water_m"), "нет рядом"),
            "dist_train_m": _fmt_dist(p.get("dist_train_m"), "нет в 3,75 км"),
            "train_lines_n": p.get("train_lines_n"),
            "dist_shop_m": _fmt_dist(extra["dist_shop_m"], "нет рядом"),
            "dist_repair_m": _fmt_dist(extra["dist_repair_m"], "нет рядом"),
            "run_friendly_pct": run_pct,
            "low_stress_pct": _fmt_qty(low_pct, "%", ndigits=0),
            "high_stress_pct": high_pct,
            "bike_infra_km_r": _fmt_qty(p.get("bike_infra_km_r"), "км"),
            # сырые доли/числа для why-тегов в браузере
            "run_friendly_share": float(p.get("run_friendly_share") or 0),
            "low_stress_share": float(p.get("low_stress_share") or 0),
            "high_stress_share": float(p.get("high_stress_share") or 0),
            "n_park_ha_run": float(p.get("park_ha_run") or 0),
            "n_park_ha_bike": float(p.get("park_ha_bike") or 0),
            "n_dist_park_m": float(p.get("dist_park_m") or 1e6),
            "n_dist_water_m": float(p.get("dist_water_m") or 1e6),
            "n_dist_train_m": float(p.get("dist_train_m") or 1e6),
            "n_bike_infra_km_r": float(p.get("bike_infra_km_r") or 0),
            "mix_magistral_km": extra["mix_magistral_km"],
            "mix_street_km": extra["mix_street_km"],
            "mix_walk_km": extra["mix_walk_km"],
            "mix_path_km": extra["mix_path_km"],
            "mix_bike_km": extra["mix_bike_km"],
            "rail_km": round(extra["rail_km"], 3),
            "n_rail_km": float(extra["rail_km"]),
            "housing_ha": round(extra["housing_ha"], 4),
            "has_housing": int(extra["has_housing"]),
            "park_near": int(extra["park_near"]),
        }
        for key, val in p.items():
            if key.startswith("score_") and val is not None:
                props[key] = int(val)
                score_lists.setdefault(key, []).append(float(val))
            elif key.startswith("park_ha_"):
                props[key] = _fmt_qty(val, "га")
                props["n_" + key] = float(val or 0)
            elif key.startswith("bike_infra_km_"):
                props[key] = _fmt_qty(val, "км")
                props["n_" + key] = float(val or 0)
            elif key.startswith("dist_train_m_"):
                props[key] = _fmt_dist(val, "нет в радиусе")
                props["n_" + key] = float(val if val is not None else 1e6)
            elif key.startswith("train_lines_n_"):
                props[key] = val
        feat["properties"] = props
        _walk_coords(feat["geometry"]["coordinates"], xs, ys)

    dest.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    bounds = [[min(xs), min(ys)], [max(xs), max(ys)]]
    return score_lists, bounds


def quantile_stops(values: list[float]):
    return compute_color_stops(
        values,
        method="quantile",
        n_stops=7,
        colors=RDYLGN_RED_LOW_7,
        precision=0,
    )


def load_weights() -> dict:
    path = ROOT / "pipeline" / "scoring_weights.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {p: cfg[p]["weights"] for p in ("run", "bike")}


def load_why_cfg() -> dict:
    path = ROOT / "pipeline" / "scoring_weights.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "run": {
            "park_ha0_by_km": cfg["run"].get("park_ha0_by_km") or {},
            "park_ha0": cfg["run"]["park_ha0"],
        },
        "bike": {
            "park_ha0_by_km": cfg["bike"].get("park_ha0_by_km") or {},
            "park_ha0": cfg["bike"]["park_ha0"],
            "bike_infra_km0_by_km": cfg["bike"].get("bike_infra_km0_by_km") or {},
            "bike_infra_km0": cfg["bike"]["bike_infra_km0"],
            "park_mix": cfg["bike"]["park_mix"],
            "train_lines_cap": cfg["bike"]["train_lines_cap"],
        },
    }


def main(out_html: Path | None = None) -> None:
    out_html = out_html or OUTPUT_HTML
    score_lists, bounds = prepare_geojson(SOURCE_GEOJSON, PREPARED_GEOJSON)
    ramps = {key: quantile_stops(vals) for key, vals in score_lists.items() if vals}
    stops_run = ramps.get("score_run") or ramps.get("score_run_5") or []
    stops_bike = ramps.get("score_bike") or ramps.get("score_bike_20") or []

    source = FileSource(
        id=SOURCE_ID,
        path=str(PREPARED_GEOJSON),
        promote_id="h3_index",
    )
    layer = FillLayer(
        id=LAYER_ID,
        source=source,
        fill_color=feature_state_color(
            state_key="active",
            color_ramp_key="value",
            color_stops=stops_run,
            inactive="#F0F0F0",
            default="#E0E0E0",
        ),
        fill_opacity=0.38,
        stroke_color="#8B5A2B",
        stroke_width=1.01,
        feature_state={"active": True, "value": "score_run"},
    )

    legend = Legend(
        position="bottom-left",
        collapsed=False,
        show_toggle=True,
        description="Красный — хуже, зелёный — лучше",
        layer_labels={LAYER_ID: "Балл района"},
        layer_color_ramps={
            LAYER_ID: {
                "stops": [[v, c] for v, c in stops_run],
                "label_min": f"{stops_run[0][0]:.0f} · хуже",
                "label_max": f"{stops_run[-1][0]:.0f} · лучше",
            }
        },
    )

    sidebar = Sidebar(
        position="right",
        width=360,
        title_field="h3_index",
        fields_by_layer={LAYER_ID: SIDEBAR_FIELDS},
        field_labels=FIELD_LABELS,
        hide_empty_fields=True,
        show_on_click=True,
        close_on_map_click=True,
    )

    m = Map(
        center=[37.62, 55.75],
        zoom=11,
        title="ActiveHabitat",
        tiles="osm",
        locale="ru-RU",
        embedded=True,
        use_compression=True,
    )
    m.add_layer(layer)
    m.add_component(legend)
    m.add_component(sidebar)
    m.add_component(Controls(zoom=True, scale=True, fullscreen=True, hash=False))
    m.embed_data("extent", {"bounds": bounds})
    m.embed_data("profileFields", {"run": RUN_LABELS, "bike": BIKE_LABELS})
    # Балл собираем в браузере из факторов, поэтому веса и палитра едут в карту:
    # «нужна ли электричка» — это снятие веса trains и перенормировка остальных,
    # никаких новых метрик для этого считать не надо.
    m.embed_data("weights", load_weights())
    m.embed_data("rampColors", RDYLGN_RED_LOW_7)
    m.embed_data("kmOptions", {"options": KM_OPTIONS, "default": KM_DEFAULT})
    m.embed_data("routeKm", ROUTE_KM)
    m.embed_data("whyCfg", load_why_cfg())
    m.embed_data(
        "ramps",
        {key: [[v, c] for v, c in stops] for key, stops in ramps.items()},
    )

    m.add_custom_css(
        """
.llmaps-legend,
.ah-chrome {
  background: rgba(255, 255, 255, 0.38);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(229, 231, 235, 0.85);
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.ah-chrome {
  position: absolute;
  z-index: 10;
  top: 12px;
  left: 54px;
  padding: 10px 12px;
  max-width: 320px;
  font-size: 12px;
  color: #111827;
}
.ah-chrome h1 {
  margin: 0 0 4px;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.2;
}
.ah-chrome p {
  margin: 0 0 8px;
  line-height: 1.35;
  color: rgba(17, 24, 39, 0.72);
}
.ah-btns { display: flex; gap: 6px; }
.ah-btns button {
  font-size: 12px;
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid rgba(229, 231, 235, 0.85);
  background: rgba(255, 255, 255, 0.35);
  color: #111827;
  cursor: pointer;
}
.ah-btns button.is-on {
  background: rgba(0, 0, 0, 0.16);
  font-weight: 600;
}
/* Настройки маршрута. Живут рядом с профилем, потому что меняют смысл всей
   карты, а не выбранного гекса. */
.ah-opts { margin-top: 9px; }
.ah-opts:empty { display: none; }
.ah-opt-row + .ah-opt-row { margin-top: 7px; }
.ah-opt-cap {
  display: block;
  margin-bottom: 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: rgba(17, 24, 39, 0.62);
}
.ah-chips { display: flex; gap: 4px; }
.ah-chips button {
  flex: 1 1 0;
  min-width: 0;
  font-size: 12px;
  padding: 5px 6px;
  border-radius: 7px;
  border: 1px solid rgba(229, 231, 235, 0.85);
  background: rgba(255, 255, 255, 0.35);
  color: #111827;
  cursor: pointer;
  white-space: nowrap;
}
.ah-chips button.is-on {
  background: rgba(107, 63, 29, 0.14);
  border-color: rgba(107, 63, 29, 0.45);
  font-weight: 600;
}
.ah-switch {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  cursor: pointer;
  line-height: 1.3;
}
.ah-switch input { margin: 0; cursor: pointer; }
/* Выкл. состояние тогла «Балл района» — заметнее на полупрозрачном фоне. */
.llmaps-toggle-slider {
  background-color: #9ca3af;
  box-shadow: inset 0 0 0 1px rgba(17, 24, 39, 0.22);
}
.llmaps-layer-toggle:hover .llmaps-toggle-slider {
  background-color: #6b7280;
}
.ah-filters {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(229, 231, 235, 0.9);
  max-width: 100%;
  box-sizing: border-box;
}
.ah-filters-title {
  margin: 0 0 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(17, 24, 39, 0.55);
}
.ah-score-ends {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 11px;
  color: rgba(17, 24, 39, 0.55);
  margin-bottom: 2px;
}
.ah-ball-static {
  font-size: 12px;
  font-weight: 600;
  color: rgba(17, 24, 39, 0.62);
  letter-spacing: 0.02em;
}
.ah-ramp-wrap {
  position: relative;
  height: 44px;
  margin: 0 0 4px;
  --lo: 0%;
  --hi: 100%;
  touch-action: none;
  cursor: grab;
  user-select: none;
  width: 100%;
  box-sizing: border-box;
}
.ah-ramp-wrap.is-dragging { cursor: grabbing; }
/* Отступ = радиус бегунка + запас под обводку, иначе .llmaps-legend {overflow:hidden} срезает край. */
.ah-ramp-inner {
  position: absolute;
  left: 12px;
  right: 12px;
  top: 0;
  bottom: 0;
  pointer-events: none;
}
.ah-ramp-full {
  position: absolute;
  left: 0; right: 0; top: 10px; height: 10px;
  border-radius: 5px;
  background: linear-gradient(to right, #d73027, #fc8d59, #f0b429, #e8d24a, #d9ef8b, #91cf60, #1a9850);
}
.ah-ramp-gray-l, .ah-ramp-gray-r {
  position: absolute; top: 10px; height: 10px;
  background: #d1d5db;
  z-index: 1;
}
.ah-ramp-gray-l { left: 0; width: var(--lo); border-radius: 5px 0 0 5px; }
.ah-ramp-gray-r { right: 0; width: calc(100% - var(--hi)); border-radius: 0 5px 5px 0; }
.ah-thumb {
  position: absolute;
  top: 15px;
  width: 18px;
  height: 18px;
  margin-left: -9px;
  border-radius: 50%;
  background: #fff;
  border: 2px solid #6B3F1D;
  box-shadow: 0 1px 4px rgba(0,0,0,.22);
  transform: translateY(-50%);
  z-index: 3;
}
.ah-float-val {
  position: absolute;
  top: 28px;
  transform: translateX(-50%);
  font-size: 12px;
  font-weight: 400;
  color: rgba(17, 24, 39, 0.55);
  font-variant-numeric: tabular-nums;
  z-index: 3;
  white-space: nowrap;
}
.ah-f-check {
  display: flex; align-items: flex-start; gap: 8px;
  margin: 7px 0; font-size: 13px; cursor: pointer; color: #111827;
  max-width: 100%;
  box-sizing: border-box;
  white-space: normal;
  overflow-wrap: anywhere;
  line-height: 1.35;
}
.ah-f-check input { margin: 2px 0 0; cursor: pointer; flex: none; }
.ah-f-check.is-hidden { display: none; }
.ah-f-count {
  margin: 8px 0 0;
  font-size: 11px;
  color: rgba(17, 24, 39, 0.55);
}
.llmaps-legend-ramp-labels { display: none !important; }
.llmaps-legend .llmaps-legend-ramp { display: none !important; }
.llmaps-legend:not(.collapsed) {
  width: 280px;
  max-width: 280px;
  box-sizing: border-box;
}
/* Никакой анимации размеров и никакого height на панели —
   иначе 40px→auto интерполируется к высоте viewport/#map-root. */
.llmaps-legend {
  transition: none !important;
  height: auto !important;
  max-height: none !important;
}
.llmaps-legend.collapsed {
  height: auto !important;
  max-height: none !important;
}
.llmaps-legend-content {
  max-width: 100%;
  box-sizing: border-box;
}
.llmaps-sidebar-header {
  align-items: flex-start;
}
.llmaps-sidebar-title {
  flex: 1;
  font-weight: 400;
}
.ah-score {
  font-size: 40px;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1;
  color: #111827;
}
.ah-profile {
  margin: 4px 0 0;
  font-size: 13px;
  color: #6b7280;
}
.ah-why {
  margin: 10px 0 0;
  font-size: 13px;
  line-height: 1.4;
  color: #374151;
}
.llmaps-sidebar-title .ah-why {
  flex: 1 0 100%;
  margin-top: 6px;
}
.ah-mix {
  margin: 0 20px 8px;
  padding: 12px 0 10px;
  border-bottom: 1px solid #f3f4f6;
}
.ah-mix-title,
.ah-facts-title {
  font-size: 14px;
  font-weight: 700;
  color: #111827;
  letter-spacing: -0.01em;
}
.ah-mix-title { margin: 2px 0 10px; }
.ah-facts-title { margin: 0 0 6px; }
.ah-ico {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  display: block;
}
.ah-mix-hint {
  margin: 0 0 10px;
  font-size: 11px;
  line-height: 1.35;
  color: #9ca3af;
}
.ah-mix-row {
  display: grid;
  grid-template-columns: 16px 1fr 80px 64px;
  gap: 8px;
  align-items: center;
  margin: 5px 0;
  font-size: 12px;
  color: #4b5563;
}
.ah-mix-row-label { min-width: 0; }
.ah-mix-row-track {
  height: 8px;
  border-radius: 8px;
  background: #f3f4f6;
  overflow: hidden;
}
.ah-mix-row-track span {
  display: block;
  height: 100%;
  border-radius: 8px;
}
.ah-mix-row-val {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: #111827;
}
.llmaps-sidebar-content.ah-facts {
  padding-top: 8px;
  padding-bottom: 12px;
}
.llmaps-sidebar-content.ah-facts .llmaps-sidebar-section {
  margin-bottom: 0;
}
.llmaps-sidebar-content.ah-facts .llmaps-sidebar-field {
  align-items: flex-start;
  justify-content: flex-start;
  gap: 20px;
  padding: 5px 0;
  font-size: 12px;
  line-height: 1.35;
}
.llmaps-sidebar-content.ah-facts .llmaps-sidebar-field-label {
  display: inline-flex;
  align-items: flex-start;
  gap: 8px;
  flex: 0 0 10.4em;
  max-width: 10.4em;
  min-width: 10.4em;
  margin-right: 0;
  font-size: 12px;
}
.llmaps-sidebar-content.ah-facts .llmaps-sidebar-field-value {
  flex: 1 1 0;
  min-width: 0;
  text-align: right;
  line-height: 1.35;
}
.llmaps-legend.bottom-left {
  left: 12px;
  bottom: 40px;
}
.llmaps-legend-instructions {
  display: none;
}
.llmaps-legend-icon {
  display: none;
}
/* иконку слоя мы скрыли, а отступы под неё остались: из-за них шкала уезжала
   правее подписей. Сводим заголовок, описание и шкалу на один левый край. */
.llmaps-legend-header {
  padding: 8px 12px 6px;
}
.llmaps-legend-header:hover {
  background: transparent;
}
.llmaps-legend-title {
  font-size: 13px;
  line-height: 1.2;
}
.llmaps-legend-layer-header {
  padding-left: 0;
  padding-right: 0;
  gap: 8px;
}
.llmaps-legend-layer-header:hover {
  background: transparent;
}
.llmaps-legend-layer-header,
.llmaps-layer-toggle {
  -webkit-tap-highlight-color: transparent;
}
.llmaps-legend-item-label {
  flex: 1;
}
.llmaps-legend-description {
  margin-left: 0;
}
.llmaps-legend-ramp,
.llmaps-legend-ramp-labels {
  margin-left: 0;
  width: 100%;
}
.ah-sheet-handle {
  display: none;
}
.ah-tabs {
  display: flex;
  gap: 3px;
  padding: 0 16px;
  margin-bottom: 10px;
  border-bottom: 1px solid rgba(229, 231, 235, 0.9);
}
.ah-tab {
  flex: 1;
  padding: 9px 8px;
  border: 0;
  background: transparent;
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  color: #6b7280;
  cursor: pointer;
  border-radius: 9px 9px 0 0;
}
.ah-tab.is-on {
  color: #111827;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: inset 0 -2px 0 #111827;
}
.ah-routes { padding: 0 16px 12px; }
.ah-route {
  border: 1px solid rgba(229, 231, 235, 0.9);
  border-left: 4px solid var(--edge);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.72);
  margin-bottom: 8px;
  overflow: hidden;
}
.ah-route.sel { background: #fff; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07); }
.ah-route-head { display: flex; align-items: center; gap: 10px; padding: 9px 11px; cursor: pointer; }
.ah-glyph { flex: none; opacity: 0.85; }
.ah-route-title { font-weight: 600; font-size: 13px; }
.ah-route-sum { color: #6b7280; font-size: 12px; }
.ah-chev { margin-left: auto; color: #6b7280; }
.ah-route.sel .ah-chev { transform: rotate(90deg); }
.ah-route-body { padding: 2px 11px 12px; }
.ah-sub-h {
  margin: 12px 0 6px;
  font-size: 11px;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.ah-stack { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: rgba(0, 0, 0, 0.06); }
.ah-stack i { display: block; }
.ah-keys { display: flex; flex-wrap: wrap; gap: 3px 14px; margin-top: 7px; font-size: 12px; color: #6b7280; }
.ah-keys b { color: #111827; font-variant-numeric: tabular-nums; }
.ah-dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }
.ah-kv {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 5px 0;
  font-size: 12px;
  border-bottom: 1px dashed rgba(229, 231, 235, 0.9);
}
.ah-kv:last-child { border-bottom: 0; }
.ah-kv span { color: #6b7280; }
.ah-kv b { font-weight: 600; font-variant-numeric: tabular-nums; }
.ah-verdict {
  font-size: 12px;
  padding: 6px 9px;
  border-radius: 8px;
  background: rgba(201, 79, 71, 0.09);
  color: #8a332d;
  margin: 8px 0 2px;
}
.ah-verdict.ok { background: rgba(47, 125, 79, 0.1); color: #1f5c39; }
.ah-soon {
  margin-top: 12px;
  padding: 9px 10px;
  border: 1px dashed rgba(229, 231, 235, 0.9);
  border-radius: 9px;
  color: #6b7280;
  font-size: 12px;
}
.ah-soon b { color: #111827; }
.ah-elev { margin-top: 12px; }
.ah-elev-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin: 12px 0 6px;
}
.ah-elev-head .ah-sub-h { margin: 0; }
.ah-elev-mini { position: relative; }
.ah-elev-svg { display: block; width: 100%; height: 56px; }
.ah-elev-axis {
  display: flex;
  justify-content: space-between;
  margin-top: 3px;
  color: #9ca3af;
  font-size: 11px;
}
.ah-elev-nums {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  margin-top: 6px;
  color: #6b7280;
  font-size: 12px;
}
.ah-elev-nums b { color: #111827; font-variant-numeric: tabular-nums; }
.ah-elev-dir { white-space: nowrap; }
.ah-elev-dir i {
  display: inline-block;
  width: 0.85em;
  font-style: normal;
  font-size: 11px;
  line-height: 1;
}
.ah-elev-open {
  width: 22px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border: 1px solid rgba(107, 63, 29, 0.35);
  background: #fff;
  border-radius: 6px;
  padding: 0;
  color: #6B3F1D;
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.ah-elev-open svg { width: 13px; height: 13px; display: block; }
.ah-elev-open:hover { border-color: rgba(107, 63, 29, 0.55); background: #faf7f4; }
.ah-elev-modal {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(17, 24, 39, 0.42);
}
.ah-elev-modal.is-on { display: flex; }
.ah-elev-card {
  background: #fff;
  border-radius: 16px;
  padding: 18px 20px 16px;
  width: min(860px, 100%);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
}
.ah-elev-card header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.ah-elev-card h3 { margin: 0; font-size: 16px; }
.ah-elev-full { display: block; width: 100%; height: auto; aspect-ratio: 720 / 148; }
.ah-elev-close {
  margin-left: auto;
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 6px;
  background: none;
  color: #6b7280;
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.ah-elev-close:hover {
  background: #e5e7eb;
  color: #0a0a0a;
}
@media (max-width: 768px) {
  .ah-chrome {
    left: 54px;
    top: 12px;
    bottom: auto;
    max-width: min(100vw - 24px, 320px);
  }
  .ah-chrome p { display: none; }
  .llmaps-legend {
  max-width: min(100vw - 24px, 300px);
  font-size: 12px;
}
  .llmaps-legend.bottom-left {
    bottom: 36px;
  }
  .llmaps-legend-toggle-btn { min-width: 44px; min-height: 44px; }
  .llmaps-sidebar-header {
    padding: 2px 12px 8px 16px;
    align-items: center;
  }
  .llmaps-sidebar-title {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
  }
  .ah-score {
    font-size: 22px;
    line-height: 1;
  }
  .ah-profile {
    margin: 0;
    font-size: 13px;
  }
  .llmaps-sidebar-title .ah-why {
    margin-top: 4px;
    font-size: 12px;
  }
  .ah-sheet-handle {
    display: block;
    padding: 14px 0 12px;
    cursor: grab;
    touch-action: none;
    flex-shrink: 0;
  }
  .ah-sheet-handle i {
    display: block;
    width: 48px;
    height: 5px;
    margin: 0 auto;
    border-radius: 99px;
    background: #d1d5db;
  }
  .llmaps-sidebar,
  .llmaps-sidebar.right {
    top: auto !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    height: 42vh !important;
    max-height: 100vh;
    border-radius: 18px 18px 0 0;
    border: 0 !important;
    box-shadow: 0 -8px 28px rgba(0, 0, 0, 0.18);
    transform: translateY(110%) !important;
  }
  .llmaps-sidebar.open {
    transform: translateY(0) !important;
  }
}
@media (min-width: 769px) {
  .llmaps-legend { max-width: 280px; }
}
"""
    )
    m.add_custom_html(
        """
<div class="ah-chrome" role="group" aria-label="Профиль">
  <h1>ActiveHabitat</h1>
  <p>Где в Москве удобнее бегать и ездить на велосипеде</p>
  <div class="ah-btns">
    <button type="button" data-ah-profile="run" class="is-on">Бег</button>
    <button type="button" data-ah-profile="bike">Вело</button>
  </div>
  <div class="ah-opts" id="ah-opts"></div>
</div>
<div class="ah-elev-modal" id="ah-elev-modal" hidden></div>
"""
    )
    m.add_custom_js(
        """
(function () {
  var SOURCE_ID = "hex";
  var LAYER_ID = "hex-fill";
  var FIELD = { run: "score_run", bike: "score_bike" };
  var PROFILE_NAME = { run: "Бег", bike: "Вело" };
  // показатели независимы и в целое не складываются: у улицы обычно есть тротуар
  var MIX = [
    { key: "mix_magistral_km", label: "Магистрали", color: "#c0564e", icon: "hwy" },
    { key: "mix_street_km", label: "Улицы", color: "#c9a05b", icon: "street" },
    { key: "mix_walk_km", label: "Тротуары", color: "#93a3ae", icon: "walk" },
    { key: "mix_path_km", label: "Тропы в парках", color: "#4f8a5b", icon: "path" },
    { key: "mix_bike_km", label: "Велодорожки", color: "#3f7fb5", icon: "bike" }
  ];
  // доли покрытия дают 100%, поэтому здесь составная полоса уместна
  var SURF = [
    { key: "surf_asphalt", label: "асфальт", color: "#7c8794" },
    { key: "surf_stone", label: "брусчатка", color: "#c2410c" },
    { key: "surf_soil", label: "грунт", color: "#8B5A2B" }
  ];
  // коричневый из палитры карты; белая подложка отбивает его от любой заливки гекса
  var SELECT_COLOR = "#6B3F1D";
  var SELECT_CASING_COLOR = "#ffffff";
  var SELECT_SOURCE = "hex-sel";
  var SELECT_LAYER = "hex-sel-line";
  var SELECT_CASING = "hex-sel-casing";
  var LOOP_SOURCE = "loops";
  var LOOP_CASING = "loops-casing";
  var LOOP_LINE = "loops-line";
  var LOOP_ARROWS = "loops-arrows";
  var LOOP_ARROW_SRC = "loop-arrows";
  var LOOP_START_SRC = "loop-start";
  var LOOP_START = "loop-start-pt";
  var LOOP_COLOR = ["#2f7d4f", "#c94f47"];
  var currentProfile = "run";
  var currentKm = { run: 5, bike: 20 };
  var wantTrain = false;
  var filterState = {
    scoreMin: 0,
    scoreMax: 100,
    housing: false,
    park: false,
    quiet: false,
    train: false
  };
  var lastProps = null;
  var observer = null;
  var hexGeom = null;
  var hexProps = null;
  var LOOP_HALO = "loops-halo";
  var loopCache = {};
  var loopHex = null;
  var ahTab = "hex";
  var openRank = null;
  var elevOpen = false;

  function kmCfg() {
    return (window.llmapsData && window.llmapsData.kmOptions) || {
      options: { run: [5, 10, 15], bike: [10, 20, 30] },
      default: { run: 5, bike: 20 }
    };
  }

  function fmtRadius(km) {
    var r = km / 4;
    return (Math.round(r * 100) / 100).toLocaleString("ru-RU") + " км";
  }

  function scoreKey() {
    if (currentProfile === "run") return "score_run_" + currentKm.run;
    return "score_bike_" + currentKm.bike + (wantTrain ? "" : "_nt");
  }

  function profileLabels(profile) {
    var km = currentKm[profile] || kmCfg().default[profile];
    var r = fmtRadius(km);
    var labels = [
      "Парки поблизости",
      "Водоёмы поблизости",
      "Парки в радиусе " + r,
      "До парка пешком",
      "До воды пешком"
    ];
    if (profile === "bike") {
      if (wantTrain) labels.push("До электрички по прямой", "Веток электрички");
      labels.push("До веломагазина по прямой", "До велосервиса по прямой", "Спокойные улицы",
        "Велодорожки в радиусе " + r);
    }
    return labels;
  }

  function renderOpts() {
    var box = document.getElementById("ah-opts");
    if (!box) return;
    var cfg = kmCfg();
    var opts = (cfg.options && cfg.options[currentProfile]) || [5, 10, 15];
    var chips = opts.map(function (km) {
      return '<button type="button" data-ah-km="' + km + '"' +
        (Number(km) === Number(currentKm[currentProfile]) ? ' class="is-on"' : "") +
        ">" + km + " км</button>";
    }).join("");
    box.innerHTML =
      '<div class="ah-opt-row"><span class="ah-opt-cap">Длина маршрута</span>' +
      '<div class="ah-chips">' + chips + "</div></div>";
    syncFilterTrainRow();
  }

  function colorExpr(stops) {
    var interp = ["interpolate", ["linear"], ["feature-state", "value"]];
    for (var i = 0; i < stops.length; i++) {
      interp.push(stops[i][0], stops[i][1]);
    }
    return [
      "case",
      ["==", ["feature-state", "active"], true],
      interp,
      ["==", ["feature-state", "active"], false],
      "#F0F0F0",
      "#E0E0E0"
    ];
  }

  function updateLegend(stops) {
    // подписи хуже/лучше переехали в блок фильтров
    var minEl = document.querySelector(".llmaps-legend-ramp-min");
    var maxEl = document.querySelector(".llmaps-legend-ramp-max");
    if (minEl) minEl.textContent = Math.round(stops[0][0]) + " · хуже";
    if (maxEl) maxEl.textContent = Math.round(stops[stops.length - 1][0]) + " · лучше";
    syncFilterRamp();
  }

  function scoreBounds() {
    var stops = currentStops() || [];
    var vals = stops.map(function (s) { return Number(s[0]); }).filter(isFinite);
    if (!vals.length) return { min: 0, max: 100 };
    return { min: Math.round(vals[0]), max: Math.round(vals[vals.length - 1]) };
  }

  function syncFilterTrainRow() {
    var row = document.getElementById("ah-f-train");
    if (!row) return;
    row.classList.toggle("is-hidden", currentProfile !== "bike");
    if (currentProfile !== "bike") filterState.train = false;
  }

  function syncFilterRamp() {
    var wrap = document.getElementById("ah-ramp-wrap");
    if (!wrap) return;
    var b = scoreBounds();
    var span = Math.max(1, b.max - b.min);
    var loPct = ((filterState.scoreMin - b.min) / span) * 100;
    var hiPct = ((filterState.scoreMax - b.min) / span) * 100;
    loPct = Math.max(0, Math.min(100, loPct));
    hiPct = Math.max(0, Math.min(100, hiPct));
    wrap.style.setProperty("--lo", loPct + "%");
    wrap.style.setProperty("--hi", hiPct + "%");
    var tMin = document.getElementById("ah-thumb-min");
    var tMax = document.getElementById("ah-thumb-max");
    var vMin = document.getElementById("ah-val-min");
    var vMax = document.getElementById("ah-val-max");
    if (tMin) tMin.style.left = loPct + "%";
    if (tMax) tMax.style.left = hiPct + "%";
    // Не даём подписям наезжать друг на друга и вылезать за края (иначе ширина панели пляшет).
    var minGapPct = 14;
    var edgePad = 6;
    var loLab = loPct;
    var hiLab = hiPct;
    if (hiLab - loLab < minGapPct) {
      var mid = (loPct + hiPct) / 2;
      loLab = mid - minGapPct / 2;
      hiLab = mid + minGapPct / 2;
    }
    loLab = Math.max(edgePad, Math.min(100 - edgePad, loLab));
    hiLab = Math.max(edgePad, Math.min(100 - edgePad, hiLab));
    if (hiLab - loLab < minGapPct) {
      if (loLab <= edgePad) hiLab = Math.min(100 - edgePad, loLab + minGapPct);
      else if (hiLab >= 100 - edgePad) loLab = Math.max(edgePad, hiLab - minGapPct);
    }
    if (vMin) {
      vMin.style.left = loLab + "%";
      vMin.textContent = String(filterState.scoreMin);
    }
    if (vMax) {
      vMax.style.left = hiLab + "%";
      vMax.textContent = String(filterState.scoreMax);
    }
  }

  function bindScoreSlider(wrap) {
    var dragging = null; // "min" | "max"
    var lastHandle = "max";

    function trackEl() {
      return wrap.querySelector(".ah-ramp-inner") || wrap;
    }

    function pctFromEvent(ev) {
      var rect = trackEl().getBoundingClientRect();
      if (!rect.width) return 0;
      return Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    }

    function valueFromPct(pct) {
      var b = scoreBounds();
      return Math.round(b.min + pct * (b.max - b.min));
    }

    function pickHandle(pct) {
      var b = scoreBounds();
      var span = Math.max(1, b.max - b.min);
      var lo = (filterState.scoreMin - b.min) / span;
      var hi = (filterState.scoreMax - b.min) / span;
      var dLo = Math.abs(pct - lo);
      var dHi = Math.abs(pct - hi);
      // Наложение: у правого края берём min (чтобы увести влево), у левого — max.
      if (Math.abs(dLo - dHi) < 0.03) {
        if (lo >= 0.98 && hi >= 0.98) return "min";
        if (lo <= 0.02 && hi <= 0.02) return "max";
        return lastHandle;
      }
      return dLo < dHi ? "min" : "max";
    }

    function applyDrag(ev) {
      if (!dragging) return;
      var val = valueFromPct(pctFromEvent(ev));
      if (dragging === "min") {
        filterState.scoreMin = Math.min(val, filterState.scoreMax);
      } else {
        filterState.scoreMax = Math.max(val, filterState.scoreMin);
      }
      lastHandle = dragging;
      if (window._ahSourceData && window.llmaps_map) applyFilters(window._ahSourceData, window.llmaps_map);
      else syncFilterRamp();
    }

    function onPointerDown(ev) {
      if (ev.button != null && ev.button !== 0) return;
      dragging = pickHandle(pctFromEvent(ev));
      lastHandle = dragging;
      wrap.classList.add("is-dragging");
      wrap.setPointerCapture(ev.pointerId);
      applyDrag(ev);
      ev.preventDefault();
    }

    function onPointerMove(ev) {
      if (!dragging) return;
      applyDrag(ev);
    }

    function onPointerUp(ev) {
      if (!dragging) return;
      dragging = null;
      wrap.classList.remove("is-dragging");
      try { wrap.releasePointerCapture(ev.pointerId); } catch (e) {}
    }

    wrap.addEventListener("pointerdown", onPointerDown);
    wrap.addEventListener("pointermove", onPointerMove);
    wrap.addEventListener("pointerup", onPointerUp);
    wrap.addEventListener("pointercancel", onPointerUp);
  }

  function ensureFiltersPanel() {
    var legend = document.querySelector(".llmaps-legend-content");
    if (!legend || document.getElementById("ah-filters")) return;
    var box = document.createElement("div");
    box.id = "ah-filters";
    box.className = "ah-filters";
    box.innerHTML =
      '<div class="ah-filters-title">Фильтры</div>' +
      '<div class="ah-score-ends"><span>хуже</span><span class="ah-ball-static">балл</span><span>лучше</span></div>' +
      '<div class="ah-ramp-wrap" id="ah-ramp-wrap" role="group" aria-label="Диапазон балла">' +
      '<div class="ah-ramp-inner">' +
      '<div class="ah-ramp-full"></div>' +
      '<div class="ah-ramp-gray-l"></div>' +
      '<div class="ah-ramp-gray-r"></div>' +
      '<div class="ah-thumb" id="ah-thumb-min"></div>' +
      '<div class="ah-thumb" id="ah-thumb-max"></div>' +
      '<span class="ah-float-val" id="ah-val-min">0</span>' +
      '<span class="ah-float-val" id="ah-val-max">100</span>' +
      "</div></div>" +
      '<label class="ah-f-check"><input type="checkbox" data-ah-filter="housing">Есть жильё</label>' +
      '<label class="ah-f-check"><input type="checkbox" data-ah-filter="park">Рядом парк</label>' +
      '<label class="ah-f-check"><input type="checkbox" data-ah-filter="quiet">Не у магистрали и ж/д</label>' +
      '<label class="ah-f-check is-hidden" id="ah-f-train">' +
      '<input type="checkbox" data-ah-filter="train">Недалеко от станции электрички</label>' +
      '<p class="ah-f-count" id="ah-f-count"></p>';
    legend.appendChild(box);

    var b = scoreBounds();
    filterState.scoreMin = b.min;
    filterState.scoreMax = b.max;
    syncFilterRamp();
    syncFilterTrainRow();
    bindScoreSlider(document.getElementById("ah-ramp-wrap"));

    box.addEventListener("change", function (ev) {
      var t = ev.target;
      if (!t || !t.getAttribute("data-ah-filter")) return;
      filterState[t.getAttribute("data-ah-filter")] = !!t.checked;
      if (window._ahSourceData && window.llmaps_map) applyFilters(window._ahSourceData, window.llmaps_map);
    });
  }

  function hexPassesFilters(props) {
    if (!props) return true;
    var field = scoreKey();
    var score = Number(props[field]);
    if (!isFinite(score)) score = Number(props[FIELD[currentProfile]]);
    if (!(score >= filterState.scoreMin && score <= filterState.scoreMax)) return false;
    if (filterState.housing) {
      var hh = Number(props.housing_ha || 0);
      var has = Number(props.has_housing || 0) === 1 || hh >= 0.05;
      if (!has) return false;
    }
    if (filterState.park && Number(props.park_near || 0) !== 1) return false;
    if (filterState.quiet) {
      var mag = Number(props.mix_magistral_km || 0);
      var rail = Number(props.n_rail_km != null ? props.n_rail_km : props.rail_km || 0);
      if (!(mag < 0.05 && rail < 0.25)) return false;
    }
    if (filterState.train && currentProfile === "bike") {
      var km = currentKm.bike;
      var d = Number(props["n_dist_train_m_" + km]);
      if (!isFinite(d)) d = Number(props.n_dist_train_m);
      // По прямой до станции электрички — до 2 км.
      if (!(isFinite(d) && d <= 2000)) return false;
    }
    return true;
  }

  function applyFilters(data, map) {
    ensureFiltersPanel();
    syncFilterTrainRow();
    syncFilterRamp();
    var field = scoreKey();
    var fallback = FIELD[currentProfile] || FIELD.run;
    var features = data && data.features ? data.features : [];
    // полные props из hexProps, если есть
    var on = 0, total = 0;
    for (var i = 0; i < features.length; i++) {
      var f = features[i];
      var id = f.properties && f.properties.h3_index;
      if (id == null) continue;
      total++;
      var props = (hexProps && hexProps[id]) || f.properties;
      var val = props[field];
      if (val == null) val = props[fallback];
      var pass = hexPassesFilters(props);
      if (pass) on++;
      window.llmapsSetFeatureState(SOURCE_ID, id, {
        active: !!pass,
        value: val
      });
    }
    if (map && map.getLayer(LAYER_ID)) {
      map.setPaintProperty(LAYER_ID, "fill-opacity", [
        "case",
        ["boolean", ["feature-state", "active"], true],
        0.42,
        0
      ]);
      if (map.getLayer("hex-fill-outline")) {
        map.setPaintProperty("hex-fill-outline", "line-opacity", [
          "case",
          ["boolean", ["feature-state", "active"], true],
          0.55,
          0
        ]);
      }
    }
    var cnt = document.getElementById("ah-f-count");
    if (cnt) cnt.textContent = "показано " + on + " из " + total + " гексагонов";
  }

  function applyView(data, map) {
    var buttons = document.querySelectorAll("[data-ah-profile]");
    for (var j = 0; j < buttons.length; j++) {
      var on = buttons[j].getAttribute("data-ah-profile") === currentProfile;
      buttons[j].classList.toggle("is-on", on);
    }
    renderOpts();
    applyRamp(map);
    var b = scoreBounds();
    if (filterState.scoreMin < b.min) filterState.scoreMin = b.min;
    if (filterState.scoreMax > b.max) filterState.scoreMax = b.max;
    if (filterState.scoreMin > filterState.scoreMax) {
      filterState.scoreMin = b.min;
      filterState.scoreMax = b.max;
    }
    applyFilters(data, map);
    openRank = null;
    closeElevModal();
    drawLoops(map);
    enhanceSidebar();
  }

  function currentStops() {
    var ramps = (window.llmapsData && window.llmapsData.ramps) || {};
    return ramps[scoreKey()] || ramps.score_run || ramps.run;
  }

  function applyRamp(map) {
    var stops = currentStops();
    if (!stops || !map || !map.getLayer(LAYER_ID)) return;
    map.setPaintProperty(LAYER_ID, "fill-color", colorExpr(stops));
    updateLegend(stops);
  }

  function pickProp() {
    if (!lastProps) return "";
    for (var i = 0; i < arguments.length; i++) {
      var v = lastProps[arguments[i]];
      if (v != null && v !== "") return v;
    }
    return "";
  }

  function syncKmFields() {
    var km = currentKm[currentProfile];
    var r = fmtRadius(km);
    var parkVal = currentProfile === "run"
      ? pickProp("park_ha_run_" + km, "park_ha_run")
      : pickProp("park_ha_bike_" + km, "park_ha_bike");
    var bikeVal = pickProp("bike_infra_km_" + km, "bike_infra_km_r");
    var trainVal = pickProp("dist_train_m_" + km, "dist_train_m");
    var linesVal = pickProp("train_lines_n_" + km, "train_lines_n");
    var rows = document.querySelectorAll("#llmaps-sidebar .llmaps-sidebar-field");
    for (var j = 0; j < rows.length; j++) {
      var lab = rows[j].querySelector(".llmaps-sidebar-field-label");
      var valEl = rows[j].querySelector(".llmaps-sidebar-field-value");
      if (!lab || !valEl) continue;
      var t = factText(lab);
      if (t.indexOf("Парки в радиусе") === 0) {
        lab.textContent = "Парки в радиусе " + r;
        if (parkVal) valEl.textContent = parkVal;
      } else if (t.indexOf("Велодорожки в радиусе") === 0) {
        lab.textContent = "Велодорожки в радиусе " + r;
        if (bikeVal) valEl.textContent = bikeVal;
      } else if (t.indexOf("До электрички") === 0 && trainVal) {
        valEl.textContent = trainVal;
      } else if (t === "Веток электрички" && linesVal !== "") {
        valEl.textContent = linesVal;
      }
    }
  }

  function filterSidebar(profile) {
    syncKmFields();
    var allow = {};
    var labels = profileLabels(profile);
    for (var i = 0; i < labels.length; i++) allow[labels[i]] = true;
    var rows = document.querySelectorAll("#llmaps-sidebar .llmaps-sidebar-field");
    for (var j = 0; j < rows.length; j++) {
      var lab = rows[j].querySelector(".llmaps-sidebar-field-label");
      var text = lab ? factText(lab) : "";
      rows[j].style.display = allow[text] ? "" : "none";
    }
  }

  function renderHead() {
    var title = document.querySelector("#llmaps-sidebar.open .llmaps-sidebar-title");
    if (!title || !lastProps) return;
    var key = scoreKey();
    var score = lastProps[key] != null ? lastProps[key]
      : (lastProps[FIELD[currentProfile]] != null ? lastProps[FIELD[currentProfile]] : "—");
    var why = whyLine(Number(score));
    title.innerHTML =
      '<div class="ah-score">' + score + '</div>' +
      '<div class="ah-profile">' + PROFILE_NAME[currentProfile] + '</div>' +
      (why ? '<div class="ah-why">' + why + "</div>" : "");
  }

  function whyCfg() {
    return (window.llmapsData && window.llmapsData.whyCfg) || {
      run: { park_ha0: 150, park_ha0_by_km: { 5: 150, 10: 600, 15: 1350 } },
      bike: {
        park_ha0: 1000,
        park_ha0_by_km: { 10: 444, 20: 1778, 30: 4000 },
        bike_infra_km0: 8,
        bike_infra_km0_by_km: { 10: 3.6, 20: 14.2, 30: 32 },
        park_mix: { area: 0.5, park_dist: 0.25, water_dist: 0.25 },
        train_lines_cap: 6
      }
    };
  }

  function whyByKm(map, km, fallback) {
    if (!map) return fallback;
    if (map[km] != null) return Number(map[km]);
    if (map[String(km)] != null) return Number(map[String(km)]);
    return fallback;
  }

  function saturate(x, x0) {
    x = Math.max(0, Number(x) || 0);
    x0 = Math.max(1e-9, Number(x0) || 1);
    return 1 - Math.exp(-x / x0);
  }

  function decay(dist, radius) {
    return Math.max(0, Math.min(1, 1 - (Number(dist) || 0) / Math.max(1, Number(radius) || 1)));
  }

  function propNum() {
    if (!lastProps) return NaN;
    for (var i = 0; i < arguments.length; i++) {
      var v = lastProps[arguments[i]];
      if (v == null || v === "") continue;
      if (typeof v === "number" && isFinite(v)) return v;
      var n = parseFloat(String(v).replace(",", ".").replace(/[^\d.eE+-]/g, ""));
      if (isFinite(n)) return n;
    }
    return NaN;
  }

  // Share fields may arrive as 0–1 or as "41 %"/41 from the sidebar formatters.
  function shareOf() {
    for (var i = 0; i < arguments.length; i++) {
      var v = propNum(arguments[i]);
      if (!(v >= 0)) continue;
      return v > 1 ? Math.min(v / 100, 1) : v;
    }
    return NaN;
  }

  function axisTag(f, plusText, minusText) {
    if (!(f >= 0)) return null;
    if (f >= 0.63) return { dist: Math.abs(f - 0.5), pol: "+", text: plusText, f: f };
    if (f <= 0.35) return { dist: Math.abs(f - 0.5), pol: "-", text: minusText, f: f };
    return null;
  }

  function isMinusText(text) {
    return /^(мало |шумные |велодорожек |электрички |ж\/д )/.test(text);
  }

  function pickWhyTags(axes, low, high) {
    var active = axes.filter(Boolean).slice().sort(function (a, b) { return b.dist - a.dist; });
    var pluses = active.filter(function (a) { return a.pol === "+"; });
    var minuses = active.filter(function (a) { return a.pol === "-"; });
    var out = [];
    function push(list) {
      for (var i = 0; i < list.length && out.length < 3; i++) {
        if (out.indexOf(list[i].text) === -1) out.push(list[i].text);
      }
    }
    if (low) {
      push(minuses);
      push(pluses);
    } else if (high) {
      push(pluses);
      push(minuses);
    } else {
      if (pluses.length) out.push(pluses[0].text);
      if (minuses.length && out.length < 3) out.push(minuses[0].text);
      push(active);
    }
    return out.slice(0, 3);
  }

  function forceMinusTag(tags, candidates) {
    if (tags.some(isMinusText)) return tags;
    candidates = candidates.slice().sort(function (a, b) { return a[0] - b[0]; });
    if (!candidates.length) return tags;
    var forced = candidates[0][1];
    return [forced].concat(tags.filter(function (t) { return t !== forced; })).slice(0, 3);
  }

  function whyLine(score) {
    if (!lastProps || !isFinite(score)) return "";
    var km = Number(currentKm[currentProfile]);
    var r = (km * 1000) / 4;
    var cfg = whyCfg();
    var axes = [];
    var forceCand = [];
    if (currentProfile === "run") {
      var ha0 = whyByKm(cfg.run.park_ha0_by_km, km, cfg.run.park_ha0);
      var parkHa = propNum("n_park_ha_run_" + km, "n_park_ha_run", "park_ha_run_" + km, "park_ha_run");
      var dPark = propNum("n_dist_park_m", "dist_park_m");
      var dWater = propNum("n_dist_water_m", "dist_water_m");
      var fPark = 0.7 * saturate(parkHa, ha0) + 0.3 * 0.5 * (decay(dPark, r) + decay(dWater, r));
      var fNet = shareOf("run_friendly_share", "run_friendly_pct");
      if (!(fNet >= 0)) fNet = 0;
      var highRun = shareOf("high_stress_share", "high_stress_pct");
      if (!(highRun >= 0)) highRun = 0;
      var magRun = propNum("mix_magistral_km");
      if (!(magRun >= 0)) magRun = 0;
      var fStress = 1 - highRun;
      // Visible magistrals must not read as "мало магистралей".
      if (highRun >= 0.22 || magRun >= 1) fStress = Math.min(fStress, 0.34);
      axes = [
        axisTag(fPark, "много парков поблизости", "мало парков поблизости"),
        axisTag(fNet, "удобные дорожки", "мало дорожек"),
        axisTag(fStress, "мало магистралей", "шумные магистрали")
      ];
      forceCand = [
        [fPark, "мало парков поблизости"],
        [fNet, "мало дорожек"],
        [fStress, "шумные магистрали"]
      ];
    } else {
      var bcfg = cfg.bike;
      var ha0b = whyByKm(bcfg.park_ha0_by_km, km, bcfg.park_ha0);
      var infra0 = whyByKm(bcfg.bike_infra_km0_by_km, km, bcfg.bike_infra_km0);
      var mix = bcfg.park_mix || { area: 0.5, park_dist: 0.25, water_dist: 0.25 };
      var parkHab = propNum("n_park_ha_bike_" + km, "n_park_ha_bike", "park_ha_bike_" + km, "park_ha_bike");
      var dParkB = propNum("n_dist_park_m", "dist_park_m");
      var dWaterB = propNum("n_dist_water_m", "dist_water_m");
      var fParkB =
        mix.area * saturate(parkHab, ha0b) +
        mix.park_dist * decay(dParkB, r) +
        mix.water_dist * decay(dWaterB, r);
      var infraKm = propNum(
        "n_bike_infra_km_" + km, "n_bike_infra_km_r",
        "bike_infra_km_" + km, "bike_infra_km_r"
      );
      if (!(infraKm >= 0)) infraKm = 0;
      var fInfra = saturate(infraKm, infra0);
      var low = shareOf("low_stress_share", "low_stress_pct");
      if (!(low >= 0)) low = 0;
      var high = shareOf("high_stress_share", "high_stress_pct");
      if (!(high >= 0)) high = 0;
      var magKm = propNum("mix_magistral_km");
      if (!(magKm >= 0)) magKm = 0;
      var fStreets = (0.2 / 0.36) * low + (0.16 / 0.36) * (1 - high);
      // Do not praise quiet streets when magistrals dominate the hex.
      if (high >= 0.22 || magKm >= 1) fStreets = Math.min(fStreets, 0.34);
      var dTrain = propNum("n_dist_train_m_" + km, "n_dist_train_m", "dist_train_m_" + km, "dist_train_m");
      var nLines = propNum("train_lines_n_" + km, "train_lines_n");
      if (!isFinite(nLines)) nLines = 0;
      var fTr = 0.5 * decay(dTrain, r) + 0.5 * Math.min(nLines / (bcfg.train_lines_cap || 6), 1);
      var infraAxis = infraKm < 0.05
        ? { dist: 0.5, pol: "-", text: "велодорожек поблизости нет", f: 0 }
        : axisTag(fInfra, "много велодорожек поблизости", "мало велодорожек поблизости");
      var streets = axisTag(fStreets, "спокойные улицы", "шумные магистрали");
      var parks = axisTag(fParkB, "много парков поблизости", "мало парков поблизости");
      var trains = axisTag(fTr, "электричка рядом", "электрички далеко");
      axes = [infraAxis, streets];
      if (wantTrain) {
        var third = [parks, trains].filter(Boolean).sort(function (a, b) { return b.dist - a.dist; });
        if (third.length) axes.push(third[0]);
      } else if (parks) {
        axes.push(parks);
      }
      forceCand = [
        [infraKm < 0.05 ? 0 : fInfra, infraKm < 0.05 ? "велодорожек поблизости нет" : "мало велодорожек поблизости"],
        [fStreets, "шумные магистрали"],
        [fParkB, "мало парков поблизости"]
      ];
      if (wantTrain) forceCand.push([fTr, "электрички далеко"]);
    }
    var railKm = propNum("n_rail_km", "rail_km");
    if (!(railKm >= 0)) railKm = 0;
    // Только внутри гекса; порог отсекает уголки путей.
    var railAxis = railKm >= 0.25
      ? { dist: 0.48, pol: "-", text: "ж/д через район", f: 0 }
      : null;
    if (railAxis) {
      axes.push(railAxis);
      forceCand.push([0, "ж/д через район"]);
    }
    var stops = currentStops() || [];
    var vals = stops.map(function (s) { return Number(s[0]); }).filter(isFinite);
    var mid = vals.length ? vals[Math.floor(vals.length / 2)] : 60;
    var hi = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 2 / 3))] : 70;
    var tags = pickWhyTags(axes, score < mid, score >= hi);
    if (score < mid) tags = forceMinusTag(tags, forceCand);
    // Доступ к электричке и «пути режут район» — разные истории; плюс прячет минус.
    if (tags.indexOf("электричка рядом") !== -1) {
      tags = tags.filter(function (t) { return t !== "ж/д через район"; });
    }
    // «Мало магистралей» рядом с ж/д звучит как ошибка — пути важнее автонуля.
    if (tags.indexOf("ж/д через район") !== -1) {
      tags = tags.filter(function (t) { return t !== "мало магистралей"; });
    }
    return tags.slice(0, 3).join(" · ");
  }

  function fmtKm(value) {
    var n = Number(value || 0);
    var tenths = Math.round(n * 10) / 10;
    if (Math.abs(tenths - Math.round(tenths)) < 0.05) {
      return Math.round(tenths) + " км";
    }
    return tenths.toLocaleString("ru-RU", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    }) + " км";
  }

  function ahIco(kind) {
    var fill = {
      hwy: "#c0564e", street: "#c9a05b", walk: "#93a3ae", path: "#4f8a5b",
      bike: "#3f7fb5", park: "#4f8a5b", parkArea: "#4f8a5b", parkWalk: "#4f8a5b",
      water: "#3f7fb5", waterWalk: "#3f7fb5", train: "#6b7280", rails: "#6b7280",
      shop: "#8B5A2B", repair: "#6B3F1D", quiet: "#4f8a5b"
    }[kind] || "#9ca3af";
    var inner = {
      hwy: '<rect x="2" y="5.2" width="12" height="2.2" rx="1"/><rect x="2" y="8.6" width="12" height="2.2" rx="1"/>',
      street: '<rect x="1.6" y="6.4" width="12.8" height="3.2" rx="1.4"/>',
      walk: '<circle cx="8" cy="3.6" r="1.6"/><path d="M8 5.4v3.4M5.8 7.4h4.4M6.2 14 8 8.8 9.8 14" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>',
      path: '<path d="M3 12c2-4 3-4 5-1s3 3 5-2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
      bike: '<circle cx="5" cy="11" r="2.2" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="11.4" cy="11" r="2.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M5 11 8 6.2h3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>',
      park: '<ellipse cx="8" cy="6.1" rx="4.6" ry="4.3"/><rect x="7" y="9.2" width="2" height="4.6" rx="0.4"/>',
      parkArea: '<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5"/><ellipse cx="8" cy="7.2" rx="2.6" ry="2.4"/><rect x="7.3" y="9" width="1.4" height="2.6" rx="0.3"/>',
      parkWalk: '<path d="M8 1.6c-1.9 0-3.4 1.5-3.4 3.3 0 2.6 3.4 8 3.4 8s3.4-5.4 3.4-8c0-1.8-1.5-3.3-3.4-3.3z"/><circle cx="8" cy="4.9" r="1.25" fill="#fff"/>',
      water: '<path d="M2 6.4c1.3 0 1.3 1.7 2.7 1.7S6 6.4 7.4 6.4 8.7 8.1 10.1 8.1 11.4 6.4 12.8 6.4M2 10.6c1.3 0 1.3 1.7 2.7 1.7S6 10.6 7.4 10.6 8.7 12.3 10.1 12.3 11.4 10.6 12.8 10.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
      waterWalk: '<path d="M8 2.2S3.6 7.2 3.6 10.2a4.4 4.4 0 108.8 0C12.4 7.2 8 2.2 8 2.2z"/>',
      train: '<rect x="3" y="3.5" width="10" height="8" rx="1.5"/><rect x="5" y="5.5" width="2.4" height="2.2" fill="#fff"/><rect x="8.6" y="5.5" width="2.4" height="2.2" fill="#fff"/><circle cx="5.5" cy="13" r="1.1"/><circle cx="10.5" cy="13" r="1.1"/>',
      rails: '<path d="M4.2 2.4v11.2M11.8 2.4v11.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M3 5.2h10M3 8h10M3 10.8h10" fill="none" stroke="currentColor" stroke-width="1.25"/>',
      shop: '<path d="M2.4 6.8 3.9 14h8.2l1.5-7.2z"/><path d="M5 6.8C5 3.8 6.3 2.2 8 2.2s3 1.6 3 4.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M5.1 9v3.2M8 8.8v3.6M10.9 9v3.2" fill="none" stroke="#fff" stroke-width="1.2" stroke-linecap="round"/>',
      repair: '<g transform="rotate(-32 8 8)"><rect x="1.4" y="6.7" width="8.2" height="2.6" rx="1.3"/><path d="M9.2 4.2h5.2v2.6h-1.7V5.6h-1.8v4.8h1.8v-.8h1.7V12.2H9.2z"/></g>',
      quiet: '<path d="M9 3.2c-3.2.4-5.5 3.2-5.5 6.6S5.8 16 9 16c.4 0 .7 0 1.1-.1C8 15 6.4 12.8 6.4 10s1.6-5 3.7-5.9V3.2z"/>'
    }[kind] || '<circle cx="8" cy="8" r="3"/>';
    return '<svg class="ah-ico" viewBox="0 0 16 16" fill="' + fill + '" color="' + fill + '" aria-hidden="true">' + inner + "</svg>";
  }

  function factText(lab) {
    if (!lab) return "";
    var clone = lab.cloneNode(true);
    var icos = clone.querySelectorAll(".ah-ico");
    for (var i = 0; i < icos.length; i++) icos[i].remove();
    return (clone.textContent || "").replace(/\s+/g, " ").trim();
  }

  function iconForFact(text) {
    if (text.indexOf("Парки в радиусе") === 0) return "parkArea";
    if (text.indexOf("До парка") === 0) return "parkWalk";
    if (text.indexOf("Парк") === 0) return "park";
    if (text.indexOf("До воды") === 0) return "waterWalk";
    if (text.indexOf("Водо") === 0) return "water";
    if (text.indexOf("Веток") === 0) return "rails";
    if (text.indexOf("электр") !== -1) return "train";
    if (text.indexOf("веломагаз") !== -1) return "shop";
    if (text.indexOf("велосерв") !== -1) return "repair";
    if (text.indexOf("Спокойн") === 0) return "quiet";
    if (text.indexOf("Велодорож") === 0) return "bike";
    return "park";
  }

  function dressHexFacts() {
    var content = document.querySelector("#llmaps-sidebar.open .llmaps-sidebar-content");
    if (!content) return;
    content.classList.add("ah-facts");
    var section = content.querySelector(".llmaps-sidebar-section") || content;
    var oldTitles = section.querySelectorAll(".ah-facts-title");
    for (var t = 0; t < oldTitles.length; t++) oldTitles[t].remove();
    var rows = section.querySelectorAll(".llmaps-sidebar-field");
    var first = null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].style.display === "none") continue;
      var lab = rows[i].querySelector(".llmaps-sidebar-field-label");
      if (!lab) continue;
      var text = factText(lab);
      var oldIco = lab.querySelector(".ah-ico");
      if (oldIco) oldIco.remove();
      lab.insertAdjacentHTML("afterbegin", ahIco(iconForFact(text)));
      if (!first) first = rows[i];
    }
    if (first) {
      var title = document.createElement("div");
      title.className = "ah-facts-title";
      title.textContent = "Окружение";
      first.parentNode.insertBefore(title, first);
    }
  }

  function renderMix() {
    var sidebar = document.getElementById("llmaps-sidebar");
    if (!sidebar || !sidebar.classList.contains("open") || !lastProps) return;
    var old = sidebar.querySelector(".ah-mix");
    if (old) old.remove();
    var parts = [];
    var maxKm = 0;
    for (var i = 0; i < MIX.length; i++) {
      var km = Number(lastProps[MIX[i].key] || 0);
      if (km > maxKm) maxKm = km;
      parts.push({ meta: MIX[i], km: km });
    }
    if (maxKm <= 0) return;
    parts.sort(function (a, b) {
      var aHwy = a.meta.icon === "hwy" ? 1 : 0;
      var bHwy = b.meta.icon === "hwy" ? 1 : 0;
      if (aHwy !== bHwy) return aHwy - bHwy;
      return b.km - a.km;
    });
    var rows = "";
    for (var j = 0; j < parts.length; j++) {
      var width = Math.max(2, (parts[j].km / maxKm) * 100);
      if (parts[j].km <= 0) width = 0;
      rows +=
        '<div class="ah-mix-row">' + ahIco(parts[j].meta.icon) +
        '<div class="ah-mix-row-label">' + parts[j].meta.label + "</div>" +
        '<div class="ah-mix-row-track"><span style="width:' + width +
        "%;background:" + parts[j].meta.color + '"></span></div>' +
        '<div class="ah-mix-row-val">' + fmtKm(parts[j].km) + "</div></div>";
    }
    var box = document.createElement("div");
    box.className = "ah-mix";
    box.innerHTML =
      '<div class="ah-mix-title">В гексагоне</div>' +
      '<p class="ah-mix-hint">Тротуары — вдоль улиц. Тропы в парках — дорожки внутри парков и леса. Дворы и подъезды не считаем.</p>' +
      rows;
    var content = sidebar.querySelector(".llmaps-sidebar-content");
    if (routesReady() && routesKnown() && !loopsForHex().length) {
      box.innerHTML +=
        '<p class="ah-mix-hint">Замкнуть петлю нужной длины здесь не удалось: ' +
        'сетка улиц не даёт круга без длинных повторов.</p>';
    }
    if (content) sidebar.insertBefore(box, content);
    else sidebar.appendChild(box);
  }

  function fmtPct(share) {
    return Math.round(Number(share || 0) * 100) + "%";
  }

  function featKm(p) {
    if (p && p.km != null) return Number(p.km);
    return p && p.profile === "run" ? 5 : 15;
  }

  function routesReady() {
    var have = (window.llmapsData && window.llmapsData.routeKm) || { run: [5], bike: [15] };
    var want = Number(currentKm[currentProfile]);
    var list = have[currentProfile];
    if (Array.isArray(list)) return list.indexOf(want) !== -1;
    return Number(list) === want;
  }

  function loopsForHex() {
    var fc = loopHex ? loopCache[loopHex] : null;
    if (!fc || !fc.features || !routesReady()) return [];
    var wantKm = Number(currentKm[currentProfile]);
    return fc.features.filter(function (f) {
      return f.properties.profile === currentProfile && featKm(f.properties) === wantKm;
    }).sort(function (a, b) { return a.properties.rank - b.properties.rank; });
  }

  // крошечный контур петли: по нему видно конфигурацию, даже когда треки наложились
  function routeGlyph(feat, size, color) {
    var cs = feat.geometry.coordinates;
    if (!cs || cs.length < 3) return "";
    var k = Math.cos(cs[0][1] * Math.PI / 180);
    var minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, i;
    for (i = 0; i < cs.length; i++) {
      var x = cs[i][0] * k, y = cs[i][1];
      if (x < minx) minx = x;
      if (x > maxx) maxx = x;
      if (y < miny) miny = y;
      if (y > maxy) maxy = y;
    }
    var span = Math.max(maxx - minx, maxy - miny) || 1;
    var pad = 2.5, sc = (size - pad * 2) / span;
    var ox = (size - (maxx - minx) * sc) / 2, oy = (size - (maxy - miny) * sc) / 2;
    var step = Math.max(1, Math.floor(cs.length / 90)), d = "";
    for (i = 0; i < cs.length; i += step) {
      var px = ox + (cs[i][0] * k - minx) * sc;
      var py = size - (oy + (cs[i][1] - miny) * sc);
      d += (i ? "L" : "M") + px.toFixed(1) + " " + py.toFixed(1);
    }
    return '<svg class="ah-glyph" width="' + size + '" height="' + size +
      '" viewBox="0 0 ' + size + " " + size + '"><path d="' + d + 'Z" fill="' + color +
      '14" stroke="' + color + '" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  }

  function surfBlock(p) {
    var items = SURF.map(function (s) { return { m: s, v: Number(p[s.key] || 0) }; })
      .filter(function (x) { return x.v > 0.001; })
      .sort(function (a, b) { return b.v - a.v; });
    if (!items.length) return "";
    var bar = "", keys = "";
    for (var i = 0; i < items.length; i++) {
      bar += '<i style="width:' + (items[i].v * 100).toFixed(1) + "%;background:" + items[i].m.color + '"></i>';
      keys += '<span><span class="ah-dot" style="background:' + items[i].m.color + '"></span>' +
        items[i].m.label + " <b>" + fmtPct(items[i].v) + "</b></span>";
    }
    return '<div class="ah-sub-h">Покрытие</div><div class="ah-stack">' + bar +
      '</div><div class="ah-keys">' + keys + "</div>";
  }

  function routeVerdict(p) {
    var stone = Number(p.surf_stone || 0), soil = Number(p.surf_soil || 0);
    if (currentProfile === "run" && stone >= 0.3)
      return '<div class="ah-verdict">Почти треть пути по брусчатке — для бега жёстко.</div>';
    if (soil >= 0.5)
      return '<div class="ah-verdict ok">Больше половины по грунту — мягко под ногу.</div>';
    if (Number(p.stress_share || 0) >= 0.1)
      return '<div class="ah-verdict">' + fmtPct(p.stress_share) + ' пути вдоль магистралей.</div>';
    return "";
  }

  function elevStats(p) {
    var zs = p.elev || [], lo = p.elev_min, hi = p.elev_max, i;
    if (!zs.length) return null;
    if (lo == null || hi == null) {
      lo = hi = zs[0];
      for (i = 1; i < zs.length; i++) {
        if (zs[i] < lo) lo = zs[i];
        if (zs[i] > hi) hi = zs[i];
      }
    }
    return { zs: zs, lo: lo, hi: hi, rng: Math.max(8, hi - lo) };
  }

  function elevLine(st, x0, y0, w, h) {
    var d = "", i, zs = st.zs;
    for (i = 0; i < zs.length; i++) {
      var x = x0 + (zs.length === 1 ? 0 : i / (zs.length - 1)) * w;
      var y = y0 + h - ((zs[i] - st.lo) / st.rng) * h;
      d += (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return d;
  }

  function elevNums(p, st) {
    return '<div class="ah-elev-nums">' +
      '<span class="ah-elev-dir"><i>▲</i> Набор <b>' + Math.round(p.gain_m || 0) +
      " м</b></span>" +
      '<span class="ah-elev-dir"><i>▼</i> Сброс <b>' + Math.round(p.loss_m || 0) +
      " м</b></span>" +
      "<span>Диапазон высот <b>" + Math.round(st.lo) + "–" + Math.round(st.hi) +
      " м</b></span></div>";
  }

  function elevSvgMini(st, color) {
    var w = 280, h = 52;
    var d = elevLine(st, 0, 4, w, h - 8);
    return '<svg class="ah-elev-svg" viewBox="0 0 ' + w + " " + h +
      '" preserveAspectRatio="none"><path d="' + d + "L" + w + " " + h +
      "L0 " + h + 'Z" fill="' + color + '24"/><path d="' + d +
      '" fill="none" stroke="' + color +
      '" stroke-width="1.7" stroke-linejoin="round"/></svg>';
  }

  function elevSvgFull(p, st, color) {
    var W = 720, H = 148, l = 48, t = 10, r = 16, b = 26;
    var pw = W - l - r, ph = H - t - b;
    var d = elevLine(st, l, t, pw, ph);
    var ticks = [0, 0.25, 0.5, 0.75, 1];
    var mid = Math.round((st.lo + st.hi) / 2);
    var yvals = [st.hi, mid, st.lo];
    var grid = "", ylab = "", xlab = "", i, x, y, frac;
    for (i = 0; i < yvals.length; i++) {
      frac = (yvals[i] - st.lo) / st.rng;
      y = t + ph - frac * ph;
      grid += '<line x1="' + l + '" y1="' + y.toFixed(1) + '" x2="' + (l + pw) +
        '" y2="' + y.toFixed(1) + '" stroke="#e5e7eb" stroke-width="1"/>';
      ylab += '<text x="' + (l - 6) + '" y="' + (y + 3).toFixed(1) +
        '" text-anchor="end" fill="#6b7280" font-size="11">' +
        Math.round(yvals[i]) + " м</text>";
    }
    for (i = 0; i < ticks.length; i++) {
      x = l + ticks[i] * pw;
      grid += '<line x1="' + x.toFixed(1) + '" y1="' + t + '" x2="' + x.toFixed(1) +
        '" y2="' + (t + ph) + '" stroke="#f3f4f6" stroke-width="1"/>';
      xlab += '<text x="' + x.toFixed(1) + '" y="' + (H - 10) +
        '" text-anchor="middle" fill="#6b7280" font-size="11">' +
        (i === 0 ? "старт" : fmtKm(p.length_m * ticks[i] / 1000)) + "</text>";
    }
    return '<svg class="ah-elev-full" viewBox="0 0 ' + W + " " + H +
      '" preserveAspectRatio="xMidYMid meet">' + grid + ylab + xlab +
      '<path d="' + d + "L" + (l + pw) + " " + (t + ph) + "L" + l + " " +
      (t + ph) + 'Z" fill="' + color + '24"/><path d="' + d +
      '" fill="none" stroke="' + color +
      '" stroke-width="2" stroke-linejoin="round"/></svg>';
  }

  function closeElevModal() {
    elevOpen = false;
    renderElevModal();
  }

  function renderElevModal() {
    var host = document.getElementById("ah-elev-modal");
    if (!host) return;
    var list = loopsForHex();
    var feat = null, i;
    if (elevOpen && openRank != null) {
      for (i = 0; i < list.length; i++) {
        if (list[i].properties && list[i].properties.rank === openRank) {
          feat = list[i];
          break;
        }
      }
    }
    var st = feat ? elevStats(feat.properties) : null;
    if (!st) {
      host.className = "ah-elev-modal";
      host.hidden = true;
      host.innerHTML = "";
      return;
    }
    var p = feat.properties;
    var c = LOOP_COLOR[p.rank - 1] || LOOP_COLOR[0];
    host.hidden = false;
    host.className = "ah-elev-modal is-on";
    host.innerHTML = '<div class="ah-elev-card" role="dialog" aria-modal="true">' +
      "<header><h3>Маршрут " + p.rank + " · рельеф</h3>" +
      '<button type="button" class="ah-elev-close" data-ah-elev-close aria-label="Закрыть" title="Закрыть">&times;</button></header>' +
      elevSvgFull(p, st, c) + elevNums(p, st) + "</div>";
  }

  function elevBlock(p, color) {
    var st = elevStats(p);
    if (!st) {
      return '<div class="ah-soon"><b>Набор высоты</b> — профиля нет.</div>';
    }
    return '<div class="ah-elev"><div class="ah-elev-head"><div class="ah-sub-h">Рельеф</div>' +
      '<button type="button" class="ah-elev-open" data-ah-elev aria-label="Развернуть" title="Развернуть">' +
      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9"/></svg></button></div>' +
      '<div class="ah-elev-mini">' + elevSvgMini(st, color) + "</div>" +
      '<div class="ah-elev-axis"><span>старт</span><span>' +
      fmtKm(p.length_m / 2000) + "</span><span>" +
      fmtKm(p.length_m / 1000) + "</span></div>" +
      elevNums(p, st) + "</div>";
  }

  function routeDetail(p, color) {
    var rows =
      '<div class="ah-kv"><span>В парке</span><b>' + fmtKm(p.park_m / 1000) + "</b></div>" +
      '<div class="ah-kv"><span>По велодорожкам</span><b>' + fmtKm(p.bike_m / 1000) + "</b></div>" +
      '<div class="ah-kv"><span>Хороших дорожек</span><b>' + fmtPct(p.good_share) + "</b></div>" +
      '<div class="ah-kv"><span>Вдоль магистралей</span><b>' + fmtPct(p.stress_share) + "</b></div>" +
      '<div class="ah-kv"><span>Повторы пути</span><b>' + fmtPct(p.repeat_share) + "</b></div>";
    return '<div class="ah-route-body">' + routeVerdict(p) + surfBlock(p) +
      '<div class="ah-sub-h">Подробности</div>' + rows +
      elevBlock(p, color || LOOP_COLOR[0]) + "</div>";
  }

  function routesKnown() {
    return !loopHex || !!loopCache[loopHex];
  }

  function wantRoutesTab() {
    if (loopsForHex().length) return true;
    // пока качается JSON соседнего гекса, не прячем вкладку — иначе на полсекунды
    // выкидывает в «Гексагон», хотя человек как раз сравнивает петли
    return ahTab === "routes" && !!loopHex && !routesKnown();
  }

  function renderRoutes(sidebar) {
    var box = sidebar.querySelector(".ah-routes");
    if (!box) {
      box = document.createElement("div");
      box.className = "ah-routes";
      var mix = sidebar.querySelector(".ah-mix");
      if (mix) sidebar.insertBefore(box, mix);
      else sidebar.appendChild(box);
    }
    var list = loopsForHex();
    if (!list.length) {
      box.innerHTML = !routesKnown()
        ? '<p class="ah-mix-hint">Загружаем маршруты…</p>'
        : !routesReady()
        ? '<p class="ah-mix-hint">Петли для ' + currentKm[currentProfile] +
          ' км ещё не посчитаны.</p>'
        : '<p class="ah-mix-hint">Замкнуть петлю нужной длины здесь не удалось: ' +
          'сетка улиц не даёт круга без длинных повторов.</p>';
      return;
    }
    var html = '<p class="ah-mix-hint">Профиль «' + PROFILE_NAME[currentProfile] +
      '» — переключается в шапке слева.</p>';
    for (var i = 0; i < list.length; i++) {
      var p = list[i].properties, c = LOOP_COLOR[p.rank - 1] || LOOP_COLOR[0];
      var on = openRank === p.rank;
      html += '<div class="ah-route' + (on ? " sel" : "") + '" style="--edge:' + c +
        '" data-ah-rank="' + p.rank + '"><div class="ah-route-head">' +
        routeGlyph(list[i], 28, c) +
        '<div><div class="ah-route-title">Маршрут ' + p.rank + "</div>" +
        '<div class="ah-route-sum">' + fmtKm(p.length_m / 1000) + " · парк " +
        fmtPct(p.park_m / p.length_m) +
        (p.gain_m != null ? " · +" + Math.round(p.gain_m) + " м" : "") +
        "</div></div>" +
        '<span class="ah-chev">›</span></div>' + (on ? routeDetail(p, c) : "") + "</div>";
    }
    box.innerHTML = html;
  }

  function renderTabs(sidebar) {
    var tabs = sidebar.querySelector(".ah-tabs");
    if (!tabs) {
      tabs = document.createElement("div");
      tabs.className = "ah-tabs";
      var anchor = sidebar.querySelector(".ah-routes") || sidebar.querySelector(".ah-mix") ||
        sidebar.querySelector(".llmaps-sidebar-content");
      if (anchor) sidebar.insertBefore(tabs, anchor);
      else sidebar.appendChild(tabs);
    }
    if (!wantRoutesTab()) {
      if (ahTab === "routes") ahTab = "hex";
      tabs.style.display = "none";
      tabs.innerHTML = "";
      return;
    }
    tabs.style.display = "";
    var n = loopsForHex().length;
    tabs.innerHTML =
      '<button type="button" class="ah-tab' + (ahTab === "hex" ? " is-on" : "") +
      '" data-ah-tab="hex">Гексагон</button>' +
      '<button type="button" class="ah-tab' + (ahTab === "routes" ? " is-on" : "") +
      '" data-ah-tab="routes">Маршруты' + (n ? " (" + n + ")" : "") + "</button>";
  }

  function applyTab(sidebar) {
    var routesOn = ahTab === "routes";
    var mix = sidebar.querySelector(".ah-mix");
    var routes = sidebar.querySelector(".ah-routes");
    var content = sidebar.querySelector(".llmaps-sidebar-content");
    if (mix) mix.style.display = routesOn ? "none" : "";
    if (routes) routes.style.display = routesOn ? "" : "none";
    if (content) content.style.display = routesOn ? "none" : "";
  }

  function isPhone() {
    return window.matchMedia("(max-width: 768px)").matches;
  }

  function setSheetH(sb, px) {
    var max = window.innerHeight;
    var next = Math.max(140, Math.min(max, Math.round(px)));
    // inline !important beats the stylesheet 42vh !important
    sb.style.setProperty("height", next + "px", "important");
  }

  function syncSheet() {
    var sb = document.getElementById("llmaps-sidebar");
    var open = !!(sb && sb.classList.contains("open"));
    if (!open && sb) sb.style.removeProperty("height");
  }

  function ensureSheetHandle(sb) {
    if (!sb || sb.querySelector(".ah-sheet-handle")) return;
    var h = document.createElement("div");
    h.className = "ah-sheet-handle";
    h.innerHTML = "<i></i>";
    sb.insertBefore(h, sb.firstChild);
    var startY = 0;
    var startH = 0;
    var dragged = false;
    h.addEventListener("pointerdown", function (ev) {
      if (!isPhone()) return;
      dragged = false;
      startY = ev.clientY;
      startH = sb.getBoundingClientRect().height;
      h.setPointerCapture(ev.pointerId);
    });
    h.addEventListener("pointermove", function (ev) {
      if (!h.hasPointerCapture(ev.pointerId)) return;
      if (Math.abs(ev.clientY - startY) > 6) dragged = true;
      setSheetH(sb, startH + (startY - ev.clientY));
    });
    h.addEventListener("pointerup", function () {
      if (!isPhone()) return;
      var hNow = sb.getBoundingClientRect().height;
      if (!dragged) {
        var full = hNow > window.innerHeight * 0.85;
        setSheetH(sb, full ? window.innerHeight * 0.42 : window.innerHeight);
      } else if (hNow > window.innerHeight * 0.82) {
        setSheetH(sb, window.innerHeight);
      }
    });
  }

  function enhanceSidebar() {
    if (observer) observer.disconnect();
    try {
      filterSidebar(currentProfile);
      renderHead();
      renderMix();
      dressHexFacts();
      var sb = document.getElementById("llmaps-sidebar");
      if (sb && sb.classList.contains("open")) {
        ensureSheetHandle(sb);
        renderRoutes(sb);
        renderTabs(sb);
        applyTab(sb);
      }
      syncSheet();
    } finally {
      var sidebar = document.getElementById("llmaps-sidebar");
      if (observer && sidebar) {
        observer.observe(sidebar, { childList: true, subtree: true });
      }
    }
  }

  function emptySel() {
    return { type: "FeatureCollection", features: [] };
  }

  function indexHexGeom(data) {
    hexGeom = {};
    hexProps = {};
    var features = data && data.features ? data.features : [];
    for (var i = 0; i < features.length; i++) {
      var f = features[i];
      var id = f.properties && f.properties.h3_index;
      if (id == null) continue;
      if (f.geometry) hexGeom[id] = f.geometry;
      if (f.properties) hexProps[id] = f.properties;
    }
  }

  // queryRenderedFeatures / LLMaps FileSource drop unused props (n_*, *_share).
  function resolveHexProps(props) {
    if (!props) return null;
    var id = props.h3_index;
    if (id != null && hexProps && hexProps[id]) return hexProps[id];
    var data = window._ahSourceData;
    if (id != null && data && data.features && (!hexProps || !hexProps[id])) {
      // Prefer already-indexed full props; fall back to stripped source.
      if (!hexProps) indexHexGeom(data);
      if (hexProps && hexProps[id]) return hexProps[id];
    }
    return props;
  }

  function loadFullHexProps() {
    return fetch("_hexes.geojson")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        indexHexGeom(data);
        if (lastProps && lastProps.h3_index && hexProps[lastProps.h3_index]) {
          lastProps = hexProps[lastProps.h3_index];
          enhanceSidebar();
        }
        return data;
      })
      .catch(function () { return null; });
  }

  function outlineGeom(geom) {
    if (!geom) return null;
    if (geom.type === "Polygon" && geom.coordinates && geom.coordinates[0]) {
      return { type: "LineString", coordinates: geom.coordinates[0] };
    }
    if (geom.type === "MultiPolygon" && geom.coordinates[0] && geom.coordinates[0][0]) {
      return { type: "LineString", coordinates: geom.coordinates[0][0] };
    }
    return geom;
  }

  function highlightGeom(feat) {
    if (hexGeom == null) {
      var cached = window._llmapsSourceGeoJson && window._llmapsSourceGeoJson[SOURCE_ID];
      if (cached) indexHexGeom(cached);
    }
    var id = feat && feat.properties && feat.properties.h3_index;
    var geom = (id != null && hexGeom && hexGeom[id]) || (feat && feat.geometry);
    return outlineGeom(geom);
  }

  function ensureSelectLayer(map) {
    if (!map.getSource(SELECT_SOURCE)) {
      map.addSource(SELECT_SOURCE, {
        type: "geojson",
        data: emptySel(),
        tolerance: 0,
        buffer: 128
      });
    }
    if (!map.getLayer(SELECT_CASING)) {
      map.addLayer({
        id: SELECT_CASING,
        type: "line",
        source: SELECT_SOURCE,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": SELECT_CASING_COLOR,
          "line-width": ["interpolate", ["linear"], ["zoom"], 8, 9, 12, 8.5],
          "line-opacity": 0.9
        }
      });
    }
    if (!map.getLayer(SELECT_LAYER)) {
      map.addLayer({
        id: SELECT_LAYER,
        type: "line",
        source: SELECT_SOURCE,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": SELECT_COLOR,
          "line-width": ["interpolate", ["linear"], ["zoom"], 8, 4.5, 12, 4],
          "line-opacity": 1
        }
      });
    }
  }

  function emptyFc() {
    return { type: "FeatureCollection", features: [] };
  }

  function distM(a, b) {
    var k = 111320;
    var dx = (b[0] - a[0]) * Math.cos((a[1] + b[1]) * Math.PI / 360) * k;
    var dy = (b[1] - a[1]) * k;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function bearingDeg(a, b) {
    var lat1 = a[1] * Math.PI / 180;
    var lat2 = b[1] * Math.PI / 180;
    var dlon = (b[0] - a[0]) * Math.PI / 180;
    var y = Math.sin(dlon) * Math.cos(lat2);
    var x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dlon);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }

  function arrowPoints(coords) {
    if (!coords || coords.length < 2) return [];
    var segs = [];
    var total = 0;
    var i;
    for (i = 0; i < coords.length - 1; i++) {
      var d = distM(coords[i], coords[i + 1]);
      segs.push(d);
      total += d;
    }
    if (total < 1) return [];
    var marks = [0.22, 0.5, 0.78];
    var out = [];
    for (var m = 0; m < marks.length; m++) {
      var target = marks[m] * total;
      var acc = 0;
      for (i = 0; i < segs.length; i++) {
        if (acc + segs[i] >= target || i === segs.length - 1) {
          var t = segs[i] < 1e-6 ? 0 : Math.min(1, (target - acc) / segs[i]);
          var a = coords[i];
          var b = coords[i + 1];
          out.push({
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
            },
            properties: { bearing: bearingDeg(a, b) }
          });
          break;
        }
        acc += segs[i];
      }
    }
    return out;
  }

  function ensureChevron(map) {
    if (map.hasImage("ah-chevron")) map.removeImage("ah-chevron");
    var s = 20;
    var c = document.createElement("canvas");
    c.width = s;
    c.height = s;
    var g = c.getContext("2d");
    g.lineJoin = "round";
    g.lineCap = "round";
    // треугольник смотрит вверх (север), чтобы icon-rotate = азимут
    g.beginPath();
    g.moveTo(10, 3);
    g.lineTo(16.5, 16);
    g.lineTo(3.5, 16);
    g.closePath();
    g.fillStyle = SELECT_COLOR;
    g.fill();
    g.lineWidth = 1.6;
    g.strokeStyle = "#ffffff";
    g.stroke();
    map.addImage("ah-chevron", g.getImageData(0, 0, s, s));
  }

  function ensureLoopLayer(map) {
    ensureChevron(map);
    if (!map.getSource(LOOP_SOURCE)) {
      map.addSource(LOOP_SOURCE, { type: "geojson", data: emptyFc() });
    }
    if (!map.getSource(LOOP_START_SRC)) {
      map.addSource(LOOP_START_SRC, { type: "geojson", data: emptyFc() });
    }
    if (!map.getSource(LOOP_ARROW_SRC)) {
      map.addSource(LOOP_ARROW_SRC, { type: "geojson", data: emptyFc() });
    }
    if (!map.getLayer(LOOP_HALO)) {
      map.addLayer({
        id: LOOP_HALO,
        type: "line",
        source: LOOP_SOURCE,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-width": 0, "line-opacity": 0 }
      });
    }
    if (!map.getLayer(LOOP_CASING)) {
      map.addLayer({
        id: LOOP_CASING,
        type: "line",
        source: LOOP_SOURCE,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": "#ffffff",
          "line-width": ["interpolate", ["linear"], ["zoom"], 10, 3.4, 15, 6],
          "line-opacity": 0.85
        }
      });
    }
    if (!map.getLayer(LOOP_LINE)) {
      map.addLayer({
        id: LOOP_LINE,
        type: "line",
        source: LOOP_SOURCE,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": ["case", ["==", ["get", "rank"], 1], LOOP_COLOR[0], LOOP_COLOR[1]],
          "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.4, 15, 2.8],
          "line-opacity": 0.95
        }
      });
    }
    if (!map.getLayer(LOOP_ARROWS)) {
      map.addLayer({
        id: LOOP_ARROWS,
        type: "symbol",
        source: LOOP_ARROW_SRC,
        layout: {
          "icon-image": "ah-chevron",
          "icon-size": 0.7,
          "icon-rotate": ["get", "bearing"],
          "icon-rotation-alignment": "map",
          "icon-pitch-alignment": "map",
          "icon-allow-overlap": true,
          "icon-ignore-placement": true
        }
      });
    }
    if (!map.getLayer(LOOP_START)) {
      map.addLayer({
        id: LOOP_START,
        type: "circle",
        source: LOOP_START_SRC,
        paint: {
          "circle-color": SELECT_COLOR,
          "circle-radius": 5.5,
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2
        }
      });
    }
  }

  function drawLoops(map) {
    var src = map.getSource(LOOP_SOURCE);
    if (!src) return;
    var shown = ahTab === "routes" ? loopsForHex() : [];
    src.setData({ type: "FeatureCollection", features: shown });

    // выбранный не чернеет и не толстеет: он остаётся своим цветом и получает
    // мягкое гало, а остальные бледнеют
    var color = ["case", ["==", ["get", "rank"], 1], LOOP_COLOR[0], LOOP_COLOR[1]];
    var isSel = ["==", ["get", "rank"], openRank == null ? -1 : openRank];
    var some = openRank != null;
    map.setPaintProperty(LOOP_LINE, "line-color", color);
    map.setPaintProperty(LOOP_LINE, "line-width",
      ["case", isSel, 3.6, some ? 1.7 : 2.6]);
    map.setPaintProperty(LOOP_LINE, "line-opacity",
      ["case", isSel, 1, some ? 0.28 : 0.95]);
    map.setPaintProperty(LOOP_CASING, "line-opacity", some ? ["case", isSel, 0.85, 0.3] : 0.85);
    map.setPaintProperty(LOOP_HALO, "line-color", color);
    map.setPaintProperty(LOOP_HALO, "line-width", ["case", isSel, 11, 0]);
    map.setPaintProperty(LOOP_HALO, "line-opacity", ["case", isSel, 0.18, 0]);
    var sel = null;
    if (openRank != null) {
      for (var i = 0; i < shown.length; i++) {
        if (shown[i].properties && shown[i].properties.rank === openRank) {
          sel = shown[i];
          break;
        }
      }
    }
    var coords = sel && sel.geometry && sel.geometry.coordinates;
    var startSrc = map.getSource(LOOP_START_SRC);
    if (startSrc) {
      var start = coords && coords[0];
      startSrc.setData(start
        ? { type: "FeatureCollection", features: [{
            type: "Feature",
            geometry: { type: "Point", coordinates: start },
            properties: {}
          }] }
        : emptyFc());
    }
    var arrowSrc = map.getSource(LOOP_ARROW_SRC);
    if (arrowSrc) {
      arrowSrc.setData({ type: "FeatureCollection", features: coords ? arrowPoints(coords) : [] });
    }
  }

  // one file per hex, fetched on click: the browser never holds the whole catalogue
  function showLoops(map, h3) {
    if ((h3 || null) !== loopHex) {
      openRank = null;
      closeElevModal();
      // вкладку держим, если человек сравнивает петли соседних гексов.
      // сброс — только когда сайдбар закрыли целиком
      if (!h3) ahTab = "hex";
    }
    loopHex = h3 || null;
    if (!loopHex || loopCache[loopHex]) {
      drawLoops(map);
      return;
    }
    var want = loopHex;
    fetch("loops/" + want + ".json")
      .then(function (r) { return r.ok ? r.json() : emptyFc(); })
      .catch(function () { return emptyFc(); })
      .then(function (fc) {
        loopCache[want] = fc;
        if (loopHex === want) {
          drawLoops(map);
          enhanceSidebar();
        }
      });
  }

  function setHighlight(map, feat) {
    var src = map.getSource(SELECT_SOURCE);
    if (!src) return;
    showLoops(map, feat && feat.properties ? feat.properties.h3_index : null);
    var geom = feat ? highlightGeom(feat) : null;
    if (!geom) {
      src.setData(emptySel());
      return;
    }
    src.setData({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: geom, properties: {} }]
    });
  }

  function applyProfile(profile, data, map) {
    currentProfile = profile || "run";
    applyView(data, map);
  }

  function fadeBasemap(map) {
    try {
      map.setPaintProperty("llmaps-base", "raster-saturation", -0.35);
      map.setPaintProperty("llmaps-base", "raster-contrast", -0.08);
      map.setPaintProperty("llmaps-base", "raster-brightness-max", 0.88);
      map.setPaintProperty("llmaps-base", "raster-opacity", 0.82);
    } catch (e) {}
    if (map.getLayer(LAYER_ID)) {
      map.setPaintProperty(LAYER_ID, "fill-opacity", [
        "case",
        ["boolean", ["feature-state", "active"], true],
        0.42,
        0
      ]);
    }
    if (map.getLayer("hex-fill-outline")) {
      try {
        map.setPaintProperty("hex-fill-outline", "line-color", "#5a4a3a");
        map.setPaintProperty("hex-fill-outline", "line-opacity", [
          "case",
          ["boolean", ["feature-state", "active"], true],
          0.55,
          0
        ]);
      } catch (e2) {}
    }
  }

  function setHexLayerVisible(map, on) {
    var vis = on ? "visible" : "none";
    ["hex-fill", "hex-fill-outline"].forEach(function (id) {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
    });
  }

  function hookHexToggle(map) {
    var boxes = document.querySelectorAll('.llmaps-layer-toggle input[data-layer-id="hex-fill"]');
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].addEventListener("change", function (ev) {
        setHexLayerVisible(map, !!ev.currentTarget.checked);
      });
    }
  }

  if (typeof window.llmapsOnLayersReady !== "function") return;
  window.llmapsOnLayersReady(function (map) {
    function reviveMap() {
      if (!map) return;
      try {
        map.getCanvas().style.display = "block";
        map.resize();
        map.triggerRepaint();
      } catch (e) {}
    }
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) requestAnimationFrame(reviveMap);
    });
    window.addEventListener("focus", reviveMap);
    var canvas = map.getCanvas();
    canvas.addEventListener("webglcontextlost", function (ev) {
      ev.preventDefault();
    });
    canvas.addEventListener("webglcontextrestored", reviveMap);
    if (map.getLayer("hex-fill-outline")) {
      map.setPaintProperty("hex-fill-outline", "line-width", 0.35);
      map.setPaintProperty("hex-fill-outline", "line-color", "#8B5A2B");
    }
    fadeBasemap(map);
    hookHexToggle(map);
    ensureSelectLayer(map);
    ensureLoopLayer(map);

    document.addEventListener("click", function (ev) {
      var kmBtn = ev.target.closest("[data-ah-km]");
      if (kmBtn && window._ahSourceData) {
        currentKm[currentProfile] = Number(kmBtn.getAttribute("data-ah-km"));
        applyView(window._ahSourceData, map);
        return;
      }
      var tabBtn = ev.target.closest("[data-ah-tab]");
      if (tabBtn) {
        ahTab = tabBtn.getAttribute("data-ah-tab");
        if (ahTab === "hex") {
          openRank = null;
          closeElevModal();
        }
        drawLoops(map);
        enhanceSidebar();
        return;
      }
      if (ev.target.closest("[data-ah-elev-close]") || ev.target === document.getElementById("ah-elev-modal")) {
        closeElevModal();
        return;
      }
      if (ev.target.closest("[data-ah-elev]")) {
        elevOpen = true;
        renderElevModal();
        return;
      }
      var row = ev.target.closest("[data-ah-rank]");
      if (row) {
        var rank = Number(row.getAttribute("data-ah-rank"));
        openRank = openRank === rank ? null : rank;
        if (openRank == null) closeElevModal();
        drawLoops(map);
        enhanceSidebar();
      }
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && elevOpen) closeElevModal();
    });

    document.addEventListener(
      "click",
      function (ev) {
        var t = ev.target;
        if (t && t.closest && t.closest(".llmaps-sidebar-close")) {
          setHighlight(map, null);
        }
      },
      true
    );
    function hookSidebarClose() {
      var closeSidebar = window.llmapsSidebarClose;
      if (typeof closeSidebar !== "function" || closeSidebar._ah) return;
      var wrapped = function () {
        closeElevModal();
        setHighlight(map, null);
        return closeSidebar.apply(this, arguments);
      };
      wrapped._ah = true;
      window.llmapsSidebarClose = wrapped;
    }
    hookSidebarClose();
    setTimeout(hookSidebarClose, 0);

    var ext = window.llmapsData && window.llmapsData.extent;
    if (ext && ext.bounds) {
      map.fitBounds(ext.bounds, {
        padding: { top: 72, left: 16, right: 16, bottom: 16 },
        duration: 0,
      });
    }

    map.on("mousemove", LAYER_ID, function (ev) {
      var feat = ev.features && ev.features[0];
      var props = feat && feat.properties ? resolveHexProps(feat.properties) : null;
      var ok = props && hexPassesFilters(props);
      map.getCanvas().style.cursor = ok ? "pointer" : "";
    });
    map.on("mouseleave", LAYER_ID, function () {
      map.getCanvas().style.cursor = "";
    });
    map.on("click", LAYER_ID, function (ev) {
      var feat = ev.features && ev.features[0];
      if (!feat || !feat.properties) return;
      var props = resolveHexProps(feat.properties);
      if (!hexPassesFilters(props)) return;
      lastProps = props;
      setHighlight(map, feat);
      requestAnimationFrame(enhanceSidebar);
    });

    var sidebar = document.getElementById("llmaps-sidebar");
    if (sidebar && typeof MutationObserver === "function") {
      observer = new MutationObserver(function () {
        enhanceSidebar();
      });
      observer.observe(sidebar, { childList: true, subtree: true });
      new MutationObserver(function () {
        if (!sidebar.classList.contains("open")) setHighlight(map, null);
        syncSheet();
      }).observe(sidebar, { attributes: true, attributeFilter: ["class"] });
      window.addEventListener("resize", syncSheet);
    }

    window.llmapsGetSourceData(SOURCE_ID).then(function (data) {
      if (!data) return;
      window._ahSourceData = data;
      indexHexGeom(data);
      loadFullHexProps();
      var def = kmCfg().default || {};
      currentKm.run = def.run || 5;
      currentKm.bike = def.bike || 20;
      var buttons = document.querySelectorAll("[data-ah-profile]");
      for (var i = 0; i < buttons.length; i++) {
        buttons[i].addEventListener("click", function (ev) {
          applyProfile(ev.currentTarget.getAttribute("data-ah-profile"), data, map);
        });
      }
      applyView(data, map);
    });

    document.addEventListener("change", function (ev) {
      if (ev.target && ev.target.hasAttribute("data-ah-train") && window._ahSourceData) {
        // legacy: чекбокс убран из шапки; фильтр станции — в блоке «Фильтры»
        wantTrain = !!ev.target.checked;
        applyView(window._ahSourceData, map);
      }
    });
  });
})();
"""
    )

    m.save(OUTPUT_HTML)
    html = OUTPUT_HTML.read_text(encoding="utf-8")
    html = re.sub(
        r'"visibility_optimization_source_ids"\s*:\s*\[[^\]]*\]',
        '"visibility_optimization_source_ids": []',
        html,
        count=1,
    )
    # FileSource bakes in an absolute path, which only resolves from file:// on this
    # machine. Relative keeps the map working over HTTP and on static hosting.
    baked = json.dumps(str(PREPARED_GEOJSON))[1:-1]
    if baked in html:
        html = html.replace(baked, PREPARED_GEOJSON.name)
    else:
        print("ВНИМАНИЕ: абсолютный путь к гексам не найден, проверьте вручную")
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved map to {OUTPUT_HTML}")
    print("quantile run", stops_run)
    print("quantile bike", stops_bike)


if __name__ == "__main__":
    main()
