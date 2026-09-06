"""ПРОТОТИП, удалить после выбора дизайна.

Считает разбивку по покрытиям для нескольких готовых маршрутов, чтобы понять,
осмысленна ли круговая диаграмма вообще. Правило для пустого surface задано
пользователем: track/path — грунт, остальное — асфальт.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CRS_METRIC = "EPSG:32637"
MATCH_M = 6.0

HARD = {
    "asphalt", "concrete", "paved", "chipseal", "concrete:plates",
    "concrete:lanes", "metal", "wood", "rubber", "tartan", "acrylic",
}
STONE = {
    "paving_stones", "sett", "cobblestone", "unhewn_cobblestone",
    "bricks", "brick", "paving_stones:30", "grass_paver",
}
SOFT = {
    "ground", "dirt", "earth", "mud", "sand", "grass", "gravel",
    "fine_gravel", "compacted", "pebblestone", "woodchips", "unpaved",
    "clay", "snow", "ice",
}
SOFT_HIGHWAY = {"track", "path", "bridleway"}


def classify(surface: str, highway: str) -> str:
    s = (surface or "").split(";")[0].strip().lower()
    if s in HARD:
        return "asphalt"
    if s in STONE:
        return "stone"
    if s in SOFT:
        return "soil"
    hw = (highway or "").split(";")[0].strip().lower()
    return "soil" if hw in SOFT_HIGHWAY else "asphalt"


def main():
    loops = gpd.read_file(DATA / "sample_loops.gpkg", layer="loops").to_crs(CRS_METRIC)
    hexes = sorted(loops["h3_index"].unique())[:0] or []
    # два гекса: Сокольники и тот, что смотрели в браузере
    want = [h for h in ("8811aa6301fffff", "8811aa630dfffff") if h in set(loops["h3_index"])]
    if not want:
        want = list(loops["h3_index"].unique()[:2])
    sel = loops[loops["h3_index"].isin(want)].reset_index(drop=True)
    print(f"маршрутов: {len(sel)} по гексам {want}")

    pad_deg = 0.01
    w, s, e, n = sel.to_crs(4326).total_bounds
    print("читаю рёбра из GPKG в окрестности...")
    edges = gpd.read_file(
        DATA / "moscow_active_edges.gpkg",
        layer="edges",
        bbox=(w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg),
    ).to_crs(CRS_METRIC)
    print(f"рёбер рядом: {len(edges)}")
    edges["kind"] = [
        classify(s, h) for s, h in zip(edges.get("surface", ""), edges.get("highway", ""))
    ]
    edges["surface_raw"] = edges.get("surface", pd.Series("", index=edges.index)).fillna("")

    out = {}
    for row in sel.itertuples(index=False):
        band = row.geometry.buffer(MATCH_M)
        near = edges[edges.intersects(band)]
        if near.empty:
            continue
        clipped = near.geometry.intersection(band)
        lengths = clipped.length
        by = pd.DataFrame({"kind": near["kind"].values, "m": lengths.values})
        agg = by.groupby("kind")["m"].sum()
        total = float(agg.sum())
        filled = float(near.loc[near["surface_raw"] != "", :].geometry.intersection(band).length.sum())
        key = f"{row.h3_index}|{row.profile}|{row.rank}"
        out[key] = {
            "share": {k: round(float(v) / total, 3) for k, v in agg.items()},
            "tagged_share": round(filled / total, 3) if total else 0,
            "length_m": int(row.length_m),
        }
        print(
            f"  {key}: " + ", ".join(f"{k} {100*v/total:.0f}%" for k, v in agg.items())
            + f" | тегом задано {100*filled/total:.0f}%"
        )

    (ROOT / "web" / "_proto_surface.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\nсохранено web/_proto_surface.json")


if __name__ == "__main__":
    main()
