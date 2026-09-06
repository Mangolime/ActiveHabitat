"""Rewrite park_m on existing loop JSONs from the current parks layer.

Does not reroute. Woods, forests and protected areas count. Overlapping
layers are unioned per route so a metre inside wood+ООПТ is not counted twice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely import STRtree, make_valid, union_all
from shapely import ops
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
LOOPS = ROOT / "web" / "loops"
PARKS = ROOT / "data" / "processed" / "parks_and_water.gpkg"
CRS_METRIC = "EPSG:32637"


def load_parks():
    print("читаю parks_and_water.gpkg...")
    parks = gpd.read_file(PARKS)
    parks = parks[parks["category"] == "park"].to_crs(CRS_METRIC)
    parks["geometry"] = parks.geometry.make_valid()
    parks = parks[~parks.geometry.is_empty]
    geoms = list(parks.geometry.values)
    print(f"   полигонов зелени: {len(geoms)}")
    tree = STRtree(geoms)
    to_m = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True).transform
    return geoms, tree, to_m


def park_metres(geom_wgs, geoms, tree, to_m) -> int:
    line = make_valid(shape(geom_wgs))
    if line.is_empty:
        return 0
    line = ops.transform(to_m, line)
    idx = tree.query(line, predicate="intersects")
    if len(idx) == 0:
        return 0
    bits = [geoms[int(i)].intersection(line) for i in idx]
    merged = union_all(bits)
    if merged.is_empty:
        return 0
    return int(round(merged.length))


def main() -> None:
    geoms, tree, to_m = load_parks()
    files = sorted(LOOPS.glob("*.json"))
    print(f"файлов петель: {len(files)}")
    t0 = time.time()
    n_feat = n_changed = 0
    bitsa = None
    for i, path in enumerate(files, start=1):
        fc = json.loads(path.read_text(encoding="utf-8"))
        dirty = False
        for feat in fc.get("features") or []:
            geom = feat.get("geometry")
            props = feat.setdefault("properties", {})
            if not geom:
                continue
            n_feat += 1
            old = int(props.get("park_m") or 0)
            new = park_metres(geom, geoms, tree, to_m)
            if new != old:
                props["park_m"] = new
                dirty = True
                n_changed += 1
            if (
                path.stem == "8811aa4c89fffff"
                and props.get("profile") == "run"
                and props.get("km") == 5
                and props.get("rank") == 1
            ):
                bitsa = (old, new, int(props.get("length_m") or 0))
        if dirty:
            path.write_text(
                json.dumps(fc, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        if i % 50 == 0 or i == len(files):
            print(f"   [{i}/{len(files)}] фич {n_feat}, правок {n_changed} за {time.time()-t0:.0f} с")
    print(f"готово: файлов {len(files)}, фич {n_feat}, обновлено {n_changed}")
    if bitsa:
        old, new, length = bitsa
        print(f"проверка Битца 8811aa4c89fffff run 5км #1: {old} → {new} м из {length} ({100*new/max(length,1):.0f}%)")


if __name__ == "__main__":
    main()
