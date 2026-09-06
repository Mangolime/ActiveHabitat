"""Circular sample routes: one far point + two near-disjoint paths.

Double-path heuristic (Lewis & Corcoran, Journal of Heuristics 2022 — fixed-length
circuits in street networks), with three changes over the first version:
  * far points are picked by **network** distance from the Dijkstra tree (~L/2.2),
    not by straight line, so the loop length is steered instead of filtered;
  * the straight line is kept only as a shape test: a round loop of length L has
    path/straight ~ pi/2 on each half;
  * everything runs on a local slice of the cached city table (see loop_net.py)
    with scipy CSR Dijkstra, so no Overpass calls and no per-hex re-weighting.

A node farther than L/2 from the start cannot lie on a loop of length L, which
bounds the slice radius exactly instead of by a guessed buffer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_net  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CRS_METRIC = "EPSG:32637"

SOKOLNIKI = "8811aa6301fffff"
TARGET = {"run": 5000.0, "bike": 15000.0}
KM_OPTIONS = {"run": [5, 10, 15], "bike": [10, 20, 30]}
LOOPS_DIR = ROOT / "web" / "loops"

# outbound leg as a share of the target: the return is 1.1-1.3x longer because the
# corridor penalty pushes it wide, so aiming at L/2 would overshoot
OUT_FRAC = (1.0 / 2.40, 1.0 / 1.95)
OUT_FRAC_WIDE = (1.0 / 3.00, 1.0 / 1.70)   # reachable on a retry after a length miss
DETOUR_BAND = (1.25, 2.20)        # outbound path / straight line; circle gives pi/2
N_FAR = 12
N_AZIMUTH_BINS = 12
MAX_ATTEMPTS = 3                  # per direction, correcting the far point each time

LENGTH_TOL = 0.20
MIN_Q = 0.26                      # isoperimetric quotient floor
MIN_DETOUR = 1.20                 # loop half-length / straight line to the far point
MAX_REPEAT = 0.18                 # share of length walked twice
REUSE_MULT = 400.0                # outbound edges: forbidden unless it is a bridge
CORRIDOR_MULT = 5.0
CORRIDOR_M = {"run": 220.0, "bike": 600.0}
OVERLAP_IOU = 0.65        # hard cap: above this it is the same route twice
OVERLAP_W = 0.5           # price of overlap, in score points per unit of IoU
GOOD_W = 1.5              # a route is the evidence for the hex score, so parks lead
ROUTE_BUF_M = {"run": 160.0, "bike": 280.0}
N_ROUTES = 2
SNAP_MAX_M = 600.0                # res-8 hexes are ~460 m across, so this always hits


# ---------------------------------------------------------------- local network


class LocalNet:
    """Undirected graph slice backed by numpy arrays and a scipy CSR matrix."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        u, v, node_x, node_y = loop_net.node_ids(gdf)
        keep = u != v
        gdf = gdf[keep].reset_index(drop=True)
        u, v = u[keep], v[keep]

        # collapse parallel edges, keeping the cheapest
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        key = lo.astype(np.int64) * len(node_x) + hi
        order = np.lexsort((gdf["w"].to_numpy(), key))
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        sel = np.sort(order[first])

        self.gdf = gdf.iloc[sel].reset_index(drop=True)
        self.u, self.v = u[sel], v[sel]
        self.node_x, self.node_y = node_x, node_y
        self.length = self.gdf["length_m"].to_numpy(dtype=float)
        self.w = self.gdf["w"].to_numpy(dtype=float)
        self.good = self.gdf["good"].to_numpy(dtype=bool)
        self.in_park = self.gdf["in_park"].to_numpy(dtype=bool)
        self.is_bike = self.gdf["is_bike"].to_numpy(dtype=bool)
        self.near_stress = self.gdf["near_stress"].to_numpy(dtype=bool)
        self.surf = self.gdf["surf"].to_numpy(dtype=np.int8)
        self.geoms = self.gdf.geometry.values
        self.n_nodes = len(node_x)
        self.n_edges = len(self.u)

        row = np.concatenate([self.u, self.v])
        col = np.concatenate([self.v, self.u])
        eid = np.concatenate([np.arange(self.n_edges), np.arange(self.n_edges)])
        self._csr = coo_matrix(
            (eid.astype(np.float64) + 1.0, (row, col)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        self._order = self._csr.data.astype(np.int64) - 1  # edge index per CSR slot
        self._edge_of = np.concatenate([np.arange(self.n_edges), np.arange(self.n_edges)])[self._order]
        self._csr.data = np.concatenate([self.w, self.w])[self._order]

        # (node pair) -> edge index, for reading a path back off the predecessors
        self._pair = {}
        for i in range(self.n_edges):
            self._pair[(int(self.u[i]), int(self.v[i]))] = i
            self._pair[(int(self.v[i]), int(self.u[i]))] = i

        self._kdt = cKDTree(np.column_stack([node_x, node_y]))
        self._sindex = self.gdf.sindex

    def set_weights(self, w: np.ndarray) -> None:
        self._csr.data = np.concatenate([w, w])[self._order]

    def shortest(self, source: int):
        dist, pred = dijkstra(self._csr, directed=False, indices=source, return_predecessors=True)
        return dist, pred

    def snap(self, pt: Point):
        d, i = self._kdt.query([pt.x, pt.y])
        return (None, d) if d > SNAP_MAX_M else (int(i), float(d))

    def walk(self, pred: np.ndarray, source: int, target: int):
        """Edge indices along the predecessor path, source -> target."""
        nodes = [int(target)]
        cur = int(target)
        while cur != source:
            nxt = int(pred[cur])
            if nxt < 0:
                return None, None
            nodes.append(nxt)
            cur = nxt
            if len(nodes) > 100000:
                return None, None
        nodes.reverse()
        edges = []
        for a, b in zip(nodes, nodes[1:]):
            i = self._pair.get((a, b))
            if i is None:
                return None, None
            edges.append(i)
        return nodes, np.asarray(edges, dtype=int)

    def line(self, nodes, edges) -> LineString | None:
        coords: list[tuple[float, float]] = []
        for node, ei in zip(nodes, edges):
            seg = list(self.geoms[ei].coords)
            ax, ay = self.node_x[node], self.node_y[node]
            if (seg[-1][0] - ax) ** 2 + (seg[-1][1] - ay) ** 2 < (seg[0][0] - ax) ** 2 + (
                seg[0][1] - ay
            ) ** 2:
                seg.reverse()
            coords.extend(seg[1:] if coords else seg)
        return LineString(coords) if len(coords) > 1 else None

    def corridor_mask(self, line: LineString, radius: float) -> np.ndarray:
        mask = np.zeros(self.n_edges, dtype=bool)
        hits = self._sindex.query(line.buffer(radius), predicate="intersects")
        mask[np.asarray(hits, dtype=int)] = True
        return mask


def metres_along_tree(net: LocalNet, dist_w: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """True length in metres of each tree path, accumulated in cost order."""
    out = np.full(net.n_nodes, np.inf)
    finite = np.isfinite(dist_w)
    order = np.argsort(np.where(finite, dist_w, np.inf))
    for node in order:
        p = pred[node]
        if p < 0:
            out[node] = 0.0 if dist_w[node] == 0 else np.inf
            continue
        if not np.isfinite(out[p]):
            continue
        ei = net._pair.get((int(p), int(node)))
        if ei is None:
            continue
        out[node] = out[p] + net.length[ei]
    return out


# ---------------------------------------------------------------- geometry bits


def isoperimetric(line) -> float:
    """Q = 4*pi*A/L^2 on the closed route; 1.0 is a circle, ~0 an out-and-back."""
    if line is None or line.is_empty:
        return 0.0
    try:
        polys = list(polygonize(unary_union(line)))
    except Exception:
        return 0.0
    length = line.length
    if length <= 0:
        return 0.0
    return 4.0 * math.pi * sum(p.area for p in polys) / (length * length)


def iou_poly(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 1.0
    union = a.union(b).area
    return 1.0 if union <= 0 else a.intersection(b).area / union


# ---------------------------------------------------------------- far points


def far_points(net: LocalNet, s: int, dist_m: np.ndarray, target: float) -> list[dict]:
    """One bundle of candidate far points per azimuth sector.

    The first pick in a sector is the node whose network distance suggests a loop
    of about `target` and whose detour is closest to pi/2. The rest of the bundle
    is kept so a length miss can be corrected in the same direction instead of
    throwing the direction away.
    """
    wide_lo, wide_hi = target * OUT_FRAC_WIDE[0], target * OUT_FRAC_WIDE[1]
    ok = np.isfinite(dist_m) & (dist_m >= wide_lo) & (dist_m <= wide_hi)
    if not ok.any():
        return []

    sx, sy = net.node_x[s], net.node_y[s]
    dx = net.node_x - sx
    dy = net.node_y - sy
    straight = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        detour = np.where(straight > 1, dist_m / np.maximum(straight, 1e-9), 0.0)
    ok &= (detour >= DETOUR_BAND[0]) & (detour <= DETOUR_BAND[1])
    if not ok.any():
        return []

    pref_node = np.zeros(net.n_nodes, dtype=bool)
    g = net.good
    pref_node[net.u[g]] = True
    pref_node[net.v[g]] = True

    idx = np.flatnonzero(ok)
    azimuth = (np.degrees(np.arctan2(dy[idx], dx[idx])) + 360.0) % 360.0
    binw = 360.0 / N_AZIMUTH_BINS
    bin_of = np.minimum((azimuth / binw).astype(int), N_AZIMUTH_BINS - 1)
    narrow = (dist_m[idx] >= target * OUT_FRAC[0]) & (dist_m[idx] <= target * OUT_FRAC[1])
    quality = np.abs(detour[idx] - math.pi / 2) - 1.5 * pref_node[idx] + 1.0 * ~narrow

    bundles = []
    for b in range(N_AZIMUTH_BINS):
        sel = np.flatnonzero(bin_of == b)
        if not len(sel):
            continue
        sel = sel[np.argsort(quality[sel])]
        bundles.append(
            {
                "nodes": idx[sel],
                "out_m": dist_m[idx[sel]],
                "straight_m": straight[idx[sel]],
                "quality": float(quality[sel[0]]),
                "kind": "preferred" if pref_node[idx[sel[0]]] else "plain",
            }
        )
    bundles.sort(key=lambda p: p["quality"])
    return bundles[:N_FAR]


# ---------------------------------------------------------------- candidates


def build_candidate(net: LocalNet, e1, e2, target: float, profile: str, straight_m: float):
    length = float(net.length[e1].sum() + net.length[e2].sum())
    if length <= 0:
        return None
    shared = np.intersect1d(e1, e2, assume_unique=False)
    repeat = 2.0 * float(net.length[shared].sum()) / length
    loop = np.concatenate([e1, e2])
    good = float(net.length[loop][net.good[loop]].sum())
    return {
        "length": length,
        "good": good,
        "repeat": repeat,
        "park_m": float(net.length[loop][net.in_park[loop]].sum()),
        "bike_m": float(net.length[loop][net.is_bike[loop]].sum()),
        "stress_m": float(net.length[loop][net.near_stress[loop]].sum()),
        "surf_asphalt_m": float(net.length[loop][net.surf[loop] == loop_net.SURF_ASPHALT].sum()),
        "surf_stone_m": float(net.length[loop][net.surf[loop] == loop_net.SURF_STONE_CODE].sum()),
        "surf_soil_m": float(net.length[loop][net.surf[loop] == loop_net.SURF_SOIL].sum()),
        "detour": (length / 2.0) / straight_m if straight_m > 1 else 0.0,
        "rel": abs(length - target) / target,
    }


def try_far(net: LocalNet, s: int, pred, t: int, profile: str, target: float, straight_m: float):
    """Build one loop through far point `t`. Returns (candidate, reason, length)."""
    nodes1, e1 = net.walk(pred, s, t)
    if e1 is None or not len(e1):
        return None, "path", 0.0
    line1 = net.line(nodes1, e1)
    if line1 is None:
        return None, "path", 0.0

    w2 = net.w * np.where(net.corridor_mask(line1, CORRIDOR_M[profile]), CORRIDOR_MULT, 1.0)
    w2[e1] = net.w[e1] * REUSE_MULT
    net.set_weights(w2)
    _, pred2 = net.shortest(s)
    nodes2, e2 = net.walk(pred2, s, t)
    net.set_weights(net.w)
    if e2 is None or not len(e2):
        return None, "path", 0.0

    cand = build_candidate(net, e1, e2, target, profile, straight_m)
    if cand is None:
        return None, "path", 0.0
    got = cand["length"]
    if cand["rel"] > LENGTH_TOL:
        return None, "len", got
    if cand["repeat"] > MAX_REPEAT:
        return None, "repeat", got
    if cand["detour"] < MIN_DETOUR:
        return None, "detour", got

    loop_nodes = list(nodes1) + list(reversed(nodes2))[1:]
    loop_edges = list(e1) + list(reversed(e2))
    line = net.line(loop_nodes, loop_edges)
    if line is None:
        return None, "path", got
    q = isoperimetric(line)
    if q < MIN_Q:
        return None, "q", got
    cand["q"] = q
    cand["geom_m"] = line
    cand["poly"] = line.buffer(ROUTE_BUF_M[profile], cap_style=2, join_style=2)
    cand["score"] = (
        GOOD_W * cand["good"] / cand["length"]
        + 0.6 * q
        - 1.2 * cand["rel"]
        - 1.5 * cand["repeat"]
        - 1.0 * cand["stress_m"] / cand["length"]
    )
    return cand, "", got


def routes_for_hex(net: LocalNet, start_pt: Point, profile: str, target: float, verbose: bool):
    s, snap_d = net.snap(start_pt)
    if s is None:
        return [], "нет стартового узла"

    net.set_weights(net.w)
    dist_w, pred = net.shortest(s)
    dist_m = metres_along_tree(net, dist_w, pred)
    targets = far_points(net, s, dist_m, target)
    if verbose:
        kinds = pd.Series([t["kind"] for t in targets]).value_counts().to_dict() if targets else {}
        print(f"      дальних точек {len(targets)} {kinds}")
    if not targets:
        return [], "нет дальних точек"

    candidates = []
    rejected = {"q": 0, "len": 0, "repeat": 0, "detour": 0, "path": 0}
    tried = 0
    for bundle in targets:
        want = None
        seen: set[int] = set()
        for _ in range(MAX_ATTEMPTS):
            if want is None:
                k = 0
            else:
                free = np.array([i for i in range(len(bundle["nodes"])) if i not in seen])
                if not len(free):
                    break
                k = int(free[int(np.argmin(np.abs(bundle["out_m"][free] - want)))])
            if k in seen:
                break
            seen.add(k)
            tried += 1
            cand, reason, got = try_far(
                net, s, pred, int(bundle["nodes"][k]), profile, target,
                float(bundle["straight_m"][k]),
            )
            if cand is not None:
                candidates.append(cand)
                break
            rejected[reason] += 1
            if reason != "len":
                break
            # aim the next far point at the distance the miss implies
            want = bundle["out_m"][k] * target / max(got, 1.0)

    if verbose:
        print(f"      проб {tried}, годных {len(candidates)}, отсев {rejected}")
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Diversity is a price, not a veto: an overlapping route still wins if it buys
    # enough park and path back. Only near-duplicates are refused outright.
    chosen = []
    pool = list(candidates)
    while pool and len(chosen) < N_ROUTES:
        best, best_eff = None, -math.inf
        for c in pool:
            overlap = max((iou_poly(c["poly"], p["poly"]) for p in chosen), default=0.0)
            if overlap > OVERLAP_IOU:
                continue
            eff = c["score"] - OVERLAP_W * overlap
            if eff > best_eff:
                best, best_eff = c, eff
        if best is None:
            break
        pool.remove(best)
        chosen.append(best)
        if verbose and len(chosen) == 1:
            print("      пул кандидатов (хороших / Q / пересечение с #1 / оценка):")
            for c in sorted(pool, key=lambda c: c["score"], reverse=True):
                ov = iou_poly(c["poly"], best["poly"])
                print(
                    f"         {100*c['good']/c['length']:3.0f}%  Q {c['q']:.2f}"
                    f"  IoU {ov:.2f}  {c['score']:+.2f} → {c['score'] - OVERLAP_W*ov:+.2f}"
                )
    return chosen, ""


# ---------------------------------------------------------------- driver


def _parse_km(text: str | None, profile: str) -> list[int]:
    if text is None or text.strip() == "":
        return [int(round(TARGET[profile] / 1000.0))]
    if text.strip() in ("-", "none"):
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def infer_km(props: dict) -> int:
    if props.get("km") is not None:
        return int(props["km"])
    return 5 if props.get("profile") == "run" else 15


def backfill_km(folder: Path = LOOPS_DIR) -> int:
    n = 0
    if not folder.exists():
        return 0
    for path in folder.glob("*.json"):
        fc = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for feat in fc.get("features") or []:
            p = feat.setdefault("properties", {})
            if p.get("km") is None:
                p["km"] = infer_km(p)
                changed = True
        if changed:
            path.write_text(
                json.dumps(fc, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            n += 1
    return n


def merge_hex(h3_index: str, profile: str, km: int, new_feats: list, folder: Path = LOOPS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{h3_index}.json"
    old = []
    if path.exists():
        fc = json.loads(path.read_text(encoding="utf-8"))
        for feat in fc.get("features") or []:
            p = feat.setdefault("properties", {})
            if p.get("km") is None:
                p["km"] = infer_km(p)
            if p.get("profile") == profile and int(p["km"]) == int(km):
                continue
            old.append(feat)
    for feat in new_feats:
        g = feat["geometry"]
        g["coordinates"] = [[round(x, 5), round(y, 5)] for x, y in g["coordinates"]]
        feat.setdefault("properties", {})["km"] = int(km)
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": old + new_feats},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def build_tasks(mode: str, pct: float, km_by_profile: dict[str, list[int]]):
    """(hex, x, y, profile, target_m, km) for the hexes worth a route."""
    hexes = gpd.read_parquet(DATA / "moscow_grid_h3.parquet")[["h3_index", "geometry"]]
    hexes = hexes.to_crs(CRS_METRIC)
    scored_path = DATA / "moscow_h3_scored_new.geojson"
    if not scored_path.exists():
        scored_path = DATA / "moscow_h3_scored.geojson"
    scored = gpd.read_file(scored_path)[["h3_index", "score_run", "score_bike"]]

    if mode == "sample":
        want = {SOKOLNIKI, scored.nlargest(1, "score_bike")["h3_index"].iloc[0],
                scored.sort_values("score_run").iloc[len(scored) // 2]["h3_index"]}
        chosen = {"run": want, "bike": want}
    elif mode == "all":
        all_h = set(scored["h3_index"])
        chosen = {"run": all_h, "bike": all_h}
        print(f"   все гексы: {len(all_h)}")
    else:
        chosen = {}
        for profile, col in (("run", "score_run"), ("bike", "score_bike")):
            s = scored[col].fillna(0)
            thr = float(s.quantile(pct / 100.0))
            chosen[profile] = set(scored.loc[s >= thr, "h3_index"])
            print(f"   {profile}: порог {thr:.0f}, гексов {len(chosen[profile])}")

    cent = hexes.set_index("h3_index").geometry.centroid
    tasks = []
    for profile, kms in km_by_profile.items():
        for h in sorted(chosen[profile]):
            if h in cent.index:
                for km in kms:
                    tasks.append(
                        (h, float(cent[h].x), float(cent[h].y), profile, float(km) * 1000.0, int(km))
                    )
    return tasks


def to_feature(h3_index, profile, rank, rt, geom_wgs, km: int | None = None):
    feat = {
        "type": "Feature",
        "properties": {
            "h3_index": h3_index,
            "profile": profile,
            "km": int(km) if km is not None else infer_km({"profile": profile}),
            "rank": rank,
            "length_m": int(round(rt["length"])),
            "good_m": int(round(rt["good"])),
            "good_share": round(rt["good"] / rt["length"], 3) if rt["length"] else 0,
            "park_m": int(round(rt["park_m"])),
            "bike_m": int(round(rt["bike_m"])),
            "stress_share": round(rt["stress_m"] / rt["length"], 3) if rt["length"] else 0,
            "surf_asphalt": round(rt["surf_asphalt_m"] / rt["length"], 3) if rt["length"] else 0,
            "surf_stone": round(rt["surf_stone_m"] / rt["length"], 3) if rt["length"] else 0,
            "surf_soil": round(rt["surf_soil_m"] / rt["length"], 3) if rt["length"] else 0,
            "repeat_share": round(rt["repeat"], 3),
            "roundness_q": round(rt["q"], 3),
            "detour": round(rt["detour"], 2),
        },
        "geometry": json.loads(gpd.GeoSeries([geom_wgs], crs=4326).to_json())["features"][0]["geometry"],
    }
    try:
        import dem

        dem.try_annotate(feat)
    except Exception:
        pass
    return feat


VERBOSE = False
SIMPLIFY_M = 0.0


def report(res, n, total, t0):
    h3_index, profile, km, feats, err, secs = res
    if VERBOSE or not feats or n % 25 == 0:
        done = time.time() - t0
        eta = done / n * (total - n)
        print(
            f"   [{n}/{total}] {h3_index} {profile} {km}км: петель {len(feats)}"
            f" за {secs:.1f} с {err} | осталось ~{eta/60:.0f} мин"
        )
    if VERBOSE:
        for f in feats:
            p = f["properties"]
            print(
                f"      #{p['rank']} {p['length_m']/1000:.2f} км |"
                f" хороших {100*p['good_share']:.0f}% |"
                f" вдоль магистралей {100*p['stress_share']:.0f}% |"
                f" Q {p['roundness_q']:.2f} | повтор {100*p['repeat_share']:.0f}%"
            )
    return res


def run_task(task):
    h3_index, x, y, profile, target, km = task
    start_pt = Point(x, y)
    tick = time.time()
    try:
        gdf = loop_net.load_local(profile, start_pt, target / 2.0)
        if len(gdf) < 50:
            return h3_index, profile, km, [], f"мало рёбер ({len(gdf)})", time.time() - tick
        net = LocalNet(gdf)
        routes, err = routes_for_hex(net, start_pt, profile, target, VERBOSE)
    except Exception as exc:
        return h3_index, profile, km, [], f"{type(exc).__name__}: {exc}", time.time() - tick

    if not routes:
        return h3_index, profile, km, [], err, time.time() - tick
    geoms = gpd.GeoSeries([r["geom_m"] for r in routes], crs=CRS_METRIC)
    if SIMPLIFY_M > 0:
        geoms = geoms.simplify(SIMPLIFY_M)
    geoms = geoms.to_crs(4326)
    feats = [
        to_feature(h3_index, profile, rank, rt, gw, km)
        for rank, (rt, gw) in enumerate(zip(routes, geoms), start=1)
    ]
    return h3_index, profile, km, feats, err, time.time() - tick


def main():
    global VERBOSE, SIMPLIFY_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "top", "all"], default="sample")
    ap.add_argument("--pct", type=float, default=75.0, help="score percentile cutoff")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--simplify", type=float, default=0.0, help="Douglas-Peucker, metres")
    ap.add_argument("--run-km", default="", help="comma list, e.g. 10,15")
    ap.add_argument("--bike-km", default="", help="comma list, e.g. 10,20,30")
    args = ap.parse_args()
    SIMPLIFY_M = args.simplify
    km_by_profile = {
        "run": _parse_km(args.run_km, "run"),
        "bike": _parse_km(args.bike_km, "bike"),
    }

    for profile in ("run", "bike"):
        loop_net.build(profile)

    stamped = backfill_km()
    if stamped:
        print(f"   km в старых файлах: {stamped}")

    print("1. Гексы...")
    tasks = build_tasks(args.mode, args.pct, km_by_profile)
    VERBOSE = len(tasks) <= 8
    print(f"   задач: {len(tasks)} ({km_by_profile})")

    n_ok = n_lines = 0
    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp

        with mp.Pool(args.jobs) as pool:
            for n, res in enumerate(pool.imap_unordered(run_task, tasks, chunksize=2), start=1):
                report(res, n, len(tasks), t0)
                h3_index, profile, km, feats, _err, _secs = res
                merge_hex(h3_index, profile, km, feats)
                n_ok += 1
                n_lines += len(feats)
    else:
        for n, task in enumerate(tasks, start=1):
            res = report(run_task(task), n, len(tasks), t0)
            h3_index, profile, km, feats, _err, _secs = res
            merge_hex(h3_index, profile, km, feats)
            n_ok += 1
            n_lines += len(feats)

    biggest = max((p.stat().st_size for p in LOOPS_DIR.glob("*.json")), default=0)
    print(f"   по гексам: {len(list(LOOPS_DIR.glob('*.json')))} файлов, самый большой {biggest/1024:.0f} КБ")
    print(f"\nСохранено линий: {n_lines} из {n_ok} задач за {time.time()-t0:.0f} с")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
