"""Walking distance from hex centres to parks and water along the street graph.

Straight-line distance lies exactly where it matters most: a park can sit 40 m
away across a river and still cost four kilometres of walking to the nearest
bridge. We reuse the routing table that `loop_net` already builds for the whole
city — only the endpoint columns are needed, so edge geometry is never read.

The whole city is one multi-source Dijkstra per target set: every node inside a
park starts at zero, and scipy relaxes outwards from all of them at once. That
is one pass over 1.2 M edges, not one pass per park.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import loop_net

CRS_METRIC = "EPSG:32637"

# Ребро может пройти через парк одним концом, а другим торчать за 300 м от него,
# поэтому целями считаем узлы, а не концы задетых рёбер.
ENTRY_M = 20.0  # запасной допуск для парков, внутри которых троп в OSM нет
SNAP_MAX_M = 600.0  # гекс res-8 около 460 м в поперечнике, дальше искать незачем


class CityGraph:
    """Undirected street graph of the whole city, weighted by real metres."""

    def __init__(self, profile: str = "run") -> None:
        cols = ["ux", "uy", "vx", "vy", "length_m"]
        edges = pd.read_parquet(loop_net.cache_path(profile), columns=cols)
        u, v, self.x, self.y = loop_net.node_ids(edges)
        w = edges["length_m"].to_numpy(dtype=float)

        keep = u != v
        u, v, w = u[keep], v[keep], w[keep]

        # Рёбра лежат в файле в обе стороны — наследие MultiDiGraph. coo_matrix
        # СУММИРУЕТ дубликаты пар, так что без схлопывания вес каждого ребра
        # удваивается и весь город оказывается ровно вдвое длиннее, чем он есть.
        n = len(self.x)
        key = np.minimum(u, v).astype(np.int64) * n + np.maximum(u, v)
        order = np.lexsort((w, key))
        sel = order[np.unique(key[order], return_index=True)[1]]
        self.u, self.v, self.w = u[sel], v[sel], w[sel]
        self.n = n

        self._csr = coo_matrix(
            (np.concatenate([self.w, self.w]),
             (np.concatenate([self.u, self.v]), np.concatenate([self.v, self.u]))),
            shape=(n, n),
        ).tocsr()
        self._kdt = cKDTree(np.column_stack([self.x, self.y]))
        self._nodes = None

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        if self._nodes is None:
            self._nodes = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(self.x, self.y), crs=CRS_METRIC
            )
        return self._nodes

    def targets_in(
        self,
        polys: gpd.GeoDataFrame,
        entry_m: float = ENTRY_M,
        barriers: gpd.GeoDataFrame | None = None,
    ) -> np.ndarray:
        """Nodes you can stand on and call yourself 'there'."""
        if polys is None or polys.empty:
            return np.empty(0, dtype=np.int64)
        polys = polys[["geometry"]].reset_index(drop=True)
        hit = gpd.sjoin(self.nodes, polys, predicate="within", how="inner")
        inside = np.unique(hit.index.to_numpy())
        if entry_m <= 0:
            return inside

        # Половина зелени в Новой Москве — это безымянный natural=wood, внутри
        # которого в OSM нет ни одной тропы. Строго по узлам такой лес выглядит
        # недостижимым: у гекса 8811aa4da5fffff до леса 37 м, а обход — 4.4 км.
        # Поэтому для таких полигонов — и только для них — засчитываем дорожку
        # впритык к границе.
        got = set(hit["index_right"].unique())
        blind = polys.drop(index=list(got), errors="ignore").reset_index(drop=True)
        if blind.empty:
            return inside
        ring = blind.copy()
        ring["geometry"] = ring.geometry.buffer(entry_m)
        cand = gpd.sjoin(self.nodes, ring, predicate="within", how="inner")
        if cand.empty:
            return inside

        # Но допуск не должен перепрыгивать преграду: тротуар на том берегу тоже
        # попадёт в кольцо вокруг парка. Оставляем узел, только если отрезок до
        # парка не пересекает воду, — иначе вернём ровно ту ошибку прямой линии,
        # ради которой мы и ушли на граф.
        if barriers is not None and not barriers.empty:
            legs = shapely.shortest_line(
                self.nodes.geometry.values[cand.index.to_numpy()],
                blind.geometry.values[cand["index_right"].to_numpy()],
            )
            blocked = np.zeros(len(legs), dtype=bool)
            tree = shapely.STRtree(barriers.geometry.values)
            hits = tree.query(legs, predicate="intersects")
            blocked[np.unique(hits[0])] = True
            cand = cand[~blocked]
        return np.union1d(inside, np.unique(cand.index.to_numpy()))

    def dist_from(self, targets: np.ndarray) -> np.ndarray:
        """Metres from every node to the nearest target, in one sweep."""
        if len(targets) == 0:
            return np.full(self.n, np.inf)
        return dijkstra(self._csr, directed=False, indices=targets, min_only=True)

    def dist_for_points(self, pts: gpd.GeoSeries, node_dist: np.ndarray) -> np.ndarray:
        """Snap each point to the network, then read off the precomputed field."""
        snap_d, snap_i = self._kdt.query(np.column_stack([pts.x, pts.y]))
        out = node_dist[snap_i] + snap_d
        return np.where(snap_d > SNAP_MAX_M, np.inf, out)


def walking_dist(
    graph: CityGraph,
    points: gpd.GeoSeries,
    polys: gpd.GeoDataFrame,
    bank_m: float = 0.0,
    entry_m: float = ENTRY_M,
    barriers: gpd.GeoDataFrame | None = None,
) -> np.ndarray:
    """Metres on foot from each point to the nearest polygon.

    `bank_m` is for water: посреди пруда узлов не бывает, дойти можно только до
    берега, поэтому цель — кольцо вокруг воды. Для парков наоборот: цель это
    сам полигон, внутрь которого заходят тропинки.
    """
    if polys is None or polys.empty:
        return np.full(len(points), np.inf)
    reach = polys[["geometry"]].reset_index(drop=True)
    if bank_m > 0:
        reach = reach.copy()
        reach["geometry"] = reach.geometry.buffer(bank_m)
        entry_m = 0.0

    targets = graph.targets_in(reach, entry_m, barriers)
    d = graph.dist_for_points(points, graph.dist_from(targets))

    pts = gpd.GeoDataFrame(geometry=points.values, crs=CRS_METRIC)
    hit = gpd.sjoin(pts, reach, predicate="within", how="left")
    inside = hit.groupby(level=0)["index_right"].first().notna().to_numpy()
    return np.where(inside, 0.0, d)
