"""FABDEM elevation: mosaic Moscow tiles and sample route profiles.

DEM is Copernicus GLO-30 with forest/buildings removed (Hawker et al.).
Profiles are a route property, not a hex score.

Sampling: every 25 m, bilinear; 150 m moving average; gain/loss ignore
reversals smaller than 2 m (hysteresis), so 30 m noise does not become
«набор высоты».
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TILE_DIR = RAW / "fabdem"
ZIP_PATH = RAW / "N50E030-N60E040_FABDEM_V1-2.zip"
ZIP_URL = (
    "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
    "N50E030-N60E040_FABDEM_V1-2.zip"
)
DEM_PATH = PROCESSED / "dem_fabdem.tif"
EXTENT_PATH = PROCESSED / "extract_extent.geojson"

STEP_M = 25.0
SMOOTH_M = 150.0
MIN_RISE_M = 2.0
STORE_STEP_M = 50.0
EARTH_M = 6_371_000.0


def extent_bounds(pad_deg: float = 0.03) -> tuple[float, float, float, float]:
    g = json.loads(EXTENT_PATH.read_text(encoding="utf-8"))
    geom = shape(g["features"][0]["geometry"])
    west, south, east, north = geom.bounds
    return west - pad_deg, south - pad_deg, east + pad_deg, north + pad_deg


def tiles_for_bounds(bounds) -> list[str]:
    west, south, east, north = bounds
    names = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            hemi = "N" if lat >= 0 else "S"
            side = "E" if lon >= 0 else "W"
            names.append(f"{hemi}{abs(lat):02d}{side}{abs(lon):03d}")
    return names


def extract_tiles(zip_path: Path = ZIP_PATH, dest: Path = TILE_DIR) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    want = set(tiles_for_bounds(extent_bounds()))
    found: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name.endswith(".tif") or "FABDEM" not in name:
                continue
            stem = name.split("_")[0]
            if stem not in want:
                continue
            out = dest / name
            if not out.exists() or out.stat().st_size < 1_000_000:
                print(f"   extract {name}")
                out.write_bytes(zf.read(info))
            found[stem] = out
    missing = sorted(want - set(found))
    if missing:
        raise FileNotFoundError(f"в архиве нет тайлов FABDEM: {missing}")
    return [found[k] for k in sorted(found)]


def mosaic(tile_paths: list[Path], out_path: Path = DEM_PATH) -> Path:
    bounds = extent_bounds()
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, transform = merge(srcs, bounds=bounds)
        meta = srcs[0].meta.copy()
    finally:
        for s in srcs:
            s.close()
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic_arr)
    print(f"   DEM {mosaic_arr.shape[2]}×{mosaic_arr.shape[1]} → {out_path}")
    return out_path


def ensure_dem() -> Path:
    if DEM_PATH.exists() and DEM_PATH.stat().st_size > 100_000:
        return DEM_PATH
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"нет {ZIP_PATH.name}. Скачайте {ZIP_URL}"
        )
    tiles = extract_tiles()
    return mosaic(tiles)


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def densify(coords, step_m: float = STEP_M):
    """Evenly spaced lon/lat samples along a WGS84 line, plus cumulative metres."""
    pts = [(float(x), float(y)) for x, y in coords if x is not None]
    if len(pts) < 2:
        return np.zeros((0, 2)), np.zeros(0)
    segs = [
        haversine_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        for i in range(len(pts) - 1)
    ]
    total = float(sum(segs))
    if total < step_m:
        return np.array([pts[0], pts[-1]], dtype=float), np.array([0.0, total])
    n = int(total // step_m)
    targets = list(np.arange(0, n + 1) * step_m)
    if total - targets[-1] > 2.0:
        targets.append(total)
    else:
        targets[-1] = total
    xy = []
    si = 0
    acc = 0.0
    for t in targets:
        while si < len(segs) - 1 and acc + segs[si] < t - 1e-6:
            acc += segs[si]
            si += 1
        denom = segs[si] if segs[si] > 1e-9 else 1.0
        frac = min(1.0, max(0.0, (t - acc) / denom))
        lon1, lat1 = pts[si]
        lon2, lat2 = pts[si + 1]
        xy.append((lon1 + (lon2 - lon1) * frac, lat1 + (lat2 - lat1) * frac))
    return np.asarray(xy, dtype=float), np.asarray(targets, dtype=float)


def _smooth(z: np.ndarray, step_m: float, window_m: float) -> np.ndarray:
    half = max(1, int(round((window_m / step_m) / 2)))
    win = 2 * half + 1
    if len(z) < win:
        return z.copy()
    k = np.ones(win, dtype=np.float64) / win
    pad = np.pad(z.astype(np.float64), half, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _gain_loss(z: np.ndarray, min_rise: float = MIN_RISE_M) -> tuple[float, float]:
    """Cumulative ascent/descent; wiggles under min_rise do not flip direction."""
    if len(z) < 2:
        return 0.0, 0.0
    gain = loss = 0.0
    last = float(z[0])
    going = 0
    for raw in z[1:]:
        h = float(raw)
        if going == 0:
            if h - last >= min_rise:
                gain += h - last
                last = h
                going = 1
            elif last - h >= min_rise:
                loss += last - h
                last = h
                going = -1
        elif going == 1:
            if h >= last:
                gain += h - last
                last = h
            elif last - h >= min_rise:
                loss += last - h
                last = h
                going = -1
        else:
            if h <= last:
                loss += last - h
                last = h
            elif h - last >= min_rise:
                gain += h - last
                last = h
                going = 1
    return gain, loss


class DEM:
    def __init__(self, path: Path = DEM_PATH):
        with rasterio.open(path) as src:
            self.arr = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.nodata = src.nodata

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        t = self.transform
        cols = (lons - t.c) / t.a
        rows = (lats - t.f) / t.e
        r0 = np.floor(rows).astype(np.int32)
        c0 = np.floor(cols).astype(np.int32)
        dr = rows - r0
        dc = cols - c0
        h, w = self.arr.shape
        z = np.full(lons.shape, np.nan, dtype=np.float64)
        ok = (r0 >= 0) & (c0 >= 0) & (r0 < h - 1) & (c0 < w - 1)
        if not ok.any():
            return z
        rr, cc = r0[ok], c0[ok]
        z00 = self.arr[rr, cc].astype(np.float64)
        z01 = self.arr[rr, cc + 1].astype(np.float64)
        z10 = self.arr[rr + 1, cc].astype(np.float64)
        z11 = self.arr[rr + 1, cc + 1].astype(np.float64)
        if self.nodata is not None:
            bad = (
                (z00 == self.nodata)
                | (z01 == self.nodata)
                | (z10 == self.nodata)
                | (z11 == self.nodata)
            )
        else:
            bad = ~np.isfinite(z00 + z01 + z10 + z11)
        fr, fc = dr[ok], dc[ok]
        val = (
            z00 * (1 - fr) * (1 - fc)
            + z01 * (1 - fr) * fc
            + z10 * fr * (1 - fc)
            + z11 * fr * fc
        )
        val[bad] = np.nan
        z[ok] = val
        return z

    def profile(self, coords) -> dict | None:
        xy, dist = densify(coords, STEP_M)
        if len(xy) < 4:
            return None
        z = self.sample(xy[:, 0], xy[:, 1])
        if np.isnan(z).mean() > 0.15:
            return None
        if np.isnan(z).any():
            n = np.arange(len(z))
            good = ~np.isnan(z)
            z = np.interp(n, n[good], z[good])
        z = _smooth(z, STEP_M, SMOOTH_M)
        gain, loss = _gain_loss(z)
        stride = max(1, int(round(STORE_STEP_M / STEP_M)))
        stored = z[::stride]
        if stored[-1] != z[-1]:
            stored = np.append(stored, z[-1])
        return {
            "elev": [int(round(v)) for v in stored],
            "elev_step_m": int(STORE_STEP_M),
            "gain_m": int(round(gain)),
            "loss_m": int(round(loss)),
            "elev_min": int(round(float(np.min(z)))),
            "elev_max": int(round(float(np.max(z)))),
        }


_DEM: DEM | None = None


def load_dem() -> DEM:
    global _DEM
    if _DEM is None:
        _DEM = DEM(ensure_dem())
    return _DEM


def annotate_feature(feat: dict, dem: DEM | None = None) -> dict:
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if geom.get("type") != "LineString" or not coords:
        return feat
    src = dem or load_dem()
    prof = src.profile(coords)
    if not prof:
        return feat
    feat.setdefault("properties", {}).update(prof)
    return feat


def try_annotate(feat: dict) -> dict:
    if not DEM_PATH.exists():
        return feat
    try:
        return annotate_feature(feat)
    except Exception:
        return feat
