"""Add FABDEM profiles to existing per-hex loop JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOOPS = ROOT / "web" / "loops"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("1. DEM...")
    src = dem.load_dem()
    files = sorted(LOOPS.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    print(f"   петель: {len(files)} файлов")

    t0 = time.time()
    n_feat = n_ok = 0
    for i, path in enumerate(files, start=1):
        fc = json.loads(path.read_text(encoding="utf-8"))
        for feat in fc.get("features") or []:
            n_feat += 1
            dem.annotate_feature(feat, src)
            if "gain_m" in (feat.get("properties") or {}):
                n_ok += 1
        path.write_text(
            json.dumps(fc, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        if i % 200 == 0 or i == len(files):
            print(f"   [{i}/{len(files)}] профилей {n_ok}/{n_feat}")

    print(f"Готово: {n_ok}/{n_feat} за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
