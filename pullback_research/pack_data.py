"""
받아 둔 일봉 CSV 폴더(kiwoom_csv)를 연도별 압축 파일로 묶는다. GitHub 웹 업로드용(파일당 25MB 미만).

  python pack_data.py --path kiwoom_csv --meta kiwoom_csv/_meta.csv --out data_pack

결과: data_pack/2015.npz ... 2026.npz, data_pack/meta.csv
되읽기: data.load_pack("data_pack")
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

FIELDS = ("open", "high", "low", "close", "volume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="kiwoom_csv")
    ap.add_argument("--meta", default="kiwoom_csv/_meta.csv")
    ap.add_argument("--out", default="data_pack")
    ap.add_argument("--chunk", default="year", choices=["year", "half", "quarter"],
                    help="파일 단위: year(연), half(반기, 약 5MB), quarter(분기, 약 2.5MB)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = data.load_csv_dir(a.path, a.meta if Path(a.meta).exists() else None)
    codes = np.array(panel["close"].columns, dtype="U12")
    dates = panel["close"].index
    if a.chunk == "year":
        label = pd.Index(dates.year.astype(str))
    elif a.chunk == "half":
        label = pd.Index([f"{d.year}H{1 if d.month <= 6 else 2}" for d in dates])
    else:
        label = pd.Index([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in dates])
    years = sorted(set(label))
    total = 0
    for y in years:
        m = (label == y)
        arrs = {f: panel[f].values[m].astype("float32") for f in FIELDS}
        fp = out / f"{y}.npz"
        np.savez_compressed(fp, dates=dates[m].strftime("%Y-%m-%d").values.astype("U10"), codes=codes, **arrs)
        sz = fp.stat().st_size / 1e6
        total += sz
        print(f"{fp.name}: {m.sum()}일 × {len(codes)}종목, {sz:.1f}MB")
    if Path(a.meta).exists():
        shutil.copy(a.meta, out / "meta.csv")
    print(f"완료: {len(years)}개 파일, 총 {total:.0f}MB -> {out}")
    big = [p.name for p in out.glob("*.npz") if p.stat().st_size > 24e6]
    if big:
        print("주의: 25MB 넘는 파일이 있어 GitHub 웹 업로드가 안 됩니다:", big)


if __name__ == "__main__":
    main()
