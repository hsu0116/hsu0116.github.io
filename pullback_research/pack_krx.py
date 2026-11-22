"""
KRX 전종목 일별 시세(상장폐지 포함)를 받아 연도별 압축 파일로 저장. GitHub 웹 업로드용.
KRX 정보데이터시스템(data.krx.co.kr) 계정 필요: 실행 전에 환경변수 KRX_ID / KRX_PW 설정.

  python pack_krx.py --start 20040101 --end 20150331 --out data_pack_krx

- 2005~2014년 검증용. 앞뒤 여유(2004년: 52주 지표 계산용, 2015년 1분기: 2014년 말 급등의 이후 흐름)를 포함.
- 받은 원본은 cache_krx 폴더에 날짜별로 저장되어, 중간에 끊겨도 다시 실행하면 이어서 받음.
결과: data_pack_krx/krx_2004.npz ... krx_2015.npz, krx_meta.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

FIELDS = ("open", "high", "low", "close", "volume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20040101")
    ap.add_argument("--end", default="20150331")
    ap.add_argument("--out", default="data_pack_krx")
    ap.add_argument("--cache", default="cache_krx")
    a = ap.parse_args()
    if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
        sys.exit("환경변수 KRX_ID / KRX_PW 를 먼저 설정하세요.")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = data.load_krx(a.start, a.end, cache_dir=a.cache)
    codes = np.array(panel["close"].columns, dtype="U12")
    dates = panel["close"].index
    total = 0
    for y in sorted(set(dates.year)):
        m = dates.year == y
        arrs = {f: panel[f].values[m].astype("float32") for f in FIELDS}
        fp = out / f"krx_{y}.npz"
        np.savez_compressed(fp, dates=dates[m].strftime("%Y-%m-%d").values.astype("U10"), codes=codes, **arrs)
        sz = fp.stat().st_size / 1e6
        total += sz
        print(f"{fp.name}: {m.sum()}일 × {len(codes)}종목, {sz:.1f}MB")
    meta = panel["meta"].copy()
    meta.index.name = "code"
    meta.reset_index().to_csv(out / "krx_meta.csv", index=False, encoding="utf-8-sig")
    print(f"완료: 총 {total:.0f}MB -> {out}  (상장폐지 포함 종목 {len(codes):,}개)")


if __name__ == "__main__":
    main()
