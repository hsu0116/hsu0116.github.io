"""
눌림 후 상승 확인 매수 검증.
  python run_confirm.py --source pack --path ../stock-data --out results_confirm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm as cf  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)
SPLIT = "2022-01-01"


def market_up20(panel):
    C = panel["close"]
    ret = (C / C.ffill().shift(1) - 1).where(lambda x: x.abs() < 0.35)
    idx = (1 + ret.median(axis=1).fillna(0)).cumprod()
    return (idx > idx.rolling(20).mean()).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pack", choices=["pack", "pickle", "csv"])
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--min_n", type=int, default=200)
    ap.add_argument("--out", default="results_confirm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = {"pack": lambda: data.load_pack(a.path), "pickle": lambda: data.load_panel(a.path),
             "csv": lambda: data.load_csv_dir(a.path, a.meta)}[a.source]()
    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    is_m = (ev["date"] < SPLIT).values
    up = market_up20(panel)[ev["t"].values]
    P = en.PRE
    print(f"급등 {len(ev):,}건 ({ev.date.min().date()} ~ {ev.date.max().date()})")

    E, stops = cf.build(w)
    rows, store = [], {}
    for en_name, e in E.items():
        X = cf.exits(w, stops[en_name])
        r = np.arange(len(e.col))
        prem = np.where(e.col >= 0, e.price / stops[en_name] - 1, np.nan)    # 눌림 저점 대비 진입가
        for xn, x in X.items():
            ret, hold, xcol, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost, None, x["close_exit"], x["tp_px"])
            store[(en_name, xn)] = (ret, hold, xcol, valid, e, x)
            depth, ctype = en_name.split("_", 1)
            row = dict(depth=depth, confirm=ctype, exit=xn)
            for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
                mm = valid & pm
                s = en.stats(ret[mm], hold[mm], int(pm.sum()))
                row.update({f"{tag}_{k}": v for k, v in s.items()})
                row[f"{tag}_저점대비진입가"] = np.nanmedian(prem[mm]) if mm.any() else np.nan
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(out / "confirm_grid.csv", index=False, encoding="utf-8-sig")
    print(f"조합 {len(R)}개 (눌림 깊이 {len(cf.DEPTHS)} × 확인방식 {len(cf.CONFIRMS)}+확인없음 × 청산 3)")

    # 1) 확인 효과: 같은 깊이·청산에서 (확인 - 확인없음)
    base = R[R.confirm == "확인없음"].set_index(["depth", "exit"])
    eff = []
    for r in R[R.confirm != "확인없음"].itertuples():
        b = base.loc[(r.depth, r.exit)]
        eff.append(dict(confirm=r.confirm, depth=r.depth, exit=r.exit,
                        dIS=r.IS_wmean - b.IS_wmean, dOOS=r.OOS_wmean - b.OOS_wmean,
                        체결비율=r.IS_n / max(b.IS_n, 1)))
    EF = pd.DataFrame(eff)
    print("\n=== 확인 신호의 효과: 같은 깊이·청산의 '확인 없이 바로 매수' 대비 거래당 차이 (평균) ===")
    print(EF.groupby("confirm")[["dIS", "dOOS", "체결비율"]].mean().sort_values("dOOS", ascending=False)
          .round(4).to_string())
    print("\n깊이별:")
    print(EF.groupby("depth")[["dIS", "dOOS"]].mean().round(4).to_string())
    print("\n청산별:")
    print(EF.groupby("exit")[["dIS", "dOOS"]].mean().round(4).to_string())

    show = ["depth", "confirm", "exit", "IS_n", "IS_wmean", "IS_win", "OOS_n", "OOS_wmean", "OOS_median", "OOS_win",
            "OOS_t", "OOS_저점대비진입가", "OOS_hold"]
    print("\n=== 확인 없이 바로 매수 (기준선) ===")
    print(R[R.confirm == "확인없음"][show].round(4).to_string(index=False))

    ok = R[(R.IS_n >= a.min_n) & (R.OOS_n >= a.min_n // 2)]
    top = ok.sort_values("IS_wmean", ascending=False).head(12)
    print("\n=== 2015~2021 성적 상위 12개 → 2022~ ===")
    print(top[show].round(4).to_string(index=False))
    conf_only = ok[ok.confirm != "확인없음"]
    best = conf_only.loc[conf_only["IS_wmean"].idxmax()]
    bbase = base.loc[(best.depth, best.exit)]
    print(f"\n선택된 확인 규칙(2015~2021 최고): {best.depth} / {best.confirm} / {best.exit}")

    # 2) 계좌: 동시 10종목, 하루 2종목, 시장 상승국면에서만 / 전체
    def pf(key, gate):
        ret, hold, xcol, valid, e, x = store[key]
        m = valid & (up if gate else True)
        tin = (w.t0 + e.col - P)[m]
        tout = (w.t0 + xcol - P)[m]
        s, _ = en.portfolio_trades(w.dates, tin, tout, ret[m], slots=10, start=SPLIT, max_new_per_day=2)
        return s
    bench_e = en.limit_entry(w, w.c0 * 0.75, 20, "bench", "bench")
    bret, bhold, bxcol, bvalid = en.simulate(w, bench_e, 0.15, np.inf, 20, a.cost)
    store[("기존기준", "-")] = (bret, bhold, bxcol, bvalid, bench_e, None)
    cand = [(f"{best.depth}_{best.confirm}", best.exit), (f"{best.depth}_확인없음", best.exit), ("기존기준", "-")]
    names = ["선택된 확인 규칙", "같은 깊이·청산, 확인 없음", "기존 기준(D종가-25% 지정가)"]
    prow = []
    for nm, key in zip(names, cand):
        for gate in (False, True):
            s = pf(key, gate)
            prow.append(dict(전략=nm, 시장상승국면만=gate, **s))
    PF = pd.DataFrame(prow)
    PF.to_csv(out / "confirm_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 계좌 2022~ (동시 10종목, 하루 신규 2종목) ===")
    print(PF.round(4).to_string(index=False))

    # 연도별 (선택 규칙)
    ret, hold, xcol, valid, e, x = store[cand[0]]
    yr = pd.DataFrame({"y": ev.date.dt.year[valid], "r": ret[valid]}).groupby("y")["r"].agg(
        n="size", 평균="mean", 중앙="median", 승률=lambda s: (s > 0).mean())
    print("\n=== 선택된 확인 규칙 연도별 ===")
    print(yr.round(4).to_string())

    # 사전 판정
    pfb = PF.set_index(["전략", "시장상승국면만"])
    checks = [
        ("확인 신호 전체 평균 효과가 두 기간 모두 플러스", EF["dIS"].mean() > 0 and EF["dOOS"].mean() > 0),
        ("선택된 확인 규칙 2022~ 거래당 > 0, t >= 2", best.OOS_wmean > 0 and best.OOS_t >= 2),
        ("선택된 확인 규칙이 같은 깊이 '확인 없음'보다 2022~ +0.5%p 이상", best.OOS_wmean - bbase.OOS_wmean >= 0.005),
        ("계좌(시장상승국면만): 선택 규칙 연수익 > 기존 기준",
         pfb.loc[("선택된 확인 규칙", True), "cagr"] > pfb.loc[("기존 기준(D종가-25% 지정가)", True), "cagr"]),
    ]
    print("\n=== 사전 판정 기준 ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(bool(v) for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
