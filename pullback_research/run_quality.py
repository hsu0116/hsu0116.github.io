"""
눌림의 질(저가 높이기·거래량 감소·가격 상한) 매수 검증.
  python run_quality.py --source pack --path ../stock-data --out results_quality
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import engine as en  # noqa: E402
import quality as q  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)
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
    ap.add_argument("--out", default="results_quality")
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

    E, meta = q.build(w)
    rows, store = [], {}
    for nm, e in E.items():
        X = q.exits(w, meta[nm]["support"])
        for xn, x in X.items():
            ret, hold, xcol, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost, None, x["close_exit"])
            store[(nm, xn)] = (ret, hold, xcol, valid, e)
            mt = meta[nm]
            row = dict(rule=nm, exit=xn, hl=mt["hl"], v=mt["v"] if mt["v"] is not None else "없음",
                       tr=mt["tr"] if mt["v"] is not None else "없음", k=mt["k"] if mt["k"] is not None else "없음")
            for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
                mm = valid & pm
                s = en.stats(ret[mm], hold[mm], int(pm.sum()))
                row.update({f"{tag}_{kk}": vv for kk, vv in s.items()})
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(out / "quality_grid.csv", index=False, encoding="utf-8-sig")
    print(f"조합 {len(R)}개")

    show = ["rule", "exit", "IS_n", "IS_fill", "IS_wmean", "IS_win", "OOS_n", "OOS_wmean", "OOS_median", "OOS_win", "OOS_t", "OOS_hold"]
    main_name = q.name(*q.MAIN)
    print("\n=== [사전 등록] 메인 규칙: 저가높이기 + 거래량≤급등일 50%·감소중 + 종가≤급등일 종가 ===")
    print(R[R.rule == main_name][show].round(4).to_string(index=False))

    # 조건별 기여: 다른 조건을 같게 두고 해당 조건만 켰을 때의 차이 (모든 조합 평균)
    def contrib(col, on, off):
        key = [c for c in ("hl", "v", "tr", "k", "exit") if c != col]
        on_rows = R[R[col] == on]
        if col == "v":   # 거래량 조건 비교는 '감소중' 없는 행끼리
            on_rows = on_rows[on_rows.tr == False].assign(tr="없음")  # noqa: E712
        a_ = on_rows.set_index(key)[["IS_wmean", "OOS_wmean"]]
        b_ = R[R[col] == off].set_index(key)[["IS_wmean", "OOS_wmean"]]
        j = a_.join(b_, lsuffix="_on", rsuffix="_off", how="inner")
        return dict(조건=f"{col}: {on} vs {off}", 비교쌍=len(j),
                    dIS=(j.IS_wmean_on - j.IS_wmean_off).mean(), dOOS=(j.OOS_wmean_on - j.OOS_wmean_off).mean())
    C = [contrib("hl", True, False)]
    for v in (0.3, 0.5, 0.7):
        C.append(contrib("v", v, "없음"))
    R2 = R[R.v != "없음"]
    j = R2[R2.tr == True].set_index(["hl", "v", "k", "exit"])[["IS_wmean", "OOS_wmean"]].join(  # noqa: E712
        R2[R2.tr == False].set_index(["hl", "v", "k", "exit"])[["IS_wmean", "OOS_wmean"]], lsuffix="_on", rsuffix="_off")  # noqa: E712
    C.append(dict(조건="tr: 감소중 추가 vs 없음", 비교쌍=len(j), dIS=(j.IS_wmean_on - j.IS_wmean_off).mean(),
                  dOOS=(j.OOS_wmean_on - j.OOS_wmean_off).mean()))
    for k in (0.9, 1.0, 1.05):
        C.append(contrib("k", k, "없음"))
    CT = pd.DataFrame(C)
    print("\n=== 조건별 기여 (해당 조건을 켰을 때 거래당 절단평균 변화, 나머지 조건 동일 비교의 평균) ===")
    print(CT.round(4).to_string(index=False))
    print("\n=== 청산별 평균 ===")
    print(R.groupby("exit")[["IS_wmean", "OOS_wmean", "OOS_win"]].mean().round(4).to_string())

    ok = R[(R.IS_n >= 300) & (R.OOS_n >= 150)]
    top = ok.sort_values("IS_wmean", ascending=False).head(10)
    print("\n=== 2015~2021 상위 10 → 2022~ ===")
    print(top[show].round(4).to_string(index=False))
    rho = ok["IS_wmean"].rank().corr(ok["OOS_wmean"].rank())
    print(f"\n조합별 개발-검증 순위상관: {rho:.3f} (조합 {len(ok)}개)")

    # 계좌
    bench = en.limit_entry(w, w.c0 * 0.75, 20, "bench", "bench")
    bret, bhold, bxcol, bvalid = en.simulate(w, bench, 0.15, np.inf, 20, a.cost)
    store[("기존기준", "-")] = (bret, bhold, bxcol, bvalid, bench)
    best = top.iloc[0]
    cands = [("메인(사전 등록)", (main_name, "S1_지지이탈손절·익절15%·10일")),
             ("2015~2021 최고", (best.rule, best.exit)), ("기존 기준(D종가-25% 지정가)", ("기존기준", "-"))]
    prow = []
    for lab, key in cands:
        ret, hold, xcol, valid, e = store[key]
        for gate in (False, True):
            m = valid & (up if gate else True)
            s, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[m], (w.t0 + xcol - P)[m], ret[m],
                                       slots=10, start=SPLIT, max_new_per_day=2)
            prow.append(dict(전략=lab, 시장상승국면만=gate, **s))
    PF = pd.DataFrame(prow)
    PF.to_csv(out / "quality_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 계좌 2022~ (동시 10종목, 하루 신규 2종목) ===")
    print(PF.round(4).to_string(index=False))

    ret, hold, xcol, valid, e = store[cands[0][1]]
    yr = pd.DataFrame({"y": ev.date.dt.year[valid], "r": ret[valid]}).groupby("y")["r"].agg(
        n="size", 평균="mean", 중앙="median", 승률=lambda s: (s > 0).mean())
    print("\n=== 메인 규칙 연도별 ===")
    print(yr.round(4).to_string())

    M = R[(R.rule == main_name) & (R.exit == "S1_지지이탈손절·익절15%·10일")].iloc[0]
    pfb = PF.set_index(["전략", "시장상승국면만"])
    checks = [
        ("메인 규칙 2015~2021 거래당 > 0", M.IS_wmean > 0),
        ("메인 규칙 2022~ 거래당 > 0, t >= 2", M.OOS_wmean > 0 and M.OOS_t >= 2),
        ("세 조건(저가높이기·거래량감소·가격상한) 각각의 기여가 두 기간 모두 플러스",
         all(CT.iloc[i][["dIS", "dOOS"]].gt(0).all() for i in (0, 2, 6))),
        ("계좌(시장상승국면만): 메인 규칙 연수익 > 기존 기준",
         pfb.loc[("메인(사전 등록)", True), "cagr"] > pfb.loc[("기존 기준(D종가-25% 지정가)", True), "cagr"]),
    ]
    print("\n=== 사전 판정 기준 ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(bool(v) for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
