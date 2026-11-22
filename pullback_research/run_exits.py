"""
종목별 맞춤 청산(익절·손절) 연구.

진입은 고정하고 청산만 바꿔 비교한다.
  진입 1 (메인 후보): 4조건 선별 + 급등일 종가 -15% 지정가 (D+2부터 10일)
  진입 2 (일반성 점검): 선별 없이 같은 지정가
  진입 3 (일반성 점검): 선별 없이 급등일 종가 -25% 지정가 (D+2부터 20일)

청산 계열
  F  고정      : 익절 {10,15,20,30%,없음} × 손절 {없음,10,15%} × 보유 {5,10,20일}
  V  변동성비례: ATR = 급등 전 20일 평균 일중 범위(가격 대비 %)
                 익절 = 진입가 × (1 + a·ATR), a ∈ {2,3,4,6}; 손절 = 진입가 × (1 - b·ATR), b ∈ {없음,2,3}; 보유 {10,20}
  S  구조      : 익절가 = 급등일 종가 ×{0.95,1.0}, 급등일 고가, 진입가와 급등일 종가의 중간; 손절 없음; 보유 {10,20}
  T  추적      : 진입 후 최고 종가 × (1 - m·ATR) 아래로 종가 마감 -> 다음날 시가, m ∈ {2,3,4}; 익절 없음/30%; 보유 20
각 계열에서 2015~2021 최고를 고른 뒤 2022~ 에 그대로 적용.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import screen as sc  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 48)
INF = np.inf


def exit_specs(w, e, atr):
    """(이름, 계열, dict(tp, sl, H, tp_px, sl_px, close_exit)) 목록."""
    n, K = len(w.c0), w.K
    p = e.price
    out = []
    for tp, sl, H in product((0.10, 0.15, 0.20, 0.30, INF), (INF, 0.10, 0.15), (5, 10, 20)):
        out.append((f"F 익절{tp if tp < INF else '없음'}·손절{sl if sl < INF else '없음'}·{H}일", "F고정",
                    dict(tp=tp, sl=sl, H=H)))
    for a, b, H in product((2, 3, 4, 6), (None, 2, 3), (10, 20)):
        tp_px = p * (1 + a * atr)
        sl_px = p * (1 - b * atr) if b else None
        out.append((f"V 익절{a}ATR·손절{b or '없음'}{'ATR' if b else ''}·{H}일", "V변동성비례",
                    dict(tp=INF, sl=INF, H=H, tp_px=tp_px, sl_px=sl_px)))
    for nm, px in (("급등일종가×0.95", w.c0 * 0.95), ("급등일종가", w.c0), ("급등일고가", w.h0),
                   ("진입가~급등일종가 중간", (p + w.c0) / 2)):
        for H in (10, 20):
            out.append((f"S 익절={nm}·{H}일", "S구조", dict(tp=0.15, sl=INF, H=H, tp_px=px)))
    cols = np.arange(K)
    has = e.col >= 0
    ecol = np.where(has, e.col, K)
    runmax = np.maximum.accumulate(np.where(cols[None, :] >= ecol[:, None], np.nan_to_num(w.C, nan=-np.inf), -np.inf), axis=1)
    for m, tp in product((2, 3, 4), (INF, 0.30)):
        level = runmax * (1 - m * atr[:, None])
        out.append((f"T 추적 최고종가-{m}ATR·익절{tp if tp < INF else '없음'}·20일", "T추적",
                    dict(tp=tp, sl=INF, H=20, close_exit=level)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pack", choices=["pack", "pickle", "csv"])
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_exits")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = {"pack": lambda: data.load_pack(a.path), "pickle": lambda: data.load_panel(a.path),
             "csv": lambda: data.load_csv_dir(a.path, a.meta)}[a.source]()
    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    X = sc.features(panel, w, ev)
    P = en.PRE
    yr = ev["date"].dt.year.values
    is_m = yr < 2022
    atr = np.clip(X["atr_pct"].fillna(X["atr_pct"].median()).values, 0.01, 0.30)
    screen = ((X["ret20_pre"] <= 0) & (X["pos52"] <= -0.30) & (X["prior20"] == 0)
              & (X["v1_ratio"] <= np.log10(0.5))).values
    print(f"급등 {len(ev):,}건, 4조건 선별 {screen.sum():,}건. 평소 변동성(ATR) 중앙값 {np.median(atr):.3f}, "
          f"10~90% 범위 {np.quantile(atr, 0.1):.3f}~{np.quantile(atr, 0.9):.3f}")
    setups = {
        "선별+급등일종가-15%지정가": (en.limit_entry(w, w.c0 * 0.85, 11, "s", "s", start=2), screen),
        "전체+급등일종가-15%지정가": (en.limit_entry(w, w.c0 * 0.85, 11, "a", "s", start=2), np.ones(len(ev), bool)),
        "전체+급등일종가-25%지정가": (en.limit_entry(w, w.c0 * 0.75, 21, "b", "s", start=2), np.ones(len(ev), bool)),
    }
    dates = pd.Series(ev["date"].values)
    allrows, best_rows, prow = [], [], []
    for sname, (e, gm) in setups.items():
        rows = []
        store = {}
        for xn, fam, x in exit_specs(w, e, atr):
            ret, hold, xcol, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost, x.get("sl_px"),
                                                 x.get("close_exit"), x.get("tp_px"))
            store[xn] = (ret, hold, xcol, valid)
            row = dict(진입=sname, 청산=xn, 계열=fam)
            for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
                mm = valid & gm & pm
                s = en.stats(ret[mm], hold[mm], int((gm & pm).sum()))
                row.update({f"{tag}_{k}": s[k] for k in ("n", "wmean", "median", "win", "hold")})
            rows.append(row)
        R = pd.DataFrame(rows)
        allrows.append(R)
        fam_med = R.groupby("계열")[["IS_wmean", "OOS_wmean"]].median()
        pick = R.loc[R.groupby("계열")["IS_wmean"].idxmax()]
        print(f"\n################ {sname}")
        print("계열별 (모든 설정의 중앙값):")
        print(fam_med.round(4).to_string())
        print("계열별 2015~2021 최고 설정 → 2022~:")
        print(pick[["계열", "청산", "IS_n", "IS_wmean", "IS_win", "OOS_n", "OOS_wmean", "OOS_median", "OOS_win", "OOS_hold"]]
              .round(4).to_string(index=False))
        rho = R["IS_wmean"].rank().corr(R["OOS_wmean"].rank())
        print(f"설정별 개발-검증 순위상관 {rho:.3f}")
        for r in pick.itertuples():
            ret, hold, xcol, valid = store[r.청산]
            m = valid & gm
            mo = m & ~is_m
            mc, se = capit.clustered_mean_se(pd.Series(ret[mo]), dates[mo])
            res = {}
            for lab, sd, ed in (("15_21", "2015-01-01", "2022-01-01"), ("22_", "2022-01-01", None)):
                s_, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[m], (w.t0 + xcol - P)[m], ret[m],
                                            slots=10, start=sd, end=ed, max_new_per_day=2)
                res[f"cagr_{lab}"], res[f"mdd_{lab}"] = s_["cagr"], s_["mdd"]
            best_rows.append(dict(진입=sname, 계열=r.계열, 청산=r.청산, IS_wmean=r.IS_wmean, OOS_wmean=r.OOS_wmean,
                                  OOS_t_묶음=mc / se if se else np.nan, **res))
    pd.concat(allrows).to_csv(out / "exits_grid.csv", index=False, encoding="utf-8-sig")
    B = pd.DataFrame(best_rows)
    B.to_csv(out / "exits_best.csv", index=False, encoding="utf-8-sig")
    print("\n=== 계열별 최고 설정의 계좌 성적 (동시 10종목, 하루 2종목) ===")
    print(B.round(4).to_string(index=False))

    # 판정: 메인 후보에서 맞춤 청산(V/S/T) 최고가 고정(F) 최고보다 나은가
    M = B[B["진입"] == "선별+급등일종가-15%지정가"].set_index("계열")
    f = M.loc["F고정"]
    adapt = M.drop(index="F고정")
    better = adapt[(adapt.OOS_wmean - f.OOS_wmean >= 0.003) & (adapt["cagr_22_"] > f["cagr_22_"])
                   & (adapt["mdd_22_"] >= f["mdd_22_"] - 0.02) & (adapt["cagr_15_21"] >= f["cagr_15_21"])]
    allR = pd.concat(allrows)
    fm = allR.groupby(["진입", "계열"])[["IS_wmean", "OOS_wmean"]].median().unstack("계열")
    consistent = []
    for fam in ("V변동성비례", "S구조", "T추적"):
        d_is = (fm[("IS_wmean", fam)] - fm[("IS_wmean", "F고정")])
        d_os = (fm[("OOS_wmean", fam)] - fm[("OOS_wmean", "F고정")])
        consistent.append((fam, bool(((d_is > 0) & (d_os > 0)).all())))
    print("\n=== 판정 ===")
    print(f"  메인 후보에서 고정 청산보다 나은 맞춤 청산 계열: {list(better.index) or '없음'} "
          f"(기준: 2022~ 거래당 +0.3%p 이상, 계좌 2022~ 연수익↑, 낙폭 악화 2%p 이내, 2015~21 연수익도 같거나 높음)")
    for fam, ok in consistent:
        print(f"  [{'통과' if ok else '실패'}] {fam}: 세 진입 모두, 두 기간 모두 설정 중앙값이 고정 청산보다 높음")


if __name__ == "__main__":
    main()
