"""
최종 전략 안에서 이동평균선 활용 매수 검증 (두 시기).
고정: 모델 상위30%, 시장상승국면, 지정가 D+2부터 10거래일, 익절 7%, 진입일 저가 종가이탈 손절, 10일, 10종목·하루2.

1) 이평선을 매수 가격으로: 전일 기준 5·10·20·60일선 (매일 갱신되는 지정가), 20일선×0.95 (엔벨로프)
2) 이평선을 필터로: 예상저점(분위0.6) 지정가 + {급등일 종가 > 60일선, 급등일 20일선 상승(5일 전 대비), 급등 전 종가 < 20일선(바닥권)}
3) 혼합: 지정가 = max(예상저점, 전일 20일선) / min(예상저점, 전일 20일선)
비교: 예상저점(분위0.6) [현재 최선], 급등일종가×0.85 [기존]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import run_entry2 as r2  # noqa: E402
import run_integrated as ri  # noqa: E402
from psych import _prev  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
INF = np.inf
P = en.PRE


def ma_windows(panel, ev, K):
    C = panel["close"].astype("float32")
    out = {}
    for n in (5, 10, 20, 60):
        ma = C.rolling(n, min_periods=n).mean().values
        T = ma.shape[0]
        offs = np.arange(-P, K - P)
        tt = ev["t"].values[:, None] + offs[None, :]
        ok = (tt >= 0) & (tt < T)
        a = ma[np.clip(tt, 0, T - 1), ev["i"].values[:, None]].astype(float)
        a[~ok] = np.nan
        out[n] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_ma")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pa, pb = data.load_pack(a.path, "krx_"), data.load_pack(a.path, "")
    A, B = ri.prepare(pa), ri.prepare(pb)
    preds = ri.predict_past_only(A, B)
    lq = r2.predict_low(A, B, 0.6)
    rows = []
    for era, D, pan in (("A", A, pa), ("B", B, pb)):
        pr, thr = preds[era]
        w, ev = D["w"], D["ev"]
        sd, ed = r2.ERAS[era]
        inper = (ev["date"] >= sd).values & ((ev["date"] < ed).values if ed else True)
        sel = np.isfinite(pr) & (pr >= thr[0.7])
        gate = D["up20"][np.clip(ev["t"].values + 1, 0, len(D["up20"]) - 1)]
        MA = ma_windows(pan, ev, w.K)
        c0 = w.c0
        ql = c0 * np.clip(1 + lq[era], 0.70, 0.97)
        qlb = np.broadcast_to(ql[:, None], (len(ql), w.K))
        ma20p = _prev(MA[20])
        ma60_D, ma20_D = MA[60][:, P], MA[20][:, P]
        ma20_D5 = MA[20][:, P - 5]
        cpre = w.C[:, P - 1]
        cand = {
            "기준: 예상저점(분위0.6) [현재 최선]": (ql, None),
            "기준: 급등일종가×0.85 [기존]": (c0 * 0.85, None),
            "1 전일 5일선 지정가": (_prev(MA[5]), None),
            "1 전일 10일선 지정가": (_prev(MA[10]), None),
            "1 전일 20일선 지정가": (ma20p, None),
            "1 전일 60일선 지정가": (_prev(MA[60]), None),
            "1 엔벨로프 20일선×0.95": (ma20p * 0.95, None),
            "2 예상저점 + 급등일 종가>60일선": (ql, c0 > ma60_D),
            "2 예상저점 + 20일선 상승중": (ql, ma20_D > ma20_D5),
            "2 예상저점 + 급등 전 종가<20일선(바닥권)": (ql, cpre < MA[20][:, P - 1]),
            "3 max(예상저점, 전일20일선)": (np.fmax(qlb, ma20p), None),
            "3 min(예상저점, 전일20일선)": (np.fmin(qlb, ma20p), None),
        }
        r_ = np.arange(len(c0))
        for nm, (lv, filt) in cand.items():
            lv = np.where(np.isfinite(lv), lv, np.nan)
            if lv.ndim == 2:   # 너무 높은(급등일 종가 초과) 지정가는 막음
                lv = np.minimum(lv, c0[:, None] * 0.97)
            else:
                lv = np.minimum(lv, c0 * 0.97)
            e = en.limit_entry(w, lv, 11, "x", "x", start=2)
            el = np.where(e.col >= 0, w.L[r_, np.clip(e.col, 0, w.K - 1)], np.nan)
            ce = np.broadcast_to(el[:, None], (len(el), w.K))
            ret, hold, xcol, valid = en.simulate(w, e, 0.07, INF, 10, a.cost, None, ce)
            s_ = sel & (filt if filt is not None else True)
            m = valid & s_ & inper
            mg = m & gate
            mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(ev["date"].values[m])) if m.sum() > 2 else (np.nan, np.nan)
            pf, cs = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mg], (w.t0 + xcol - P)[mg], ret[mg],
                                         slots=10, start=sd, end=ed, max_new_per_day=2)
            yrs = cs.resample("YE").last()
            ar = yrs / yrs.shift(1).fillna(1.0) - 1 if len(yrs) else pd.Series(dtype=float)
            rows.append(dict(시기=era, 방식=nm, 체결률=m.sum() / max((sel & inper).sum(), 1),
                             급등일종가대비=float(np.nanmedian((e.price / c0 - 1)[m])) if m.any() else np.nan,
                             n=int(m.sum()), 거래당=en.wmean(ret[m]) if m.any() else np.nan,
                             승률=float((ret[m] > 0).mean()) if m.any() else np.nan, t묶음=mc / se if se else np.nan,
                             계좌연=pf["cagr"], 낙폭=pf["mdd"], 플러스연도=f"{int((ar > 0).sum())}/{len(ar)}"))
    R = pd.DataFrame(rows)
    R["calmar"] = R["계좌연"] / R["낙폭"].abs()
    R.to_csv(out / "ma.csv", index=False, encoding="utf-8-sig")
    print(R.round(4).to_string(index=False))
    cur = R[R["방식"].str.contains("현재 최선")].set_index("시기")
    print("\n=== 판정: 두 시기 모두 현재 최선보다 (연수익/낙폭) 높고 연수익 하락 1%p 이내 ===")
    for nm, g in R.groupby("방식"):
        if "현재 최선" in nm:
            continue
        g = g.set_index("시기")
        ok = all(g.loc[s, "calmar"] > cur.loc[s, "calmar"] and g.loc[s, "계좌연"] >= cur.loc[s, "계좌연"] - 0.01 for s in ("A", "B"))
        print(f"  [{'개선' if ok else '아님'}] {nm}")


if __name__ == "__main__":
    main()
