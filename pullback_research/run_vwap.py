"""
최종 전략 안에서 앵커드 VWAP 매매 검증 (두 시기).
VWAP(일봉 근사): 대표가격 = (고가+저가+종가)/3, 앵커 = 급등일(D) 또는 다음날(D+1)부터 누적
  AVWAP_d = Σ(대표가격×거래량) / Σ거래량  (앵커일 ~ d)
  밴드: 거래량가중 표준편차 σ_d, 하단 = AVWAP - σ
고정: 모델 상위30%, 시장상승국면, D+2부터 10거래일, 10종목·하루2, 비용 0.3%
기본 청산: 익절 7% + 진입일 저가 종가이탈 손절 + 10일 (현재 최선과 동일)

1) 매수 가격: 전일 AVWAP(D앵커) ×{1.00, 0.95, 0.90}, 전일 밴드 하단, 전일 AVWAP(D+1 앵커 = 추격자 평균단가)
2) 매도: 익절 = 진입일 AVWAP(D앵커) (진입가보다 1% 이상 위일 때만, 아니면 7%), 같은 손절·10일
   -> 예상저점 진입과 VWAP 진입 모두에 적용
3) 필터: 예상저점 진입 + {다음날 종가 > AVWAP, 다음날 종가 < AVWAP}
4) VWAP 재돌파: D+2~D+10 중 전일 종가 < 전일 AVWAP 이고 오늘 종가 > 오늘 AVWAP 인 첫날 종가 매수
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

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
INF = np.inf
P = en.PRE


def avwap(w, anchor):
    tp = (w.H + w.L + w.C) / 3
    cols = np.arange(w.K)
    m = (cols[None, :] >= anchor) & np.isfinite(tp)
    v = np.where(m, w.V, 0.0)
    x = np.where(m, tp, 0.0)
    cv = np.cumsum(v, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = np.cumsum(v * x, axis=1) / cv
        var = np.cumsum(v * x * x, axis=1) / cv - vw ** 2
    vw[cv <= 0] = np.nan
    sd = np.sqrt(np.clip(var, 0, None))
    sd[cv <= 0] = np.nan
    return vw, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_vwap")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    A = ri.prepare(data.load_pack(a.path, "krx_"))
    B = ri.prepare(data.load_pack(a.path, ""))
    preds = ri.predict_past_only(A, B)
    lq = r2.predict_low(A, B, 0.6)
    rows = []
    for era, D in (("A", A), ("B", B)):
        pr, thr = preds[era]
        w, ev = D["w"], D["ev"]
        sd_, ed_ = r2.ERAS[era]
        inper = (ev["date"] >= sd_).values & ((ev["date"] < ed_).values if ed_ else True)
        sel = np.isfinite(pr) & (pr >= thr[0.7])
        gate = D["up20"][np.clip(ev["t"].values + 1, 0, len(D["up20"]) - 1)]
        c0, K = w.c0, w.K
        r_ = np.arange(len(c0))
        vwD, sdD = avwap(w, P)
        vw1, _ = avwap(w, P + 1)
        pvD, psD, pv1 = _prev(vwD), _prev(sdD), _prev(vw1)
        ql = c0 * np.clip(1 + lq[era], 0.70, 0.97)
        C1, V1D = w.C[:, P + 1], vwD[:, P + 1]
        cap = lambda lv: np.minimum(lv, (c0 * 0.97)[:, None] if np.ndim(lv) == 2 else c0 * 0.97)
        entries = {
            "기준: 예상저점(분위0.6) [현재 최선]": (en.limit_entry(w, ql, 11, "q", "x", start=2), None),
            "기준: 급등일종가×0.85 [기존]": (en.limit_entry(w, c0 * 0.85, 11, "r", "x", start=2), None),
            "1 전일 VWAP 지정가": (en.limit_entry(w, cap(pvD), 11, "v", "x", start=2), None),
            "1 전일 VWAP×0.95": (en.limit_entry(w, cap(pvD * 0.95), 11, "v", "x", start=2), None),
            "1 전일 VWAP×0.90": (en.limit_entry(w, cap(pvD * 0.90), 11, "v", "x", start=2), None),
            "1 전일 VWAP 밴드하단(-1σ)": (en.limit_entry(w, cap(pvD - psD), 11, "v", "x", start=2), None),
            "1 전일 추격자 VWAP(D+1 앵커)": (en.limit_entry(w, cap(pv1), 11, "v", "x", start=2), None),
            "3 예상저점 + 다음날 종가>VWAP": (en.limit_entry(w, ql, 11, "q", "x", start=2), C1 > V1D),
            "3 예상저점 + 다음날 종가<VWAP": (en.limit_entry(w, ql, 11, "q", "x", start=2), C1 < V1D),
        }
        cols = np.arange(K)
        win = (cols[None, :] >= P + 2) & (cols[None, :] <= P + 10) & w.known
        sig = win & (_prev(w.C) < pvD) & (w.C > vwD)
        entries["4 VWAP 재돌파 종가매수"] = (en.signal_entry(w, sig, 10, "rc", "x", "close", start=2), None)
        for nm, (e, filt) in entries.items():
            el = np.where(e.col >= 0, w.L[r_, np.clip(e.col, 0, K - 1)], np.nan)
            ce = np.broadcast_to(el[:, None], (len(el), K))
            ev_vw = np.where(e.col >= 0, vwD[r_, np.clip(e.col, 0, K - 1)], np.nan)
            exits = {"익절7%": None, "익절=진입일 VWAP": ev_vw}
            for xn, tpx in exits.items():
                if xn != "익절7%" and not (nm.startswith("기준: 예상") or nm.startswith("1 ")):
                    continue
                ret, hold, xcol, valid = en.simulate(w, e, 0.07, INF, 10, a.cost, None, ce, tpx)
                s_ = sel & (filt if filt is not None else True)
                m = valid & s_ & inper
                mg = m & gate
                mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(ev["date"].values[m])) if m.sum() > 2 else (np.nan, np.nan)
                pf, cs = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mg], (w.t0 + xcol - P)[mg], ret[mg],
                                             slots=10, start=sd_, end=ed_, max_new_per_day=2)
                yrs = cs.resample("YE").last()
                ar = yrs / yrs.shift(1).fillna(1.0) - 1 if len(yrs) else pd.Series(dtype=float)
                tpw = np.nanmedian((tpx / e.price - 1)[m]) if (tpx is not None and m.any()) else np.nan
                rows.append(dict(시기=era, 방식=nm, 청산=xn, 체결률=m.sum() / max((sel & inper).sum(), 1),
                                 급등일종가대비=float(np.nanmedian((e.price / c0 - 1)[m])) if m.any() else np.nan,
                                 익절폭중앙=tpw, n=int(m.sum()), 거래당=en.wmean(ret[m]) if m.any() else np.nan,
                                 승률=float((ret[m] > 0).mean()) if m.any() else np.nan, t묶음=mc / se if se else np.nan,
                                 계좌연=pf["cagr"], 낙폭=pf["mdd"], 플러스연도=f"{int((ar > 0).sum())}/{len(ar)}"))
        # 참고: VWAP 과 급등일 종가의 거리
        print(f"[{era}] 다음날 기준 VWAP / 급등일 종가 중앙값 {np.nanmedian(V1D / c0):.3f}, "
              f"다음날 종가가 VWAP 위인 비율 {np.nanmean(C1 > V1D):.2f}")
    R = pd.DataFrame(rows)
    R["calmar"] = R["계좌연"] / R["낙폭"].abs()
    R.to_csv(out / "vwap.csv", index=False, encoding="utf-8-sig")
    print(R.round(4).to_string(index=False))
    cur = R[R["방식"].str.contains("현재 최선") & (R["청산"] == "익절7%")].set_index("시기")
    print("\n=== 판정: 두 시기 모두 현재 최선보다 (연수익/낙폭) 높고 연수익 하락 1%p 이내 ===")
    for (nm, xn), g in R.groupby(["방식", "청산"]):
        if "현재 최선" in nm and xn == "익절7%":
            continue
        g = g.set_index("시기")
        if not {"A", "B"} <= set(g.index):
            continue
        ok = all(g.loc[s, "calmar"] > cur.loc[s, "calmar"] and g.loc[s, "계좌연"] >= cur.loc[s, "계좌연"] - 0.01 for s in ("A", "B"))
        print(f"  [{'개선' if ok else '아님'}] {nm} / {xn}")


if __name__ == "__main__":
    main()
