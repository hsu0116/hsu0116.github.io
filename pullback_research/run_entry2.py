"""
최종 전략의 '매수 가격'만 바꿔 비교 (나머지 고정).
고정: 모델 상위30% 선별, 시장상승국면, 지정가 D+2부터 10거래일, 익절 7%, 종가가 진입일 저가 이탈 시 다음날 시가 손절,
      최대 10일, 동시 10종목·하루 2종목, 비용 0.3%.

매수 가격 후보
  R  고정 비율  : 급등일 종가 × {0.80, 0.83, 0.85(현재), 0.88, 0.90, 0.92}
  V  변동성 비례: 급등일 종가 × (1 - k × 평소변동성), k ∈ {1.5, 2, 2.5, 3}
  F  장대봉 되돌림: 급등 전 가격 + f × (급등일 종가 - 급등 전 가격), f ∈ {0, 0.25, 0.5}
  N  다음날 종가 기준: 다음날 종가 × {0.95, 0.92, 0.90}
  M  모델 점수 차등: 점수 상위10% 는 × 0.90, 나머지 × 0.85  /  상위10% × 0.88, 나머지 × 0.83
  Q  예상 저점 예측: 앞으로 10일(D+2~D+11) 최저가/급등일 종가 의 분위수(q=0.4/0.5/0.6)를 모델로 예측해 그 가격에 지정가
     (과거만 학습: A 는 A 의 지난해까지, B 는 A 전체 + B 의 지난해까지. 지정가는 급등일 종가의 70~97% 로 제한)
판정: 두 시기 모두 '연수익/최대낙폭'이 현재(0.85)보다 높고, 연수익이 1%p 넘게 낮아지지 않을 것.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import run_integrated as ri  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
INF = np.inf
P = en.PRE
ERAS = {"A": ("2008-01-01", "2015-01-01"), "B": ("2015-01-01", None)}


def mae_target(w):
    """D+2 ~ D+11 최저가 / 급등일 종가 - 1"""
    with np.errstate(all="ignore"):
        lo = np.nanmin(np.where(np.isnan(w.L[:, P + 2:P + 12]), np.inf, w.L[:, P + 2:P + 12]), axis=1)
    lo[~np.isfinite(lo)] = np.nan
    return lo / w.c0 - 1


def predict_low(A, B, q):
    res = {}
    for name, D, years, extra in (("A", A, range(2008, 2015), None), ("B", B, range(2015, B["yr"].max() + 1), A)):
        y = mae_target(D["w"])
        pr = np.full(len(y), np.nan)
        ok = np.isfinite(y)
        for yy in years:
            Xt, Yt = [D["X"].values[ok & (D["yr"] < yy)]], [y[ok & (D["yr"] < yy)]]
            if extra is not None:
                ye = mae_target(extra["w"])
                oe = np.isfinite(ye) & (extra["yr"] >= 2005) & (extra["yr"] < 2015)
                Xt.append(extra["X"].values[oe])
                Yt.append(ye[oe])
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, learning_rate=0.05,
                                              max_iter=200, min_samples_leaf=200, random_state=0)
            m.fit(np.vstack(Xt), np.clip(np.concatenate(Yt), -0.6, 0.3))
            te = D["yr"] == yy
            pr[te] = m.predict(D["X"].values[te])
        res[name] = pr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_entry2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    A = ri.prepare(data.load_pack(a.path, "krx_"))
    B = ri.prepare(data.load_pack(a.path, ""))
    preds = ri.predict_past_only(A, B)
    print("예상 저점 분위수 모델 학습 중...")
    lowq = {q: predict_low(A, B, q) for q in (0.4, 0.5, 0.6)}
    rows = []
    for era, D in (("A", A), ("B", B)):
        pr, thr = preds[era]
        w, ev = D["w"], D["ev"]
        sd, ed = ERAS[era]
        inper = (ev["date"] >= sd).values & ((ev["date"] < ed).values if ed else True)
        sel = np.isfinite(pr) & (pr >= thr[0.7])
        top10 = np.isfinite(pr) & (pr >= thr[0.9])
        gate = D["up20"][np.clip(ev["t"].values + 1, 0, len(D["up20"]) - 1)]
        atr = np.clip(D["X"]["atr_pct"].fillna(D["X"]["atr_pct"].median()).values, 0.01, 0.30)
        c0, p0, C1 = w.c0, w.p0, w.C[:, P + 1]
        levels = {}
        for k in (0.80, 0.83, 0.85, 0.88, 0.90, 0.92):
            levels[f"R 급등일종가×{k:.2f}" + (" (현재)" if k == 0.85 else "")] = ("R", c0 * k)
        for k in (1.5, 2.0, 2.5, 3.0):
            levels[f"V 급등일종가×(1-{k}×변동성)"] = ("V", c0 * np.clip(1 - k * atr, 0.6, 0.97))
        for f in (0.0, 0.25, 0.5):
            levels[f"F 장대봉 {int(f*100)}% 지점"] = ("F", p0 + f * (c0 - p0))
        for k in (0.95, 0.92, 0.90):
            levels[f"N 다음날종가×{k:.2f}"] = ("N", C1 * k)
        levels["M 상위10%×0.90 / 나머지×0.85"] = ("M", c0 * np.where(top10, 0.90, 0.85))
        levels["M 상위10%×0.88 / 나머지×0.83"] = ("M", c0 * np.where(top10, 0.88, 0.83))
        for q, lp in lowq.items():
            levels[f"Q 예상저점(분위 {q})"] = ("Q", c0 * np.clip(1 + lp[era], 0.70, 0.97))
        r_ = np.arange(len(c0))
        for nm, (fam, lv) in levels.items():
            e = en.limit_entry(w, lv, 11, "x", "x", start=2)
            el = np.where(e.col >= 0, w.L[r_, np.clip(e.col, 0, w.K - 1)], np.nan)
            ce = np.broadcast_to(el[:, None], (len(el), w.K))
            ret, hold, xcol, valid = en.simulate(w, e, 0.07, INF, 10, a.cost, None, ce)
            m = valid & sel & inper
            mg = m & gate
            mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(ev["date"].values[m]))
            pf, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mg], (w.t0 + xcol - P)[mg], ret[mg],
                                        slots=10, start=sd, end=ed, max_new_per_day=2)
            disc = np.nanmedian((e.price / c0 - 1)[m])
            rows.append(dict(시기=era, 계열=fam, 매수가격=nm, 체결률=m.sum() / max((sel & inper).sum(), 1),
                             급등일종가대비=disc, n=int(m.sum()), 거래당=en.wmean(ret[m]), 승률=float((ret[m] > 0).mean()),
                             t묶음=mc / se if se else np.nan, 계좌연=pf["cagr"], 낙폭=pf["mdd"]))
    R = pd.DataFrame(rows)
    R["calmar"] = R["계좌연"] / R["낙폭"].abs()
    R.to_csv(out / "entry2.csv", index=False, encoding="utf-8-sig")
    Wd = R.pivot_table(index=["계열", "매수가격"], columns="시기",
                       values=["체결률", "급등일종가대비", "거래당", "승률", "계좌연", "낙폭", "calmar"], aggfunc="first")
    Wd.columns = [f"{v}_{s}" for v, s in Wd.columns]
    Wd = Wd.reset_index()
    cols = ["계열", "매수가격", "체결률_A", "급등일종가대비_A", "거래당_A", "계좌연_A", "낙폭_A", "calmar_A",
            "체결률_B", "급등일종가대비_B", "거래당_B", "계좌연_B", "낙폭_B", "calmar_B"]
    print(Wd.sort_values(["계열", "매수가격"])[cols].round(4).to_string(index=False))
    cur = Wd[Wd["매수가격"].str.contains("현재")].iloc[0]
    print("\n=== 판정: 두 시기 모두 (연수익/낙폭) > 현재 이고, 연수익 하락 1%p 이내 ===")
    for r in Wd.itertuples():
        if "현재" in r.매수가격:
            continue
        ok = all(getattr(r, f"calmar_{s}") > cur[f"calmar_{s}"] and getattr(r, f"계좌연_{s}") >= cur[f"계좌연_{s}"] - 0.01
                 for s in ("A", "B"))
        if ok:
            print(f"  [개선] {r.매수가격}: A 연 {r.계좌연_A:+.3f}/{r.낙폭_A:.3f}, B 연 {r.계좌연_B:+.3f}/{r.낙폭_B:.3f}")
    print(f"  (현재 0.85: A 연 {cur['계좌연_A']:+.3f}/{cur['낙폭_A']:.3f}, B 연 {cur['계좌연_B']:+.3f}/{cur['낙폭_B']:.3f})")


if __name__ == "__main__":
    main()
