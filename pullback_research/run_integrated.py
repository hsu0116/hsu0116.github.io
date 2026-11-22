"""
통합 전략: 종목 선별(반등 모델) -> 눌림의 질 진입 -> 종목별 청산 -> 시장국면·분산 운용.
두 시기(A: KRX 2005~2014 상장폐지 포함 / B: 키움 2015~2026)에서 모두 검증.

모델은 그 시점까지의 과거로만 학습:
  A: 2008~2014 각 연도를 A 의 그 전 해까지로 학습해 예측
  B: 2015~2026 각 연도를 A 전체 + B 의 그 전 해까지로 학습해 예측
선별 임계값 = 그 해 모델의 학습 데이터 예측값 상위 20% / 30% 경계

선택 기준(실행 전 고정): 두 시기 계좌의 (연복리 / |최대낙폭|) 중 작은 값이 가장 큰 조합. 두 시기 모두 연복리 > 0.
  python run_integrated.py --path ../stock-data --out results_integrated
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import screen as sc  # noqa: E402
from psych import _prev  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)
INF = np.inf
P = en.PRE


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def prepare(panel):
    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    X = sc.features(panel, w, ev)
    Y = sc.targets(w)
    idx = sc.market_index(panel)
    up20 = (idx > idx.rolling(20).mean()).values
    return dict(ev=ev, w=w, X=X, Y=Y, up20=up20, yr=ev["date"].dt.year.values)


def predict_past_only(A, B):
    """반환: A 예측(2008~), B 예측(2015~), 그리고 각 행의 상위20/30/10% 임계값."""
    out = {}
    for name, D, years, extra in (("A", A, range(2008, 2015), None), ("B", B, range(2015, B["yr"].max() + 1), A)):
        n = len(D["ev"])
        pr = np.full(n, np.nan)
        thr = {q: np.full(n, np.nan) for q in (0.7, 0.8, 0.9)}
        okD = D["Y"].y_bar.notna().values
        for y in years:
            trX = [D["X"].values[okD & (D["yr"] < y)]]
            trY = [D["Y"].y_bar.values[okD & (D["yr"] < y)]]
            if extra is not None:
                okE = extra["Y"].y_bar.notna().values & (extra["yr"] >= 2005) & (extra["yr"] < 2015)
                trX.append(extra["X"].values[okE])
                trY.append(extra["Y"].y_bar.values[okE])
            Xt, Yt = np.vstack(trX), np.concatenate(trY).astype(int)
            m = gbm().fit(Xt, Yt)
            te = D["yr"] == y
            pr[te] = m.predict_proba(D["X"].values[te])[:, 1]
            ptr = m.predict_proba(Xt)[:, 1]
            for q in thr:
                thr[q][te] = np.quantile(ptr, q)
        out[name] = (pr, thr)
    return out


def entries(w):
    C, L, V, O = w.C, w.L, w.V, w.O
    n, K = len(w.c0), w.K
    r = np.arange(n)
    cols = np.arange(K)
    lk = en.locked_up(w, r, np.full(n, P + 1))
    C1, O2 = C[:, P + 1], O[:, P + 2]
    E = {
        "E0 다음날종가": en.Entry("E0", np.where(lk | ~np.isfinite(C1), -1, P + 1), C1, "close"),
        "E1 다음다음날시가": en.Entry("E1", np.where(np.isfinite(O2), P + 2, -1), O2, "open"),
        "E4 급등일종가-15%지정가": en.limit_entry(w, w.c0 * 0.85, 11, "E4", "i", start=2),
    }
    win = (cols[None, :] >= P + 2) & (cols[None, :] <= P + 10) & w.known
    base = win & (L > _prev(L)) & (V <= 0.5 * w.v0[:, None])
    for nm, k in (("E2 눌림의질(종가≤급등일종가)", 1.0), ("E3 눌림의질(종가≤급등일종가×0.9)", 0.9)):
        m = base & (C <= k * w.c0[:, None])
        j = np.where(m.any(1), np.argmax(m, 1), -1)
        ok = j >= 0
        jj = np.where(ok, j, 0)
        px = np.where(ok, C[r, jj], np.nan)
        ok &= np.isfinite(px) & ~en.locked_up(w, r, jj)
        E[nm] = en.Entry(nm, np.where(ok, jj, -1), np.where(ok, px, np.nan), "close")
    return E


def exits(w, e, atr, hi_conf):
    p = e.price
    X = {
        "F10 익절10%·10일": dict(tp=0.10, sl=INF, H=10),
        "F15 익절15%·10일": dict(tp=0.15, sl=INF, H=10),
        "V2 익절2×ATR·10일": dict(tp=INF, sl=INF, H=10, tp_px=p * (1 + 2 * atr)),
        "V3 익절3×ATR·10일": dict(tp=INF, sl=INF, H=10, tp_px=p * (1 + 3 * atr)),
        "V2s 익절2×ATR·손절2×ATR·10일": dict(tp=INF, sl=INF, H=10, tp_px=p * (1 + 2 * atr), sl_px=p * (1 - 2 * atr)),
        "S 익절=급등일종가·10일": dict(tp=0.10, sl=INF, H=10, tp_px=w.c0),
    }
    return X


def run_exit(w, e, x, cost, hi_conf=None):
    if x == "P":
        r1 = en.simulate(w, e, 0.10, INF, 10, cost)
        r2 = en.simulate(w, e, 0.15, INF, 15, cost)
        return tuple(np.where(hi_conf, b, a) for a, b in zip(r1, r2))
    return en.simulate(w, e, x["tp"], x["sl"], x["H"], cost, x.get("sl_px"), None, x.get("tp_px"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_integrated")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    A = prepare(data.load_pack(a.path, "krx_"))
    B = prepare(data.load_pack(a.path, ""))
    print(f"A(KRX) 급등 {len(A['ev']):,}건, B(키움) 급등 {len(B['ev']):,}건. 모델 학습 중(과거 데이터만)...")
    preds = predict_past_only(A, B)
    periods = {"A": (A, "2008-01-01", "2015-01-01"), "B": (B, "2015-01-01", None)}
    rows = []
    store = {}
    for era, (D, sd, ed) in periods.items():
        pr, thr = preds[era]
        w, ev = D["w"], D["ev"]
        dates = ev["date"].values
        inper = (ev["date"] >= sd).values & ((ev["date"] < ed).values if ed else True)
        gate = D["up20"][np.clip(ev["t"].values + 1, 0, len(D["up20"]) - 1)]
        atr = np.clip(D["X"]["atr_pct"].fillna(D["X"]["atr_pct"].median()).values, 0.01, 0.30)
        hi = np.isfinite(pr) & (pr >= thr[0.9])
        screens = {"선별없음": np.ones(len(ev), bool) & np.isfinite(pr),
                   "모델상위30%": np.isfinite(pr) & (pr >= thr[0.7]),
                   "모델상위20%": np.isfinite(pr) & (pr >= thr[0.8])}
        E = entries(w)
        for en_name, e in E.items():
            X = exits(w, e, atr, hi)
            X["P 확신도차등(상위10%:15%·15일/그외:10%·10일)"] = "P"
            for xn, x in X.items():
                ret, hold, xcol, valid = run_exit(w, e, x, a.cost, hi)
                for sn, sm in screens.items():
                    m = valid & sm & inper
                    mg = m & gate
                    s = en.stats(ret[m], hold[m], int((sm & inper).sum()))
                    mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(dates[m])) if m.sum() > 2 else (np.nan, np.nan)
                    pf, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mg], (w.t0 + xcol - P)[mg], ret[mg],
                                                slots=10, start=sd, end=ed, max_new_per_day=2)
                    yrs = pd.Series(ret[m]).groupby(ev["date"].dt.year.values[m]).mean()
                    rows.append(dict(시기=era, 선별=sn, 진입=en_name, 청산=xn, n=s["n"], 거래당=s["wmean"], 승률=s["win"],
                                     t묶음=mc / se if se else np.nan, 플러스연도비율=(yrs > 0).mean() if len(yrs) else np.nan,
                                     계좌연=pf["cagr"], 낙폭=pf["mdd"], 거래수=pf["trades"]))
    R = pd.DataFrame(rows)
    R.to_csv(out / "integrated_all.csv", index=False, encoding="utf-8-sig")
    W = R.pivot_table(index=["선별", "진입", "청산"], columns="시기",
                      values=["거래당", "t묶음", "계좌연", "낙폭", "거래수", "플러스연도비율", "승률"])
    W.columns = [f"{v}_{s}" for v, s in W.columns]
    W = W.reset_index()
    for s in ("A", "B"):
        W[f"calmar_{s}"] = W[f"계좌연_{s}"] / W[f"낙폭_{s}"].abs().replace(0, np.nan)
    W["min_calmar"] = W[["calmar_A", "calmar_B"]].min(axis=1)
    W["둘다플러스"] = (W["계좌연_A"] > 0) & (W["계좌연_B"] > 0)
    W.to_csv(out / "integrated_wide.csv", index=False, encoding="utf-8-sig")

    print("\n=== 요소별 효과 (모든 조합 평균, 거래당 절단평균) ===")
    for col in ("선별", "진입", "청산"):
        print(f"\n[{col}]")
        print(W.groupby(col)[["거래당_A", "거래당_B", "계좌연_A", "계좌연_B"]].mean().round(4).to_string())

    show = ["선별", "진입", "청산", "거래당_A", "t묶음_A", "계좌연_A", "낙폭_A", "거래수_A", "거래당_B", "t묶음_B",
            "계좌연_B", "낙폭_B", "거래수_B", "min_calmar"]
    ok = W[W["둘다플러스"] & (W["거래수_A"] >= 100) & (W["거래수_B"] >= 100)].sort_values("min_calmar", ascending=False)
    print(f"\n두 시기 모두 계좌 플러스인 조합: {len(ok)} / {len(W)}")
    print("\n=== 최선 조합 상위 12 (두 시기 중 나쁜 쪽의 연수익/낙폭 기준) ===")
    print(ok[show].head(12).round(4).to_string(index=False))
    ref = W[(W["선별"] == "모델상위20%") & (W["진입"] == "E0 다음날종가") & (W["청산"] == "F10 익절10%·10일")]
    print("\n=== 참고: 이전 검증 통과안 (모델상위20% + 다음날종가 + 익절10%·10일) ===")
    print(ref[show].round(4).to_string(index=False))
    if len(ok):
        best = ok.iloc[0]
        print(f"\n선택된 통합 전략: {best['선별']} / {best['진입']} / {best['청산']}")
        for era in ("A", "B"):
            sub = R[(R["시기"] == era) & (R["선별"] == best["선별"]) & (R["진입"] == best["진입"]) & (R["청산"] == best["청산"])]
            print(f"  {era}: 거래당 {sub['거래당'].iloc[0]:+.4f} (승률 {sub['승률'].iloc[0]:.2f}, 묶음 t {sub['t묶음'].iloc[0]:.2f}), "
                  f"플러스 연도 {sub['플러스연도비율'].iloc[0]:.0%}, 계좌 연 {sub['계좌연'].iloc[0]:+.3f}, 낙폭 {sub['낙폭'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
