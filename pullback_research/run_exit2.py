"""
통합 전략(모델 상위30% + 급등일종가-15% 지정가 + 시장상승국면 + 10종목·하루2) 고정, 청산만 종목별로.

1) 기대 상승폭 예측 익절: 진입 후 10일 최대 상승폭(MFE)의 중앙값을 모델로 예측 -> 익절 = 진입가 × (1 + k × 예측), k ∈ {0.6, 0.8, 1.0}
   (모델은 과거만으로 학습: A 는 A 의 지난해까지, B 는 A 전체 + B 의 지난해까지. 익절폭은 5%~30% 로 제한)
2) 종목별 손절: 변동성 2·3·4배 / 급등 전 가격 종가 이탈 / 급등일 저가 종가 이탈 / 진입일 저가 종가 이탈 (익절 10%·10일 유지)
3) 종목 특성 그룹별 청산: 변동성 3분위 × 모델점수(상위10% / 10~30%) = 6그룹, 그룹마다 익절{5,7,10,15,20%}×보유{5,10,15일} 중 최선을
   한 시기에서 골라 다른 시기에 적용 (A->B, B->A 양방향)
기준선: 고정 익절 10%·10일·손절 없음
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import product
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
pd.set_option("display.max_colwidth", 50)
INF = np.inf
P = en.PRE
ERAS = {"A": ("2008-01-01", "2015-01-01"), "B": ("2015-01-01", None)}


def mfe_target(w, e, H=10):
    n, K = len(e.col), w.K
    cols = np.arange(K)
    ec = np.where(e.col >= 0, e.col, K)
    m = (cols[None, :] > ec[:, None]) & (cols[None, :] <= ec[:, None] + H)
    hmax = np.max(np.where(m, np.nan_to_num(w.H, nan=-np.inf), -np.inf), axis=1)
    out = hmax / e.price - 1
    out[(e.col < 0) | ~np.isfinite(out)] = np.nan
    return out


def predict_mfe(A, B, eA, eB):
    """과거만 학습한 MFE 중앙값 예측 (전체 체결 거래로 학습)."""
    res = {}
    for name, D, e, years, extra in (("A", A, eA, range(2008, 2015), None), ("B", B, eB, range(2015, B["yr"].max() + 1), (A, eA))):
        y_all = mfe_target(D["w"], e)
        pr = np.full(len(y_all), np.nan)
        okD = np.isfinite(y_all)
        for yy in years:
            Xt = [D["X"].values[okD & (D["yr"] < yy)]]
            Yt = [y_all[okD & (D["yr"] < yy)]]
            if extra is not None:
                Ex, ee = extra
                ye = mfe_target(Ex["w"], ee)
                oke = np.isfinite(ye) & (Ex["yr"] >= 2005) & (Ex["yr"] < 2015)
                Xt.append(Ex["X"].values[oke])
                Yt.append(ye[oke])
            m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_depth=3, learning_rate=0.05,
                                              max_iter=200, min_samples_leaf=200, random_state=0)
            m.fit(np.vstack(Xt), np.clip(np.concatenate(Yt), -0.5, 1.0))
            te = D["yr"] == yy
            pr[te] = m.predict(D["X"].values[te])
        res[name] = pr
    return res


def evaluate(D, e, sel, gate, sd, ed, ret, hold, xcol, valid, cost=None):
    ev = D["ev"]
    w = D["w"]
    inper = (ev["date"] >= sd).values & ((ev["date"] < ed).values if ed else True)
    m = valid & sel & inper
    mg = m & gate
    mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(ev["date"].values[m])) if m.sum() > 2 else (np.nan, np.nan)
    pf, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mg], (w.t0 + xcol - P)[mg], ret[mg], slots=10,
                                start=sd, end=ed, max_new_per_day=2)
    return dict(n=int(m.sum()), 거래당=en.wmean(ret[m]) if m.any() else np.nan, 승률=float((ret[m] > 0).mean()) if m.any() else np.nan,
                t묶음=mc / se if se else np.nan, 계좌연=pf["cagr"], 낙폭=pf["mdd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_exit2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    A = ri.prepare(data.load_pack(a.path, "krx_"))
    B = ri.prepare(data.load_pack(a.path, ""))
    preds = ri.predict_past_only(A, B)
    Dm = {"A": A, "B": B}
    ctx = {}
    for era, D in Dm.items():
        pr, thr = preds[era]
        w, ev = D["w"], D["ev"]
        e = en.limit_entry(w, w.c0 * 0.85, 11, "E4", "i", start=2)
        sel = np.isfinite(pr) & (pr >= thr[0.7])
        top10 = np.isfinite(pr) & (pr >= thr[0.9])
        gate = D["up20"][np.clip(ev["t"].values + 1, 0, len(D["up20"]) - 1)]
        atr = np.clip(D["X"]["atr_pct"].fillna(D["X"]["atr_pct"].median()).values, 0.01, 0.30)
        ctx[era] = dict(D=D, e=e, sel=sel, top10=top10, gate=gate, atr=atr)
    print("MFE(진입 후 10일 최대 상승폭) 예측 모델 학습 중...")
    mfe_pred = predict_mfe(A, B, ctx["A"]["e"], ctx["B"]["e"])
    rows = []

    def add(study, name, era, res):
        rows.append(dict(연구=study, 청산=name, 시기=era, **res))

    for era, c in ctx.items():
        D, e, sel, gate, atr = c["D"], c["e"], c["sel"], c["gate"], c["atr"]
        w = D["w"]
        sd, ed = ERAS[era]
        p = e.price
        K = w.K
        # 기준선
        r = en.simulate(w, e, 0.10, INF, 10, a.cost)
        add("기준", "고정 익절10%·10일·손절없음", era, evaluate(D, e, sel, gate, sd, ed, *r))
        # 1) 기대 상승폭 예측 익절
        mp = mfe_pred[era]
        real = mfe_target(w, e)
        ok = sel & np.isfinite(mp) & np.isfinite(real)
        print(f"[{era}] MFE 예측 vs 실제 순위상관(선별 종목): {pd.Series(mp[ok]).rank().corr(pd.Series(real[ok]).rank()):.3f}, "
              f"예측 중앙값 {np.nanmedian(mp[sel]):.3f}")
        for k in (0.6, 0.8, 1.0):
            tp = np.clip(k * np.nan_to_num(mp, nan=0.10), 0.05, 0.30)
            r = en.simulate(w, e, INF, INF, 10, a.cost, None, None, p * (1 + tp))
            add("1 예측익절", f"익절 = {k}×예측상승폭 (5~30%)", era, evaluate(D, e, sel, gate, sd, ed, *r))
        # 2) 종목별 손절 (익절 10%·10일 유지)
        for kk in (2, 3, 4):
            r = en.simulate(w, e, 0.10, INF, 10, a.cost, p * (1 - kk * atr))
            add("2 종목별손절", f"손절 = 진입가 - {kk}×변동성", era, evaluate(D, e, sel, gate, sd, ed, *r))
        b = lambda lv: np.broadcast_to(lv[:, None], (len(lv), K))
        r = en.simulate(w, e, 0.10, INF, 10, a.cost, None, b(w.p0))
        add("2 종목별손절", "손절 = 급등 전 가격 종가 이탈", era, evaluate(D, e, sel, gate, sd, ed, *r))
        l0 = np.where(w.l0 < p, w.l0, np.nan)
        r = en.simulate(w, e, 0.10, INF, 10, a.cost, None, b(l0))
        add("2 종목별손절", "손절 = 급등일 저가 종가 이탈", era, evaluate(D, e, sel, gate, sd, ed, *r))
        rr = np.arange(len(p))
        el = np.where(e.col >= 0, w.L[rr, np.clip(e.col, 0, K - 1)], np.nan)
        r = en.simulate(w, e, 0.10, INF, 10, a.cost, None, b(el))
        add("2 종목별손절", "손절 = 진입일 저가 종가 이탈", era, evaluate(D, e, sel, gate, sd, ed, *r))
        # 3) 그룹별 그리드 결과 저장 (선택은 아래에서)
        c["grid"] = {}
        for tp, H in product((0.05, 0.07, 0.10, 0.15, 0.20), (5, 10, 15)):
            c["grid"][(tp, H)] = en.simulate(w, e, tp, INF, H, a.cost)

    # 3) 그룹: 변동성 3분위(경계는 고르는 시기의 선별 종목 기준) × 모델점수(상위10% / 10~30%)
    for src, dst in (("A", "B"), ("B", "A")):
        cs, cd = ctx[src], ctx[dst]
        q1, q2 = np.quantile(cs["atr"][cs["sel"]], [1 / 3, 2 / 3])

        def groups(c):
            vb = np.digitize(c["atr"], [q1, q2])
            return vb * 2 + c["top10"].astype(int)          # 0~5
        gs, gd = groups(cs), groups(cd)
        sd_s, ed_s = ERAS[src]
        ins = (cs["D"]["ev"]["date"] >= sd_s).values & ((cs["D"]["ev"]["date"] < ed_s).values if ed_s else True)
        choice = {}
        for g in range(6):
            best, bv = (0.10, 10), -np.inf
            for key, (ret, hold, xcol, valid) in cs["grid"].items():
                m = valid & cs["sel"] & ins & (gs == g)
                if m.sum() >= 30:
                    v = en.wmean(ret[m])
                    if v > bv:
                        best, bv = key, v
            choice[g] = best
        n = len(cd["e"].col)
        ret, hold, xcol = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
        valid = np.zeros(n, bool)
        for g, key in choice.items():
            rg = cd["grid"][key]
            m = gd == g
            ret[m], hold[m], xcol[m], valid[m] = rg[0][m], rg[1][m], rg[2][m], rg[3][m]
        sd, ed = ERAS[dst]
        lab = ", ".join(f"G{g}:{int(k[0]*100)}%/{k[1]}일" for g, k in choice.items())
        add("3 그룹별청산", f"{src}에서 고른 그룹별 기준 [{lab}]", dst,
            evaluate(cd["D"], cd["e"], cd["sel"], cd["gate"], sd, ed, ret, hold, xcol, valid))
    R = pd.DataFrame(rows)
    R.to_csv(out / "exit2.csv", index=False, encoding="utf-8-sig")
    W = R.pivot_table(index=["연구", "청산"], columns="시기", values=["거래당", "승률", "t묶음", "계좌연", "낙폭", "n"], aggfunc="first")
    W.columns = [f"{v}_{s}" for v, s in W.columns]
    W = W.reset_index()
    print("\n=== 결과 (그룹 G = 변동성 3분위(0 낮음~2 높음)×2 + 모델 상위10% 여부) ===")
    print(R.round(4).to_string(index=False))
    base = R[R["연구"] == "기준"].set_index("시기")
    print("\n=== 판정 (기준선 대비: 두 시기 모두 거래당 +0.2%p 이상 & 계좌 연수익/낙폭 개선) ===")
    for (st, nm), g in R[R["연구"] != "기준"].groupby(["연구", "청산"]):
        g = g.set_index("시기")
        oks = []
        for era in g.index:
            cal = g.loc[era, "계좌연"] / abs(g.loc[era, "낙폭"]) if g.loc[era, "낙폭"] else np.nan
            cal0 = base.loc[era, "계좌연"] / abs(base.loc[era, "낙폭"])
            oks.append(g.loc[era, "거래당"] - base.loc[era, "거래당"] >= 0.002 and cal > cal0)
        print(f"  [{'통과' if all(oks) else '실패'}] {st} | {nm[:60]}")


if __name__ == "__main__":
    main()
