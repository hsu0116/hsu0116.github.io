"""
반등 가능성 높은 급등주 선별 모델 검증 (걷는 검증: 매년 그 전 해까지로 학습 -> 그 해 예측).
  python run_screen.py --source pack --path ../stock-data --out results_screen
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import screen as sc  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
SPLIT = 2022
FIRST_TEST = 2018


def models():
    return {
        "GBM": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0),
        "LOGIT": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.1, max_iter=1000)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pack", choices=["pack", "pickle", "csv"])
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_screen")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = {"pack": lambda: data.load_pack(a.path), "pickle": lambda: data.load_panel(a.path),
             "csv": lambda: data.load_csv_dir(a.path, a.meta)}[a.source]()
    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    X = sc.features(panel, w, ev)
    Y = sc.targets(w)
    yr = ev["date"].dt.year.values
    ok = Y["y_bar"].notna().values
    print(f"급등 {len(ev):,}건, 목표값 있는 건 {ok.sum():,}. 전체 반등 성공률(+10%가 -10%보다 먼저) = {Y.y_bar[ok].mean():.3f}")

    # 1) 걷는 검증 예측
    pred = {m: np.full(len(ev), np.nan) for m in models()}
    for y in range(FIRST_TEST, yr.max() + 1):
        tr = ok & (yr < y)
        te = (yr == y)
        if te.sum() == 0:
            continue
        for mn, mdl in models().items():
            mdl.fit(X.values[tr], Y.y_bar.values[tr].astype(int))
            pred[mn][te] = mdl.predict_proba(X.values[te])[:, 1]
    rows = []
    for mn, p in pred.items():
        for y in range(FIRST_TEST, yr.max() + 1):
            m = ok & (yr == y) & np.isfinite(p)
            if m.sum() > 50:
                rows.append(dict(model=mn, year=y, n=m.sum(), auc=roc_auc_score(Y.y_bar[m], p[m])))
        for lab, m in (("2018-2021", ok & (yr < SPLIT) & (yr >= FIRST_TEST)), ("2022-2026", ok & (yr >= SPLIT))):
            m = m & np.isfinite(p)
            rows.append(dict(model=mn, year=lab, n=m.sum(), auc=roc_auc_score(Y.y_bar[m], p[m])))
    A = pd.DataFrame(rows)
    A.to_csv(out / "screen_auc.csv", index=False, encoding="utf-8-sig")
    print("\n=== 예측력 (AUC: 0.5 = 무작위, 1.0 = 완벽) ===")
    print(A.pivot(index="year", columns="model", values="auc").round(3).to_string())

    # 2) 예측 확률 10분위별 실제 결과 (GBM, 2022~)
    p = pred["GBM"]
    D = pd.DataFrame({"p": p, "y": Y.y_bar, "r10": Y.y_ret10, "date": ev["date"], "yr": yr})
    for lab, m in (("2018-2021", (yr >= FIRST_TEST) & (yr < SPLIT)), ("2022-2026", yr >= SPLIT)):
        d = D[m & ok & np.isfinite(p)].copy()
        d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=False) + 1
        g = d.groupby("dec").agg(n=("y", "size"), 예측=("p", "mean"), 실제성공률=("y", "mean"), 수익10일_평균=("r10", "mean"),
                                 수익10일_중앙=("r10", "median"))
        print(f"\n=== GBM 예측 10분위별 실제 결과 ({lab}, 10=반등 가능성 가장 높음) ===")
        print(g.round(4).to_string())

    # 3) 특성별 방향 안정성 (단변량 순위상관: 학습기간 vs 검증기간)
    frow = []
    for c in sc.FEATURES:
        r_is = X[c][ok & (yr < SPLIT)].rank().corr(Y.y_bar[ok & (yr < SPLIT)])
        r_os = X[c][ok & (yr >= SPLIT)].rank().corr(Y.y_bar[ok & (yr >= SPLIT)])
        frow.append(dict(특성=sc.LABEL[c], 상관_2015_21=r_is, 상관_2022_26=r_os, 방향일치=np.sign(r_is) == np.sign(r_os)))
    F = pd.DataFrame(frow).sort_values("상관_2022_26", key=abs, ascending=False)
    F.to_csv(out / "screen_features.csv", index=False, encoding="utf-8-sig")
    print("\n=== 특성별 반등 성공과의 순위상관 (+면 클수록 반등 잘 됨) ===")
    print(F.round(3).to_string(index=False))

    # 4) 매매: 선별(예측 상위 20%, 임계값은 그 해 이전 예측값 분포로 정함 -> 미래정보 없음)
    P = en.PRE
    thr = np.full(len(ev), np.nan)
    for y in range(FIRST_TEST, yr.max() + 1):
        past = np.isfinite(p) & (yr < y)
        if past.sum() < 500:   # 첫 해는 그 해 학습 데이터 예측 분포 대신 그 해 분포 사용 불가 -> 학습기간 내 예측으로 대체
            past = np.isfinite(p) & (yr <= y)
        thr[yr == y] = np.nanquantile(p[past], 0.8)
    top = np.isfinite(p) & (p >= thr)
    bot = np.isfinite(p) & (p <= np.nanquantile(p[np.isfinite(p)], 0.2))
    lk = en.locked_up(w, np.arange(len(ev)), np.full(len(ev), P + 1))
    C1 = w.C[:, P + 1]
    entries = {
        "D+1종가매수 (익절15%·손절없음·10일)": (en.Entry("a", np.where(lk | ~np.isfinite(C1), -1, P + 1), C1, "close"),
                                       dict(tp=0.15, sl=np.inf, H=10)),
        "D+1종가매수 (+10%/-10%·10일)": (en.Entry("b", np.where(lk | ~np.isfinite(C1), -1, P + 1), C1, "close"),
                                      dict(tp=0.10, sl=0.10, H=10)),
        "D종가-25% 지정가 (D+2부터 20일, 익절15%·20일)": (en.limit_entry(w, w.c0 * 0.75, 21, "c", "s", start=2),
                                                dict(tp=0.15, sl=np.inf, H=20)),
    }
    up = sc.market_index(panel)
    up20 = (up > up.rolling(20).mean()).values[np.clip(ev["t"].values + 1, 0, len(up) - 1)]
    trow, prow = [], []
    for en_name, (e, x) in entries.items():
        ret, hold, xcol, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost)
        for gname, g in (("전체", np.ones(len(ev), bool) & np.isfinite(p)), ("선별 상위20%", top), ("하위20%", bot)):
            for lab, m in (("2018-2021", (yr >= FIRST_TEST) & (yr < SPLIT)), ("2022-2026", yr >= SPLIT)):
                mm = valid & g & m
                s = en.stats(ret[mm], hold[mm], int((g & m).sum()))
                trow.append(dict(매수법=en_name, 그룹=gname, 기간=lab, **{k: s[k] for k in ("n", "wmean", "median", "win", "t")}))
            if gname != "하위20%":
                for gate in (False, True):
                    mm = valid & g & ((up20 > 0) if gate else True)
                    s, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[mm], (w.t0 + xcol - P)[mm], ret[mm],
                                               slots=10, start=f"{SPLIT}-01-01", max_new_per_day=2)
                    prow.append(dict(매수법=en_name, 그룹=gname, 시장상승국면만=gate, **s))
    T = pd.DataFrame(trow)
    T.to_csv(out / "screen_trades.csv", index=False, encoding="utf-8-sig")
    print("\n=== 매매 결과: 선별 vs 전체 (거래당 절단평균) ===")
    print(T.round(4).to_string(index=False))
    PF = pd.DataFrame(prow)
    PF.to_csv(out / "screen_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 계좌 2022~ (동시 10종목, 하루 신규 2종목) ===")
    print(PF.round(4).to_string(index=False))

    # 5) 사전 판정
    auc22 = A[(A.model == "GBM") & (A.year == "2022-2026")].auc.iloc[0]
    d = D[(yr >= SPLIT) & ok & np.isfinite(p)]
    tsel = top[(yr >= SPLIT) & ok & np.isfinite(p)]
    m1, s1 = capit.clustered_mean_se(d.y[tsel], d.date[tsel])
    m0 = d.y.mean()
    TT = T.set_index(["매수법", "그룹", "기간"])
    k = "D+1종가매수 (익절15%·손절없음·10일)"
    a_top = TT.loc[(k, "선별 상위20%", "2022-2026")]
    a_all = TT.loc[(k, "전체", "2022-2026")]
    PFi = PF.set_index(["매수법", "그룹", "시장상승국면만"])
    checks = [
        ("2022~ AUC >= 0.55 (GBM)", auc22 >= 0.55),
        ("2022~ 선별 상위20% 반등 성공률이 전체보다 +3%p 이상, t >= 3",
         (m1 - m0) >= 0.03 and (m1 - m0) / s1 >= 3),
        ("D+1종가매수: 선별 상위20% 2022~ 거래당 > 0, t >= 2, 전체보다 +0.5%p 이상",
         a_top.wmean > 0 and a_top.t >= 2 and a_top.wmean - a_all.wmean >= 0.005),
        ("계좌(시장상승국면만, 하루 2종목): 선별 매매 중 하나라도 연수익 > +2.6% (기존 최선)",
         PFi.xs("선별 상위20%", level="그룹").xs(True, level="시장상승국면만")["cagr"].max() > 0.026),
    ]
    print(f"\n2022~ 선별 상위20% 반등 성공률 {m1:.3f} vs 전체 {m0:.3f} (t={(m1 - m0) / s1:.2f})")
    print("\n=== 사전 판정 기준 ===")
    for kk, v in checks:
        print(f"  [{'통과' if v else '실패'}] {kk}")
    print("  => " + ("채택" if all(bool(v) for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
