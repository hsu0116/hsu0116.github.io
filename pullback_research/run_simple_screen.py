"""
단순 선별 규칙 + 매수법 결합 검증.

선별 조건 (D+1 장 마감 시점, 기준값은 맞추지 않고 의미 있는 값으로 고정)
  S1 급등 전 20일 수익률 <= 0        (이미 오른 종목이 아님)
  S2 52주 최고가 대비 -30% 이하       (바닥권)
  S3 첫 급등 (최근 20일 내 +15% 급등 없음)
  S4 다음날 거래량 <= 급등일 거래량 × 0.5 (관심이 빨리 식음)
  메인 선별 = S1 & S2 & S3 & S4

매수법
  A  D+1 종가 매수, 익절 15%·손절 없음·10일
  A2 D+2 시가 매수, 같은 청산 (진입 시점 민감도 점검)
  B  D 종가 -25% 지정가 (D+2부터 20일), 익절 15%·20일
  B2 D 종가 -15% 지정가 (D+2부터 10일), 익절 15%·10일

주의: 어떤 특성을 쓸지는 2022~ 결과까지 보고 정했으므로 2022~ 도 완전한 미사용 검증은 아님.
  python run_simple_screen.py --source pack --path ../stock-data --out results_simple_screen
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
import screen as sc  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pack", choices=["pack", "pickle", "csv"])
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_simple_screen")
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
    per = np.where(yr < 2022, "2015-2021", "2022-2026")

    S = pd.DataFrame({
        "S1_급등전20일수익률<=0": (X["ret20_pre"] <= 0).values,
        "S2_52주고점대비-30%이하": (X["pos52"] <= -0.30).values,
        "S3_첫급등": (X["prior20"] == 0).values,
        "S4_다음날거래량<=급등일절반": (X["v1_ratio"] <= np.log10(0.5)).values,
    })
    cnt = S.sum(axis=1).values
    main_m = cnt == 4
    print(f"급등 {len(ev):,}건. 조건별 해당 비율:")
    print(S.mean().round(3).to_string())
    print(f"4개 모두 만족: {main_m.mean():.1%} ({main_m.sum():,}건)")

    lk = en.locked_up(w, np.arange(len(ev)), np.full(len(ev), P + 1))
    C1, O2 = w.C[:, P + 1], w.O[:, P + 2]
    E = {
        "A_D+1종가·익절15%·10일": (en.Entry("A", np.where(lk | ~np.isfinite(C1), -1, P + 1), C1, "close"), 0.15, 10),
        "A2_D+2시가·익절15%·10일": (en.Entry("A2", np.where(np.isfinite(O2), P + 2, -1), O2, "open"), 0.15, 10),
        "B_D종가-25%지정가·익절15%·20일": (en.limit_entry(w, w.c0 * 0.75, 21, "B", "s", start=2), 0.15, 20),
        "B2_D종가-15%지정가·익절15%·10일": (en.limit_entry(w, w.c0 * 0.85, 11, "B2", "s", start=2), 0.15, 10),
    }
    R = {k: en.simulate(w, e, tp, np.inf, H, a.cost) + (e,) for k, (e, tp, H) in E.items()}
    dates = pd.Series(ev["date"].values)

    def st(ret, m):
        if m.sum() < 2:
            return dict(n=int(m.sum()), wmean=np.nan, median=np.nan, win=np.nan, t_묶음=np.nan)
        mc, se = capit.clustered_mean_se(pd.Series(ret[m]), dates[m])
        return dict(n=int(m.sum()), wmean=en.wmean(ret[m]), median=float(np.median(ret[m])),
                    win=float((ret[m] > 0).mean()), t_묶음=mc / se if se else np.nan)

    # 1) 조건 개수별 성적 (용량-반응)
    rows = []
    for k, (ret, hold, xcol, valid, e) in R.items():
        for c in range(5):
            for pp in ("2015-2021", "2022-2026"):
                rows.append(dict(매수법=k, 만족조건수=c, 기간=pp, **st(ret, valid & (cnt == c) & (per == pp))))
    D = pd.DataFrame(rows)
    D.to_csv(out / "count_dose.csv", index=False, encoding="utf-8-sig")
    print("\n=== 만족한 선별 조건 개수별 거래당 성적 (절단평균) ===")
    print(D.pivot_table(index=["매수법", "만족조건수"], columns="기간", values=["n", "wmean"]).round(4).to_string())

    # 2) 조건 하나씩 빼보기 (메인에서 하나 제외 시)
    rows = []
    ret, hold, xcol, valid, e = R["A_D+1종가·익절15%·10일"]
    for drop in [None] + list(S.columns):
        m = np.ones(len(ev), bool)
        for c in S.columns:
            if c != drop:
                m &= S[c].values
        for pp in ("2015-2021", "2022-2026"):
            rows.append(dict(규칙="메인(4개 모두)" if drop is None else f"{drop} 제외", 기간=pp, **st(ret, valid & m & (per == pp))))
    print("\n=== 조건 하나씩 빼보기 (매수법 A) ===")
    print(pd.DataFrame(rows).pivot_table(index="규칙", columns="기간", values=["n", "wmean", "t_묶음"]).round(4).to_string())

    # 3) 메인 선별 vs 전체: 매수법별, 기간별, 연도별
    rows, yrows = [], []
    for k, (ret, hold, xcol, valid, e) in R.items():
        for g, gm in (("메인 선별", main_m), ("전체", np.ones(len(ev), bool))):
            for pp in ("2015-2021", "2022-2026"):
                rows.append(dict(매수법=k, 그룹=g, 기간=pp, **st(ret, valid & gm & (per == pp))))
            for y in range(2015, yr.max() + 1):
                mm = valid & gm & (yr == y)
                yrows.append(dict(매수법=k, 그룹=g, year=y, n=int(mm.sum()), wmean=en.wmean(ret[mm]) if mm.any() else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(out / "main_vs_all.csv", index=False, encoding="utf-8-sig")
    print("\n=== 메인 선별 vs 전체 (t_묶음 = 같은 날 묶음 표준오차 기준) ===")
    print(T.round(4).to_string(index=False))
    Yd = pd.DataFrame(yrows).pivot_table(index="year", columns=["매수법", "그룹"], values="wmean")
    Yd.to_csv(out / "by_year.csv", encoding="utf-8-sig")
    print("\n=== 연도별 거래당 절단평균 ===")
    print(Yd.round(4).to_string())

    # 4) 계좌 2022~ (동시 10종목, 하루 2종목) / 시장 상승국면 필터 유무
    idx = sc.market_index(panel)
    up20 = (idx > idx.rolling(20).mean()).values[np.clip(ev["t"].values + 1, 0, len(idx) - 1)]
    prow = []
    for k, (ret, hold, xcol, valid, e) in R.items():
        for g, gm in (("메인 선별", main_m), ("전체", np.ones(len(ev), bool))):
            for gate in (False, True):
                m = valid & gm & (up20 if gate else True)
                s, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[m], (w.t0 + xcol - P)[m], ret[m],
                                           slots=10, start="2022-01-01", max_new_per_day=2)
                s2, _ = en.portfolio_trades(w.dates, (w.t0 + e.col - P)[m], (w.t0 + xcol - P)[m], ret[m],
                                            slots=10, start="2015-01-01", end="2022-01-01", max_new_per_day=2)
                prow.append(dict(매수법=k, 그룹=g, 시장상승국면만=gate, cagr_2022=s["cagr"], mdd_2022=s["mdd"],
                                 거래_2022=s["trades"], cagr_15_21=s2["cagr"], mdd_15_21=s2["mdd"]))
    PF = pd.DataFrame(prow)
    PF.to_csv(out / "portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 계좌 (동시 10종목, 하루 신규 2종목, 연복리 / 최대낙폭) ===")
    print(PF.round(4).to_string(index=False))

    # 5) 판정
    TT = T.set_index(["매수법", "그룹", "기간"])
    a_m, a_all = TT.loc[("A_D+1종가·익절15%·10일", "메인 선별", "2022-2026")], TT.loc[("A_D+1종가·익절15%·10일", "전체", "2022-2026")]
    a2_m = TT.loc[("A2_D+2시가·익절15%·10일", "메인 선별", "2022-2026")]
    dA = D[(D["매수법"] == "A_D+1종가·익절15%·10일")]
    dose_ok = all(dA[(dA["기간"] == pp) & (dA["만족조건수"] == 4)]["wmean"].iloc[0] >
                  dA[(dA["기간"] == pp) & (dA["만족조건수"] == 0)]["wmean"].iloc[0] for pp in ("2015-2021", "2022-2026"))
    ya = Yd[("A_D+1종가·익절15%·10일", "메인 선별")] - Yd[("A_D+1종가·익절15%·10일", "전체")]
    PFi = PF.set_index(["매수법", "그룹", "시장상승국면만"])
    best_pf = PFi.xs("메인 선별", level="그룹")
    checks = [
        ("A: 메인 선별 2022~ 거래당 > 0, 묶음 t >= 2, 전체보다 +1%p 이상",
         a_m.wmean > 0 and a_m.t_묶음 >= 2 and a_m.wmean - a_all.wmean >= 0.01),
        ("A2(D+2 시가 매수)로도 메인 선별 2022~ 거래당 > 0", a2_m.wmean > 0),
        ("조건 4개 만족 > 0개 만족 (두 기간 모두, 매수법 A)", dose_ok),
        (f"연도별: 메인 선별이 전체보다 좋은 해 {int((ya > 0).sum())}/{int(ya.notna().sum())} (기준: 80% 이상)",
         (ya > 0).mean() >= 0.8),
        ("계좌 2022~: 어떤 매수법이든 메인 선별이 연 +2.6% 초과이면서 최대낙폭 -46%보다 작음",
         ((best_pf["cagr_2022"] > 0.026) & (best_pf["mdd_2022"] > -0.46)).any()),
    ]
    print("\n=== 판정 기준 ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택(단, 2022~ 도 특성 선택에 쓰였으므로 신규 데이터 확인 필요)" if all(bool(v) for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
