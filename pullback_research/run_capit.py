"""
투매 소진 정량 점수 검증 (용량-반응 + 매매 규칙).

  # 키움 데이터 (2015~)
  python run_capit.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv --out results_capit
  # KRX 장기 (2005~, 상장폐지·회전율 포함) - run_combo 에서 저장한 패널 재사용
  python run_capit.py --source pickle --path results_combo_krx/panel_krx.pkl --out results_capit_krx
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
import combo  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def period_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return pd.Series(np.where(d < "2015-01-01", "A05_14", np.where(d < "2022-01-01", "B15_21", "C22_")), index=dates.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csv", choices=["csv", "pickle", "synthetic", "krx"])
    ap.add_argument("--path", default="kiwoom_csv")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--start", default="20050101")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--min_value", type=float, default=1e9)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_capit")
    ap.add_argument("--no_fomo", action="store_true", help="추격매수 실패 조건 없이 모든 급등 종목을 후보로 (점검·비교용)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    if a.source == "csv":
        panel = data.load_csv_dir(a.path, a.meta)
    elif a.source == "pickle":
        panel = data.load_panel(a.path)
    elif a.source == "krx":
        panel = data.load_krx(a.start, a.end)
        data.save_panel(panel, out / "panel_krx.pkl")
    else:
        panel = data.make_synthetic(2500, 1500)
        a.min_value = 0

    ev = en.find_events(panel, min_value=a.min_value)
    w = en.build_windows(panel, ev)
    S = combo.setup(w)
    fomo = np.ones(len(ev), bool) if a.no_fomo else S["fomo"]
    cand = capit.add_score(capit.candidates(w, panel, fomo))
    if len(cand) < 500:
        print(f"투매 후보일이 {len(cand)}개뿐이라 분석할 수 없습니다.")
        return
    cand["period"] = period_of(cand["date"])
    used = cand.attrs["used"]
    live = cand[~cand["burn_in"]]
    print(f"데이터 {panel['close'].index.min().date()} ~ {panel['close'].index.max().date()}, 급등 {len(ev):,}건, "
          f"추격매수 실패 {S['fomo'].sum():,}건")
    print(f"투매 후보일 {len(cand):,}개 ({"전체 급등" if a.no_fomo else "추격실패"} 종목의 D+3~D+10 신저가일), 사용 지표: {', '.join(capit.LABEL[c] for c in used)}")
    print("기간별 후보 수:", live["period"].value_counts().sort_index().to_dict())
    cand.to_csv(out / "capit_candidates.csv", index=False, encoding="utf-8-sig")

    # 1) 용량-반응: 점수·지표 5분위별 10일 선행수익 (s 종가 진입, 비용 차감 전)
    rows = []
    for per in sorted(live["period"].unique()):
        sub = live[live["period"] == per]
        for col in ["score"] + used:
            for ret in ("r5", "r10"):
                rows.append(dict(기간=per, 지표=capit.LABEL.get(col, "★ 합성 점수"), 수익=ret,
                                 **capit.dose_response(sub, col if col == "score" else col, ret)))
    dr = pd.DataFrame(rows)
    dr.to_csv(out / "capit_dose_response.csv", index=False, encoding="utf-8-sig")
    print("\n=== 용량-반응: 5분위별 10일 선행수익 (Q1=점수 낮음 … Q5=높음, t=날짜 묶음 표준오차) ===")
    show = ["기간", "지표", "n", "Q1", "Q2", "Q3", "Q4", "Q5", "spread", "t", "rho"]
    print(dr[dr["수익"] == "r10"][show].round(4).to_string(index=False))

    # 2) 매매 규칙: 점수 임계값 = 2015~2021 후보 점수의 80분위 (사전 고정)
    ref = live[live["period"] == "B15_21"]["score"]
    thr = float(ref.quantile(0.8)) if len(ref) else float(live["score"].quantile(0.8))
    print(f"\n점수 임계값(2015~2021 후보의 상위 20% 경계) = {thr:.1f}")
    qual_score = (cand["score"] >= thr) & ~cand["burn_in"]
    qual_all = ~cand["burn_in"]
    rules = {
        "점수상위20%_투매일종가": (qual_score, "none"),
        "점수상위20%_+매도소멸확인(s+1)": (qual_score, "fade"),
        "점수상위20%_+재시험확인(s+3)": (qual_score, "retest"),
        "비교: 점수무관_첫신저가일종가": (qual_all, "none"),
    }
    trow, prow, tr_store = [], [], {}
    for name, (q, delay) in rules.items():
        tr = capit.trade_rule(w, cand, q.values, S["trapped"], delay, a.cost)
        if tr is None:
            continue
        tr_store[name] = tr
        per = period_of(pd.Series(tr["date"]))
        for p in sorted(per.unique()) + ["미사용구간합(A+C)"]:
            m = (per.values == p) if p != "미사용구간합(A+C)" else per.isin(["A05_14", "C22_"]).values
            mm = tr["valid"] & m
            s = en.stats(tr["ret"][mm], tr["hold"][mm], int(m.sum()))
            trow.append(dict(규칙=name, 기간=p, **s))
        for pn, sd in (("전체", None), ("2022~", "2022-01-01")):
            s, _ = en.portfolio(tr["w"], tr["entry"], 0.15, np.inf, 10, a.cost, slots=10, start=sd,
                                max_new_per_day=2, close_exit=tr["close_exit"], tp_px=tr["tp_px"])
            prow.append(dict(규칙=name, 기간=pn, **s))
    # 기존 조합 전략(4조건) 비교
    st0 = combo.strategies(w, S)[0]
    x = st0["exit"]
    ret, hold, _, valid = en.simulate(w, st0["entry"], x["tp"], x["sl"], x["H"], a.cost, None, x["close_exit"], x["tp_px"])
    per = period_of(ev["date"])
    for p in sorted(per.unique()) + ["미사용구간합(A+C)"]:
        m = (per.values == p) if p != "미사용구간합(A+C)" else per.isin(["A05_14", "C22_"]).values
        mm = valid & st0["mask"] & m
        trow.append(dict(규칙="비교: 기존 4조건 조합", 기간=p, **en.stats(ret[mm], hold[mm], int((st0["mask"] & m).sum()))))
    tt = pd.DataFrame(trow)
    tt.to_csv(out / "capit_trades.csv", index=False, encoding="utf-8-sig")
    print("\n=== 매매 규칙 거래당 성적 (익절 추격평균가×0.98, 손절 투매일 저가 종가이탈, 최대 10일, 비용 0.3%) ===")
    print(tt[["규칙", "기간", "n", "wmean", "median", "win", "t", "hold"]].round(4).to_string(index=False))
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "capit_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 포트폴리오 (동시 10종목, 하루 신규 2종목, 연복리 cagr / 최대낙폭 mdd) ===")
    print(pf.round(4).to_string(index=False))

    # 3) 사전 판정 기준
    g = lambda per, lab="★ 합성 점수": dr[(dr["기간"] == per) & (dr["지표"] == lab) & (dr["수익"] == "r10")]
    unseen = [p for p in ("A05_14", "C22_") if p in set(live["period"])]
    T = tt.set_index(["규칙", "기간"])
    key = "미사용구간합(A+C)"
    r0, rb = ("점수상위20%_투매일종가", key), ("비교: 점수무관_첫신저가일종가", key)
    b = g("B15_21")
    checks = [
        ("2015~2021: 점수 Q5-Q1 10일수익 차이 > 0, t >= 2",
         len(b) and b["spread"].iloc[0] > 0 and b["t"].iloc[0] >= 2),
        (f"미사용 구간({', '.join(unseen)}) 각각: 차이 > 0 이고 순위상관 > 0",
         all(len(g(p)) and g(p)["spread"].iloc[0] > 0 and g(p)["rho"].iloc[0] > 0 for p in unseen)),
        ("미사용 구간 합: 점수 규칙 거래당 평균 > 0, t >= 2, 200건 이상",
         r0 in T.index and T.loc[r0, "wmean"] > 0 and T.loc[r0, "t"] >= 2 and T.loc[r0, "n"] >= 200),
        ("미사용 구간 합: 점수 무관 규칙보다 +0.5%p 이상",
         r0 in T.index and rb in T.index and T.loc[r0, "wmean"] - T.loc[rb, "wmean"] >= 0.005),
    ]
    print("\n=== 사전 판정 기준 ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(bool(v) for _, v in checks) else "기각"))
    if "A05_14" not in unseen:
        print("  (참고: KRX 2005~2014 데이터 없이 실행됨. 미사용 구간은 2022~ 하나뿐)")


if __name__ == "__main__":
    main()
