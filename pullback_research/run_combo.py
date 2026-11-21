"""
추격매수 실패 + 강제매도 + 투매 소진 조합 전략 검증.
  python run_combo.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv --out results_combo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 46)


def market_up20(panel: dict) -> np.ndarray:
    C = panel["close"]
    ret = (C / C.ffill().shift(1) - 1).where(lambda x: x.abs() < 0.35)
    idx = (1 + ret.median(axis=1).fillna(0)).cumprod()
    return (idx > idx.rolling(20).mean()).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csv", choices=["csv", "pickle", "synthetic"])
    ap.add_argument("--path", default="kiwoom_csv")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--min_value", type=float, default=1e9)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_combo")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    if a.source == "csv":
        panel = data.load_csv_dir(a.path, a.meta)
    elif a.source == "pickle":
        panel = data.load_panel(a.path)
    else:
        panel = data.make_synthetic(2000, 1500)
        a.min_value, a.split = 0, "2019-01-01"

    ev = en.find_events(panel, min_value=a.min_value)
    w = en.build_windows(panel, ev)
    up20 = market_up20(panel)
    S = combo.setup(w)
    print(f"이벤트 {len(ev):,}건 (데이터 이상 제외 {getattr(en.find_events, 'n_anomaly', 0):,}건)")
    print(f"추격매수 실패: A형 {S['fomoA'].mean():.1%}, B형 {S['fomoB'].mean():.1%}, A|B {S['fomo'].mean():.1%}")

    strats = combo.strategies(w, S)
    res = combo.evaluate(w, strats, a.split, a.cost, up20)
    res.to_csv(out / "combo_results.csv", index=False, encoding="utf-8-sig")
    show = ["전략", "IS_n", "IS_wmean", "IS_median", "IS_win", "IS_t", "OOS_n", "OOS_wmean", "OOS_median",
            "OOS_win", "OOS_t", "OOS_hold"]
    print("\n=== 거래당 성적 (wmean=상하위1% 절단 평균, t=단순평균 기준) ===")
    print(res[show].round(4).to_string(index=False))

    # 포트폴리오
    prow, curves = [], {}
    for st in strats:
        x = st["exit"]
        for mx in (0, 2):
            s, cs = en.portfolio(w, st["entry"], x["tp"], x["sl"], x["H"], a.cost, slots=10, start=a.split,
                                 mask=st["_mask"], max_new_per_day=mx, close_exit=x["close_exit"], tp_px=x["tp_px"])
            prow.append(dict(전략=st["name"], 하루신규상한=mx or "없음", **s))
            if mx == 2:
                curves[st["name"]] = cs
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "combo_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== OOS 포트폴리오 (2022~, 동시 10종목, 연복리 cagr / 최대낙폭 mdd) ===")
    print(pf.round(4).to_string(index=False))

    # 연도별 (메인)
    m0 = strats[0]
    mm = m0["_valid"] & m0["_mask"]
    yr = pd.DataFrame({"year": ev.date.dt.year[mm], "ret": m0["_ret"][mm]}).groupby("year")["ret"].agg(
        n="size", 평균="mean", 중앙="median", 승률=lambda s: (s > 0).mean())
    yr.to_csv(out / "combo_main_by_year.csv", encoding="utf-8-sig")
    print("\n=== 메인 전략 연도별 ===")
    print(yr.round(4).to_string())

    # 사전 판정 기준
    R = res.set_index("전략")
    main_n = R.index[0]
    cmp1 = "비교①a: 투매소진만(전체 급등)"
    p_main = pf[(pf["전략"] == main_n) & (pf["하루신규상한"] == 2)].iloc[0]
    p_b = pf[(pf["전략"] == "비교④: 종가-25% 지정가(기존 최선)") & (pf["하루신규상한"] == 2)].iloc[0]
    checks = [
        ("IS 거래당 평균(절단) > 0", R.loc[main_n, "IS_wmean"] > 0),
        ("OOS 거래당 평균(절단) > 0", R.loc[main_n, "OOS_wmean"] > 0),
        ("OOS t >= 3", R.loc[main_n, "OOS_t"] >= 3),
        ("OOS 비교①a 대비 +0.5%p 이상", R.loc[main_n, "OOS_wmean"] - R.loc[cmp1, "OOS_wmean"] >= 0.005),
        ("계좌: 비교④보다 연수익 높거나 최대낙폭 절반 이하",
         (p_main.cagr > p_b.cagr) or (abs(p_main.mdd) <= abs(p_b.mdd) / 2)),
        ("거래 수 IS+OOS >= 200", R.loc[main_n, "IS_n"] + R.loc[main_n, "OOS_n"] >= 200),
    ]
    print("\n=== 사전 판정 기준 (메인 전략) ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(v for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
