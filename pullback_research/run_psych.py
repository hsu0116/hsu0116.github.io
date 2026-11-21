"""
심리 기반 눌림목 전략 검증.
  python run_psych.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv --out results_psych
결과: results_psych/ (psych_all.csv, psych_portfolio.csv 등) + 화면 요약
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import engine as en  # noqa: E402
import psych  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)

MAIN = ("PSY_통합", "본전익절+급등전가격손절_20일", "통합게이트")   # 사전 등록 전략
BENCH = ("BENCH_종가-25%지정가", "15%익절_손절없음_20일", "시장공포제외")


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
    ap.add_argument("--out", default="results_psych")
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
    meta = panel["meta"]
    ev["sector"] = meta["sector"].reindex(ev.code).values if "sector" in meta else ""
    has_sector = "sector" in meta and (meta["sector"].fillna("") != "").any()
    print(f"이벤트 {len(ev):,}건, 데이터 이상 제외 {getattr(en.find_events, 'n_anomaly', 0):,}건, "
          f"업종정보 {'있음' if has_sector else '없음(대장주 = 그날 전체 거래대금 1위)'}")
    w = en.build_windows(panel, ev)
    up20 = market_up20(panel)
    g = psych.gates(w, panel, ev, up20)
    print("\n=== 게이트 해당 비율 ===")
    print(g.mean().round(3).to_string())

    E = psych.make_psych_entries(w)
    E[BENCH[0]] = en.limit_entry(w, w.c0 * 0.75, 20, BENCH[0], "bench")
    X = psych.exit_specs(w)
    res = psych.run(w, E, X, g, a.split, a.cost)
    res.to_csv(out / "psych_all.csv", index=False, encoding="utf-8-sig")
    print(f"\n테스트한 조합 {len(res)}개 (진입 {len(E)} × 청산 {len(X)} × 게이트 5). "
          f"우연으로도 OOS t>2 가 약 {len(res) * 0.025:.0f}개 나올 수 있어 사전 등록 전략과 t>3 위주로 판단")

    show = ["entry", "exit", "gate", "IS_n", "IS_wmean", "IS_median", "IS_win", "OOS_n", "OOS_fill",
            "OOS_wmean", "OOS_median", "OOS_win", "OOS_t", "OOS_hold"]
    key = lambda t: (res.entry == t[0]) & (res.exit == t[1]) & (res.gate == t[2])
    print("\n=== [사전 등록] 통합 전략 vs 기준(종가-25% 지정가 + 시장공포 제외) ===")
    print(res[key(MAIN) | key(BENCH)][show].round(4).to_string(index=False))

    print("\n=== 가설별 진입 (게이트 없음, 공통 청산 15%익절·손절없음·20일) ===")
    base = res[(res.gate == "게이트없음") & (res.exit == "15%익절_손절없음_20일")]
    print(base[show].sort_values("IS_wmean", ascending=False).round(4).to_string(index=False))

    print("\n=== 게이트 효과 (진입·청산 전체 평균, 게이트없음 대비 윈저화평균 차이) ===")
    piv = res.pivot_table(index=["entry", "exit"], columns="gate", values=["IS_wmean", "OOS_wmean"])
    eff = pd.DataFrame({gn: {"dIS": (piv["IS_wmean"][gn] - piv["IS_wmean"]["게이트없음"]).mean(),
                             "dOOS": (piv["OOS_wmean"][gn] - piv["OOS_wmean"]["게이트없음"]).mean()}
                        for gn in piv["IS_wmean"].columns if gn != "게이트없음"}).T
    print(eff.round(4).to_string())

    print("\n=== 청산 효과 (진입 전체 평균, 게이트 없음) ===")
    x0 = res[res.gate == "게이트없음"].groupby("exit")[["IS_wmean", "OOS_wmean", "OOS_median", "OOS_win", "OOS_hold"]].mean()
    print(x0.round(4).to_string())

    ok = res[(res.IS_n >= 200) & (res.OOS_n >= 100)]
    top = ok.sort_values("IS_wmean", ascending=False).head(10)
    print("\n=== IS 윈저화평균 상위 10 조합 → OOS ===")
    print(top[show].round(4).to_string(index=False))

    # 포트폴리오 (OOS, 동시 10종목, 하루 신규 2종목)
    gm = {"게이트없음": None, "FOMO실패제외": g["FOMO실패아님"].values, "시장공포제외": g["시장공포아님"].values,
          "대장주만": g["대장주"].values, "통합게이트": g["통합게이트"].values}
    cand = [MAIN, BENCH] + [tuple(r) for r in top[["entry", "exit", "gate"]].head(4).values]
    prow = []
    for en_name, xn, gn in dict.fromkeys(cand):
        x = X[xn]
        for mx in (0, 2):
            s, _ = en.portfolio(w, E[en_name], x["tp"], x["sl"], x["H"], a.cost, slots=10, start=a.split,
                                mask=gm[gn], max_new_per_day=mx, close_exit=x.get("close_exit"), tp_px=x.get("tp_px"))
            prow.append(dict(entry=en_name, exit=xn, gate=gn, 하루신규상한=mx or "없음", **s))
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "psych_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== OOS 포트폴리오 (동시 10종목, 연복리 cagr / 최대낙폭 mdd) ===")
    print(pf.round(4).to_string(index=False))

    # 연도별 (사전 등록 전략)
    e = E[MAIN[0]]
    x = X[MAIN[1]]
    ret, hold, _, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost, None, x.get("close_exit"), x.get("tp_px"))
    m = valid & g["통합게이트"].values
    yr = pd.DataFrame({"year": ev.date.dt.year[m], "ret": ret[m]}).groupby("year")["ret"].agg(
        n="size", 평균="mean", 중앙="median", 승률=lambda s: (s > 0).mean())
    yr.to_csv(out / "psych_main_by_year.csv", encoding="utf-8-sig")
    print("\n=== 통합 전략 연도별 ===")
    print(yr.round(4).to_string())


if __name__ == "__main__":
    main()
