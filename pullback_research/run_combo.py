"""
추격매수 실패 + 강제매도 + 투매 소진 조합 전략 검증.

  # 키움 데이터 (2015~, 현재 상장 종목)
  python run_combo.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv --out results_combo
  # 급등 기준 민감도 (표본 확대)
  python run_combo.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv --thr 10 --out results_combo_thr10
  # KRX 전체 이력 (2005~, 상장폐지 포함) - 표본 확대 본검증. KRX_ID / KRX_PW 환경변수 필요
  python run_combo.py --source krx --start 20050101 --end 20261120 --long --out results_combo_krx
  # 한 번 받은 KRX 패널 재사용
  python run_combo.py --source pickle --path results_combo_krx/panel_krx.pkl --long --out results_combo_krx
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
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 46)

MAIN = "메인: 추격실패(A|B)+강제매도구간+투매소진"
CMP1B = "비교①b: 투매소진, 추격실패 아닌 종목"
BENCH = "비교④: 종가-25% 지정가(기존 최선)"


def market_up20(panel: dict) -> np.ndarray:
    C = panel["close"]
    ret = (C / C.ffill().shift(1) - 1).where(lambda x: x.abs() < 0.35)
    idx = (1 + ret.median(axis=1).fillna(0)).cumprod()
    return (idx > idx.rolling(20).mean()).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csv", choices=["csv", "pickle", "synthetic", "krx"])
    ap.add_argument("--path", default="kiwoom_csv")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--start", default="20050101")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--thr", type=float, default=15.0, help="급등 기준(%%), 2015-06-15 이후")
    ap.add_argument("--thr_pre", type=float, default=None, help="2015-06-15 이전 급등 기준(기본 14.5=사실상 상한가)")
    ap.add_argument("--long", action="store_true", help="장기 검증: 2005-14 / 2015-21 / 2022- 3구간 + 전체")
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
    elif a.source == "krx":
        panel = data.load_krx(a.start, a.end)
        data.save_panel(panel, out / "panel_krx.pkl")
        print("KRX 패널 저장 ->", out / "panel_krx.pkl")
    else:
        panel = data.make_synthetic(2000, 1500)
        a.min_value, a.split = 0, "2019-01-01"

    ev = en.find_events(panel, thr=a.thr, thr_pre=a.thr_pre, min_value=a.min_value)
    w = en.build_windows(panel, ev)
    up20 = market_up20(panel)
    S = combo.setup(w)
    print(f"데이터 {panel['close'].index.min().date()} ~ {panel['close'].index.max().date()}, "
          f"종목 {panel['close'].shape[1]:,}개")
    print(f"이벤트 {len(ev):,}건 (급등 기준 {a.thr}%, 데이터 이상 제외 {getattr(en.find_events, 'n_anomaly', 0):,}건)")
    print(f"추격매수 실패: A형 {S['fomoA'].mean():.1%}, B형 {S['fomoB'].mean():.1%}, A|B {S['fomo'].mean():.1%}")

    d = ev["date"]
    if a.long:
        periods = {"A05_14": (d < "2015-01-01").values, "B15_21": ((d >= "2015-01-01") & (d < "2022-01-01")).values,
                   "C22_": (d >= "2022-01-01").values, "ALL": np.ones(len(ev), bool)}
    else:
        periods = {"IS": (d < a.split).values, "OOS": (d >= a.split).values}
    strats = combo.strategies(w, S)
    res = combo.evaluate(w, strats, periods, a.cost, up20)
    res.to_csv(out / "combo_results.csv", index=False, encoding="utf-8-sig")
    print("\n=== 거래당 성적 (wmean=상하위1% 절단 평균, t=단순평균 기준) ===")
    for tag in periods:
        cols = ["전략"] + [f"{tag}_{k}" for k in ("n", "wmean", "median", "win", "t", "hold")]
        print(f"\n[{tag}]")
        print(res[cols].round(4).to_string(index=False))

    # 포트폴리오: 장기 모드는 전체 기간과 2022~ 두 가지
    starts = {"전체": None, "2022~": "2022-01-01"} if a.long else {"검증구간": a.split}
    prow = []
    for st in strats:
        x = st["exit"]
        for pn, sd in starts.items():
            for mx in (0, 2):
                s, _ = en.portfolio(w, st["entry"], x["tp"], x["sl"], x["H"], a.cost, slots=10, start=sd,
                                    mask=st["_mask"], max_new_per_day=mx, close_exit=x["close_exit"], tp_px=x["tp_px"])
                prow.append(dict(전략=st["name"], 기간=pn, 하루신규상한=mx or "없음", **s))
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "combo_portfolio.csv", index=False, encoding="utf-8-sig")
    print("\n=== 포트폴리오 (동시 10종목, 연복리 cagr / 최대낙폭 mdd) ===")
    print(pf.round(4).to_string(index=False))

    m0 = strats[0]
    mm = m0["_valid"] & m0["_mask"]
    yr = pd.DataFrame({"year": d.dt.year[mm], "ret": m0["_ret"][mm]}).groupby("year")["ret"].agg(
        n="size", 평균="mean", 중앙="median", 승률=lambda s: (s > 0).mean())
    yr.to_csv(out / "combo_main_by_year.csv", encoding="utf-8-sig")
    print("\n=== 메인 전략 연도별 ===")
    print(yr.round(4).to_string())

    R = res.set_index("전략")
    g = lambda tag, k, s=MAIN: R.loc[s, f"{tag}_{k}"]
    if a.long:
        checks = [
            ("신규 표본(2005~2014) 거래당 평균 > 0 이고 t >= 2", g("A05_14", "wmean") > 0 and g("A05_14", "t") >= 2),
            ("전체(2005~) 거래당 평균 > 0 이고 t >= 3", g("ALL", "wmean") > 0 and g("ALL", "t") >= 3),
            ("전체 거래 수 >= 300", g("ALL", "n") >= 300),
            ("세 구간 중 2개 이상 플러스", sum(g(t, "wmean") > 0 for t in ("A05_14", "B15_21", "C22_")) >= 2),
            ("전체: 추격실패 아닌 종목의 투매소진 대비 +0.5%p 이상", g("ALL", "wmean") - g("ALL", "wmean", CMP1B) >= 0.005),
        ]
    else:
        pm = pf[(pf["전략"] == MAIN) & (pf["하루신규상한"] == 2)].iloc[0]
        pb = pf[(pf["전략"] == BENCH) & (pf["하루신규상한"] == 2)].iloc[0]
        checks = [
            ("IS 거래당 평균(절단) > 0", g("IS", "wmean") > 0),
            ("OOS 거래당 평균(절단) > 0", g("OOS", "wmean") > 0),
            ("OOS t >= 3", g("OOS", "t") >= 3),
            ("OOS 비교①a 대비 +0.5%p 이상",
             g("OOS", "wmean") - g("OOS", "wmean", "비교①a: 투매소진만(전체 급등)") >= 0.005),
            ("계좌: 비교④보다 연수익 높거나 최대낙폭 절반 이하",
             (pm.cagr > pb.cagr) or (abs(pm.mdd) <= abs(pb.mdd) / 2)),
            ("거래 수 IS+OOS >= 200", g("IS", "n") + g("OOS", "n") >= 200),
        ]
    print("\n=== 사전 판정 기준 (메인 전략) ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(v for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
