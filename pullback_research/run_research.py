"""
사용법
  # KRX (상폐 포함, 권장) - KRX 데이터포털 계정 필요
  KRX_ID=... KRX_PW=... python run_research.py --source krx --start 20150101 --end 20261113
  # 네이버 (현재 상장 종목만, 생존편향 있음)
  python run_research.py --source naver --start 2015-01-01 --end 2026-11-13
  # 엔진 검증
  python run_research.py --source synthetic
결과: results/ 폴더 (CSV, PNG, summary.md)
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

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)


def event_study(w: en.Win) -> pd.DataFrame:
    P = en.PRE
    c0 = w.c0[:, None]
    d = np.arange(1, 21)
    rel_c = w.C[:, P + d] / c0 - 1
    min_low = np.minimum.accumulate(w.L[:, P + 1:P + 21], axis=1) / c0 - 1
    max_high = np.maximum.accumulate(w.H[:, P + 1:P + 21], axis=1) / c0 - 1
    retr = (w.c0 - np.minimum.accumulate(w.L[:, P + 1:P + 21], axis=1).T).T / (w.c0 - w.p0)[:, None]
    out = pd.DataFrame({
        "day": d,
        "종가/c0_평균": np.nanmean(rel_c, 0),
        "종가/c0_중앙": np.nanmedian(rel_c, 0),
        "누적최저가/c0_중앙": np.nanmedian(min_low, 0),
        "누적최고가/c0_중앙": np.nanmedian(max_high, 0),
        "장대봉되돌림비율_중앙": np.nanmedian(retr, 0),
        "전일종가(p0)이탈비율": np.nanmean(retr >= 1, 0),
    })
    return out


def segment_table(w: en.Win, entry: en.Entry, tp, sl, H, cost) -> pd.DataFrame:
    ret, hold, _, valid = en.simulate(w, entry, tp, sl, H, cost)
    ev = w.ev.copy()
    ev["ret"], ev["valid"] = ret, valid
    ev = ev[ev.valid]
    ev["등락률구간"] = pd.cut(ev.chg, [15, 20, 25, 29.5, 31], right=False,
                          labels=["15~20", "20~25", "25~29.5", "상한가"])
    ev["거래대금구간"] = pd.qcut(ev.value, 4, labels=["Q1(작음)", "Q2", "Q3", "Q4(큼)"], duplicates="drop")
    ev["직전20일급등"] = np.where(ev.prior20 > 0, "재급등", "첫급등")
    ev["연도"] = ev.date.dt.year
    rows = []
    for col in ("등락률구간", "market", "거래대금구간", "직전20일급등", "연도"):
        for k, g in ev.groupby(col, observed=True):
            rows.append(dict(구분=col, 값=str(k), n=len(g), 평균=g.ret.mean(), 중앙=g.ret.median(),
                             승률=(g.ret > 0).mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["krx", "naver", "csv", "synthetic", "pickle"])
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--end", default="20261113")
    ap.add_argument("--path", default="")
    ap.add_argument("--meta", default=None, help="csv 소스용 종목정보(code,name,market)")
    ap.add_argument("--split", default="2022-01-01", help="이 날짜 이전=IS(개발), 이후=OOS(검증)")
    ap.add_argument("--thr", type=float, default=15.0)
    ap.add_argument("--max_chg", type=float, default=1e9, help="예: 29.5 -> 상한가 마감 종목 제외")
    ap.add_argument("--min_value", type=float, default=1e9, help="급등일 거래대금 하한(원)")
    ap.add_argument("--cost", type=float, default=0.003, help="왕복 비용(세금+수수료+슬리피지)")
    ap.add_argument("--min_n", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    if a.source == "krx":
        panel = data.load_krx(a.start, a.end)
        data.save_panel(panel, out / "panel.pkl")
    elif a.source == "naver":
        panel = data.load_naver(a.start, a.end)
        data.save_panel(panel, out / "panel.pkl")
    elif a.source == "csv":
        panel = data.load_csv_dir(a.path, a.meta)
    elif a.source == "pickle":
        panel = data.load_panel(a.path)
    else:
        panel = data.make_synthetic(2500, 1500)
        a.min_value, a.split = 0, "2020-01-01"

    ev = en.find_events(panel, thr=a.thr, min_value=a.min_value, max_chg=a.max_chg)
    print(f"이벤트 {len(ev):,}건 ({ev.date.min().date()} ~ {ev.date.max().date()})")
    ev.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    w = en.build_windows(panel, ev)

    es = event_study(w)
    es.to_csv(out / "event_study.csv", index=False, encoding="utf-8-sig")
    print(es.round(4).to_string(index=False))

    entries = en.make_entries(w)
    g = en.run_grid(w, entries, a.split, cost=a.cost)
    g.to_csv(out / "grid_all.csv", index=False, encoding="utf-8-sig")
    print(f"조합 {len(g):,}개 테스트")

    ok = g[(g.IS_n >= a.min_n) & (g.OOS_n >= a.min_n // 3)].copy()
    # 진입 규칙별 IS 최적 청산 -> OOS 성과
    best_exit = ok.loc[ok.groupby("entry")["IS_mean"].idxmax()].sort_values("IS_mean", ascending=False)
    cols = ["entry", "tp", "sl", "H", "IS_n", "IS_fill", "IS_mean", "IS_win", "IS_t", "IS_per_event",
            "OOS_n", "OOS_mean", "OOS_win", "OOS_t", "OOS_per_event", "OOS_hold"]
    best_exit[cols].to_csv(out / "best_by_entry.csv", index=False, encoding="utf-8-sig")
    print("\n=== 진입규칙별 IS 최적청산 → OOS (상위 25) ===")
    print(best_exit[cols].head(25).round(4).to_string(index=False))

    # 계열별: 청산 조합 전체의 중앙값 (과최적화에 덜 민감한 지표)
    fam = ok.groupby("family")[["IS_mean", "OOS_mean", "IS_per_event", "OOS_per_event"]].median()
    fam.to_csv(out / "family_median.csv", encoding="utf-8-sig")
    print("\n=== 계열별 (모든 청산조합 중앙값) ===")
    print(fam.round(4))

    # 상위 후보 포트폴리오 시뮬레이션 (OOS 구간)
    emap = {e.name: e for e in entries}
    top = best_exit.head(10)
    prow, seg_all = [], []
    curves = {}
    for r in top.itertuples():
        e = emap[r.entry]
        for slots in (5, 10):
            s, cs = en.portfolio(w, e, r.tp, r.sl, int(r.H), a.cost, slots=slots, start=a.split)
            prow.append(dict(entry=r.entry, tp=r.tp, sl=r.sl, H=r.H, slots=slots, **s))
            if slots == 10:
                curves[f"{r.entry}|tp{r.tp}|sl{r.sl}|H{r.H}"] = cs
        st = segment_table(w, e, r.tp, r.sl, int(r.H), a.cost)
        st.insert(0, "entry", r.entry)
        seg_all.append(st)
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "portfolio_oos.csv", index=False, encoding="utf-8-sig")
    print("\n=== OOS 포트폴리오 (동시보유 5/10종목) ===")
    print(pf.round(4).to_string(index=False))
    seg = pd.concat(seg_all)
    seg.to_csv(out / "segments_top.csv", index=False, encoding="utf-8-sig")

    # 기준선(D종가/D+1시가)과 상위 후보의 등락률 구간별 성과 (상한가 쏠림 확인용)
    print("\n=== 등락률 구간별 성과 (전체기간, IS 최적청산 기준) ===")
    show = [n for n in ("BASE_D종가", "BASE_D+1시가") if n in emap] + list(top.entry.head(4))
    for name in dict.fromkeys(show):
        r = ok[ok.entry == name].sort_values("IS_mean").iloc[-1]
        st = segment_table(w, emap[name], r.tp, r.sl, int(r.H), a.cost)
        st = st[st["구분"].isin(["등락률구간", "직전20일급등"])]
        print(f"\n[{name}] tp={r.tp} sl={r.sl} H={int(r.H)}")
        print(st[["구분", "값", "n", "평균", "중앙", "승률"]].round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(es.day, es["종가/c0_평균"], label="mean close/c0")
        ax[0].plot(es.day, es["종가/c0_중앙"], label="median close/c0")
        ax[0].plot(es.day, es["누적최저가/c0_중앙"], label="median running-min low/c0")
        ax[0].axhline(0, c="gray", lw=0.5)
        ax[0].set_title("Event study after +15% close"); ax[0].legend()
        for n_, (k, cs) in enumerate(list(curves.items())[:6], 1):
            ax[1].plot(cs.index, cs.values, label=f"#{n_}")  # 번호는 portfolio_oos.csv 순서
        ax[1].set_title("OOS equity (10 slots)"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "charts.png", dpi=110)
    except Exception as ex:  # noqa: BLE001
        print("chart fail", ex)


if __name__ == "__main__":
    main()
