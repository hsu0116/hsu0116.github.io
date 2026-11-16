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
import ytrules as yt  # noqa: E402

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
    print("지표 계산 중...")
    G = yt.compute_panels(panel)
    X = {k: yt.gather(G[k], ev, w.K) for k in ("ma5", "ma10", "ma20", "sd20", "rsi", "slowk", "slowd", "osc")}
    entries += yt.make_yt_entries(w, X)
    xa = yt.exit_arrays_factory(w, X)
    exits = en.basic_exits() + yt.struct_exits()
    print(f"진입규칙 {len(entries)}개 × 청산 {len(exits)}개 백테스트 중... (수십 분 걸릴 수 있음)")
    g = en.run_grid(w, entries, a.split, cost=a.cost, exits=exits, exit_arrays=xa)
    g.to_csv(out / "grid_all.csv", index=False, encoding="utf-8-sig")
    print(f"조합 {len(g):,}개 테스트")

    ok = g[(g.IS_n >= a.min_n) & (g.OOS_n >= a.min_n // 3)].copy()
    cols = ["entry", "tp", "sl", "H", "xs", "IS_n", "IS_fill", "IS_mean", "IS_win", "IS_t", "IS_per_event",
            "OOS_n", "OOS_mean", "OOS_win", "OOS_t", "OOS_per_event", "OOS_hold"]
    short = ["entry", "tp", "sl", "H", "xs", "IS_n", "IS_mean", "IS_t", "OOS_n", "OOS_mean", "OOS_win", "OOS_t"]
    # 진입 규칙별 IS 최적 청산 -> OOS 성과
    best_exit = ok.loc[ok.groupby("entry")["IS_mean"].idxmax()].sort_values("IS_mean", ascending=False)
    best_exit[cols].to_csv(out / "best_by_entry.csv", index=False, encoding="utf-8-sig")
    print("\n=== 진입규칙별 IS 최적청산 → OOS (상위 25) ===")
    print(best_exit[short].head(25).round(4).to_string(index=False))
    print(f"\n전체 {len(best_exit)}개 규칙 중 OOS 플러스 {(best_exit.OOS_mean > 0).sum()}개, OOS t>2 {(best_exit.OOS_t > 2).sum()}개")

    ytb = best_exit[best_exit.entry.str.startswith("YT_")]
    print("\n=== 유튜브식 지표 기법 (IS 최적청산 → OOS) ===")
    print(ytb[short].round(4).to_string(index=False))

    # 구조적 청산(유튜브식 손절) vs 일반 %청산: 규칙별 IS 최적끼리 비교
    sb = ok[ok["xs"] != ""]
    bb = ok[ok["xs"] == ""]
    sbest = sb.loc[sb.groupby("entry")["IS_mean"].idxmax()].set_index("entry")
    bbest = bb.loc[bb.groupby("entry")["IS_mean"].idxmax()].set_index("entry")
    cmpx = pd.DataFrame({"pct_exit": bbest.tp.astype(str) + "/" + bbest.sl.astype(str) + "/" + bbest.H.astype(str),
                         "pct_IS": bbest.IS_mean, "pct_OOS": bbest.OOS_mean,
                         "struct_exit": sbest["xs"] + "/" + sbest.tp.astype(str) + "/" + sbest.H.astype(str),
                         "struct_IS": sbest.IS_mean, "struct_OOS": sbest.OOS_mean}).dropna()
    cmpx.to_csv(out / "exit_compare.csv", encoding="utf-8-sig")
    print("\n=== 청산방식 비교 (규칙별 IS 최적, 중앙값) ===")
    print(cmpx[["pct_IS", "pct_OOS", "struct_IS", "struct_OOS"]].median().round(4).to_string())
    print("구조적 청산 종류별 OOS 중앙값:")
    print(sb.groupby("xs")["OOS_mean"].median().round(4).to_string())

    fam = ok.groupby("family")[["IS_mean", "OOS_mean", "IS_per_event", "OOS_per_event"]].median()
    fam.to_csv(out / "family_median.csv", encoding="utf-8-sig")
    print("\n=== 계열별 (모든 청산조합 중앙값) ===")
    print(fam.round(4))

    # 질적 필터 연구
    feats = yt.event_features(panel, G, ev)
    feats.assign(date=ev.date, code=ev.code).to_csv(out / "event_features.csv", index=False, encoding="utf-8-sig")
    print("\n=== 질적 필터 해당 비율 ===")
    print(feats.mean().round(3).to_string())
    picks = pd.concat([best_exit.head(6), ytb.head(6),
                       best_exit[best_exit.entry == "PCT_종가-25%_W20"]]).drop_duplicates("entry")
    emap = {e.name: e for e in entries}
    fs, combo = yt.filter_study(w, emap, picks, feats, a.split, a.cost, xa)
    fs.to_csv(out / "filters.csv", index=False, encoding="utf-8-sig")
    combo.to_csv(out / "filter_combo.csv", index=False, encoding="utf-8-sig")
    print("\n=== 필터별 효과 (후보 전략 평균, 필터=예 일 때 OOS 평균 - 전체 OOS 평균) ===")
    allrow = fs[fs["filter"] == "(전체)"].set_index("entry")["OOS_mean"]
    yes = fs[(fs["filter"] != "(전체)") & (fs["value"])].copy()
    yes["dOOS"] = yes.OOS_mean - yes.entry.map(allrow)
    yes["dIS"] = yes.IS_mean - yes.entry.map(fs[fs["filter"] == "(전체)"].set_index("entry")["IS_mean"])
    eff = yes.groupby("filter").agg(dIS=("dIS", "mean"), dOOS=("dOOS", "mean"), OOS_n=("OOS_n", "mean"))
    eff.to_csv(out / "filter_effect.csv", encoding="utf-8-sig")
    print(eff.sort_values("dOOS", ascending=False).round(4).to_string())
    print("\n=== IS 로 고른 필터 2개 조합 → OOS ===")
    print(combo[["entry", "filters", "IS_n", "IS_mean", "base_OOS", "OOS_n", "OOS_mean", "OOS_win"]].round(4).to_string(index=False))

    # 포트폴리오 (OOS 구간, 동시 10종목): 기본 / 시장상승국면만 / 하루 1종목 / 필터조합
    prow = []
    curves = {}
    fmask = {}
    for r in combo.itertuples():
        m = np.ones(len(ev), bool)
        if r.filters != "(없음)":
            for part in r.filters.split(" & "):
                c, v = part.rsplit("=", 1)
                m &= feats[c].values == (v == "예")
        fmask[r.entry] = m
    for r in picks.itertuples():
        e = emap[r.entry]
        sp, ce = xa(r.xs) if r.xs else (None, None)
        variants = {"기본": dict(), "시장상승국면20만": dict(mask=feats["시장상승국면20"].values),
                    "하루1종목제한": dict(max_new_per_day=1), "필터조합": dict(mask=fmask[r.entry]),
                    "필터조합+하루1종목": dict(mask=fmask[r.entry], max_new_per_day=1)}
        for vn, kw in variants.items():
            s_, cs = en.portfolio(w, e, r.tp, r.sl, int(r.H), a.cost, slots=10, start=a.split,
                                  sl_px=sp, close_exit=ce, **kw)
            prow.append(dict(entry=r.entry, exit=f"{r.tp}/{r.sl}/{int(r.H)}/{r.xs}", variant=vn, **s_))
            if vn == "기본":
                curves[r.entry] = cs
    pf = pd.DataFrame(prow)
    pf.to_csv(out / "portfolio_oos.csv", index=False, encoding="utf-8-sig")
    print("\n=== OOS 포트폴리오 (동시보유 10종목, 연복리 cagr / 최대낙폭 mdd) ===")
    print(pf.round(4).to_string(index=False))

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
