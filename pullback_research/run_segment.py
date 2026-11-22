"""
급등주 세분화 + 분류별 매수 방법 검증.
  python run_segment.py --source pickle --path panel.pkl --out results_segment
  python run_segment.py --source pack --path ../stock-data --out results_segment
분류별로 2015~2021 에서 (매수방법 × 청산) 중 최선을 고르고, 2022~ 에 그대로 적용.
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
import segment as sg  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)
SPLIT = "2022-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pack", choices=["pack", "pickle", "csv"])
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--min_n", type=int, default=80, help="분류별 선택 시 개발구간 최소 거래 수")
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--out", default="results_segment")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    panel = {"pack": lambda: data.load_pack(a.path), "pickle": lambda: data.load_panel(a.path),
             "csv": lambda: data.load_csv_dir(a.path, a.meta)}[a.source]()

    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    seg = sg.classify(w)
    is_m = (ev["date"] < SPLIT).values
    print(f"급등 {len(ev):,}건 ({ev.date.min().date()} ~ {ev.date.max().date()})")

    # 분류 분포와 분류별 이후 흐름 (D+1 종가 대비 D+11 종가)
    P = en.PRE
    fwd = w.C[:, P + 11] / w.C[:, P + 1] - 1
    desc = pd.DataFrame({"seg": seg, "IS": is_m, "fwd": fwd}).groupby("seg").agg(
        건수=("fwd", "size"), 비율=("fwd", lambda s: len(s) / len(ev)),
        D1종가대비_10일후_평균=("fwd", "mean"), 중앙=("fwd", "median"))
    print("\n=== 분류별 분포와 D+1 종가 이후 10거래일 흐름 ===")
    print(desc.round(4).to_string())

    M = sg.methods(w)
    rows, store = [], {}
    for mn, e in M.items():
        for xn, x in sg.EXITS.items():
            ret, hold, xcol, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], a.cost)
            store[(mn, xn)] = (ret, hold, xcol, valid, e)
            for sn in sorted(seg.unique()):
                sm = (seg == sn).values
                for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
                    mm = valid & sm & pm
                    s = en.stats(ret[mm], hold[mm], int((sm & pm).sum()))
                    rows.append(dict(seg=sn, method=mn, exit=xn, period=tag, **s))
            for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
                mm = valid & pm
                s = en.stats(ret[mm], hold[mm], int(pm.sum()))
                rows.append(dict(seg="(전체)", method=mn, exit=xn, period=tag, **s))
    R = pd.DataFrame(rows)
    R.to_csv(out / "segment_grid.csv", index=False, encoding="utf-8-sig")
    piv = R.pivot_table(index=["seg", "method", "exit"], columns="period", values=["n", "wmean", "t", "win"])
    piv.columns = [f"{p}_{v}" for v, p in piv.columns]
    piv = piv.reset_index()
    print(f"\n테스트한 칸: 분류 {seg.nunique()}개 × 매수방법 {len(M)} × 청산 {len(sg.EXITS)} = {len(piv)}개")

    # 분류별 선택 (개발구간 절단평균 최대, 최소 거래 수, 플러스일 때만 매매)
    cells = piv[(piv.seg != "(전체)") & (piv.seg != "데이터없음") & (piv.IS_n >= a.min_n)]
    pick = cells.loc[cells.groupby("seg")["IS_wmean"].idxmax()].copy()
    pick["매매"] = pick["IS_wmean"] > 0
    print("\n=== 분류별 선택 (2015~2021 최선) → 2022~ 적용 ===")
    cols = ["seg", "method", "exit", "매매", "IS_n", "IS_wmean", "IS_win", "OOS_n", "OOS_wmean", "OOS_win", "OOS_t"]
    print(pick[cols].round(4).to_string(index=False))
    pick.to_csv(out / "segment_pick.csv", index=False, encoding="utf-8-sig")

    # 세분화 전략 합치기
    ret_all = np.full(len(ev), np.nan)
    hold_all = np.full(len(ev), np.nan)
    tin = np.full(len(ev), -1)
    tout = np.full(len(ev), -1)
    use = np.zeros(len(ev), bool)
    for r in pick[pick["매매"]].itertuples():
        ret, hold, xcol, valid, e = store[(r.method, r.exit)]
        sm = (seg == r.seg).values & valid
        ret_all[sm], hold_all[sm] = ret[sm], hold[sm]
        tin[sm] = w.t0[sm] + (e.col[sm] - P)
        tout[sm] = w.t0[sm] + (xcol[sm] - P)
        use |= sm
    # 세분화 없는 최선 (같은 메뉴에서 전체 이벤트 기준)
    allc = piv[(piv.seg == "(전체)")]
    best = allc.loc[allc["IS_wmean"].idxmax()]
    bret, bhold, bxcol, bvalid, be = store[(best.method, best.exit)]
    bench = en.limit_entry(w, w.c0 * 0.75, 20, "bench", "bench")
    kret, khold, kxcol, kvalid = en.simulate(w, bench, 0.15, np.inf, 20, a.cost)

    def per(name, ret, hold, valid):
        out_ = {}
        for tag, pm in (("IS", is_m), ("OOS", ~is_m)):
            mm = valid & pm
            out_[tag] = en.stats(ret[mm], hold[mm], int(pm.sum()))
        return out_
    summ = {
        "세분화 전략(분류별 최선)": (per("seg", ret_all, hold_all, use), tin[use], tout[use], ret_all[use]),
        f"세분화 없음 최선: {best.method}/{best.exit}": (per("best", bret, bhold, bvalid),
                                                    (w.t0 + be.col - P)[bvalid], (w.t0 + bxcol - P)[bvalid], bret[bvalid]),
        "기존 기준: D종가-25% 지정가(20일)": (per("bench", kret, khold, kvalid),
                                       (w.t0 + bench.col - P)[kvalid], (w.t0 + kxcol - P)[kvalid], kret[kvalid]),
    }
    srow = []
    for k, (st, ti, to, rr) in summ.items():
        pf, _ = en.portfolio_trades(w.dates, ti, to, rr, slots=10, start=SPLIT, max_new_per_day=2)
        srow.append(dict(전략=k, IS_n=st["IS"]["n"], IS_wmean=st["IS"]["wmean"], OOS_n=st["OOS"]["n"],
                         OOS_wmean=st["OOS"]["wmean"], OOS_median=st["OOS"]["median"], OOS_win=st["OOS"]["win"],
                         OOS_t=st["OOS"]["t"], 계좌2022_cagr=pf["cagr"], 계좌2022_mdd=pf["mdd"]))
    S = pd.DataFrame(srow)
    S.to_csv(out / "segment_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 전략 비교 (계좌: 동시 10종목, 하루 신규 2종목, 2022~) ===")
    print(S.round(4).to_string(index=False))

    # 과최적화 진단: 칸별 개발 성적과 검증 성적의 순위상관
    both = piv[(piv.seg != "(전체)") & (piv.IS_n >= a.min_n) & (piv.OOS_n >= a.min_n)]
    rho = both["IS_wmean"].rank().corr(both["OOS_wmean"].rank())
    gap = (pick["OOS_wmean"] - pick["IS_wmean"]).mean()
    print(f"\n과최적화 진단: 칸 {len(both)}개의 개발-검증 성적 순위상관 = {rho:.3f}, "
          f"선택된 칸의 평균 성적 하락(검증-개발) = {gap:+.4f}")
    # 분류 자체의 정보: 같은 매수방법·청산에서 분류별 성적 차이가 기간 간 유지되나
    seg_only = both.groupby("seg")[["IS_wmean", "OOS_wmean"]].mean()
    rho2 = seg_only["IS_wmean"].rank().corr(seg_only["OOS_wmean"].rank())
    print(f"분류별 평균 성적(모든 방법 평균)의 개발-검증 순위상관 = {rho2:.3f}")
    print(seg_only.round(4).to_string())

    st_seg = S.iloc[0]
    st_best = S.iloc[1]
    st_b = S.iloc[2]
    checks = [
        ("세분화 전략 2022~ 거래당 평균 > 0, t >= 2", st_seg.OOS_wmean > 0 and st_seg.OOS_t >= 2),
        ("세분화 전략이 세분화 없음 최선보다 +0.5%p 이상", st_seg.OOS_wmean - st_best.OOS_wmean >= 0.005),
        ("세분화 전략 계좌 연수익이 두 비교 대상보다 높음",
         st_seg["계좌2022_cagr"] > max(st_best["계좌2022_cagr"], st_b["계좌2022_cagr"])),
        ("칸별 개발-검증 순위상관 >= 0.2 (분류·방법 선택이 정보를 담고 있음)", rho >= 0.2),
    ]
    print("\n=== 사전 판정 기준 ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    print("  => " + ("채택" if all(bool(v) for _, v in checks) else "기각"))


if __name__ == "__main__":
    main()
