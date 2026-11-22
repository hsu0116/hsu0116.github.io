"""
KRX 2005~2014 (상장폐지 포함, 이번 연구에서 한 번도 안 본 구간) 최종 판정.
판정 기준은 데이터를 보기 전에 이 파일에 고정함 (2026-11-22).

  python run_validate_krx.py --path ../stock-data

후보
  C1 메인    : 4조건 선별 + 급등일 종가 -15% 지정가(D+2부터 10일) + 익절 10%·손절없음·10일
  C1b        : 같은 진입, 익절 15%·10일 (익절 10% 로 바꾸기 전 원안)
  C2 (사전등록): 추격실패 종목 투매후보 중 점수 63.3 이상 + 다음날 매도소멸 확인 -> s+1 종가
  C3 (사전등록): 같은데 점수 무관
  C4         : 일반 급등(상한가 아님) + 다음날 +5~15% 추가상승 -> 급등일 종가 -25% 지정가(D+2부터 20일), 익절20%·20일
  C5         : 키움 2015~2026 으로 학습한 반등 모델 상위 20% + D+1 종가 매수, 익절15%·10일

주의: 2015-06-15 이전은 가격제한폭 15% -> 이 구간의 '급등(+14.5% 이상)'은 사실상 상한가 종목만임.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capit  # noqa: E402
import combo  # noqa: E402
import data  # noqa: E402
import engine as en  # noqa: E402
import screen as sc  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
START, END = "2005-01-01", "2015-01-01"
SCORE_THR = 63.3      # 키움 2015~2021 후보 점수의 80분위 (사전 등록값)


def cstats(ret, m, dates):
    if m.sum() < 2:
        return dict(n=int(m.sum()), wmean=np.nan, median=np.nan, win=np.nan, t=np.nan)
    mc, se = capit.clustered_mean_se(pd.Series(ret[m]), pd.Series(dates[m]))
    return dict(n=int(m.sum()), wmean=en.wmean(ret[m]), median=float(np.median(ret[m])),
                win=float((ret[m] > 0).mean()), t=mc / se if se else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../stock-data")
    ap.add_argument("--prefix", default="krx_")
    ap.add_argument("--kiwoom_path", default=None, help="C5 모델 학습용 키움 묶음 (기본: --path 와 같은 곳)")
    ap.add_argument("--cost", type=float, default=0.003)
    a = ap.parse_args()
    panel = data.load_pack(a.path, a.prefix)
    ev = en.find_events(panel, min_value=1e9)
    w = en.build_windows(panel, ev)
    P = en.PRE
    d = ev["date"].values
    inper = (ev["date"] >= START).values & (ev["date"] < END).values
    dates = ev["date"].values
    print(f"KRX 데이터 {panel['close'].index.min().date()} ~ {panel['close'].index.max().date()}, "
          f"종목 {panel['close'].shape[1]:,}개 (상장폐지 포함)")
    print(f"검증 구간 급등 {inper.sum():,}건 (데이터 이상 제외 {getattr(en.find_events, 'n_anomaly', 0):,}건)")
    X = sc.features(panel, w, ev)
    cond = pd.DataFrame({"S1": (X.ret20_pre <= 0).values, "S2": (X.pos52 <= -0.30).values,
                         "S3": (X.prior20 == 0).values, "S4": (X.v1_ratio <= np.log10(0.5)).values})
    cnt = cond.sum(1).values
    scr = cnt == 4
    print(f"4조건 선별 {int((scr & inper).sum()):,}건 ({(scr & inper).sum() / max(inper.sum(), 1):.1%})")
    res, checks = [], []

    # C1 / C1b
    e15 = en.limit_entry(w, w.c0 * 0.85, 11, "C1", "v", start=2)
    lk = en.locked_up(w, np.arange(len(ev)), np.full(len(ev), P + 1))
    C1px = w.C[:, P + 1]
    eA = en.Entry("A", np.where(lk | ~np.isfinite(C1px), -1, P + 1), C1px, "close")
    out = {}
    for nm, tp in (("C1 메인(익절10%)", 0.10), ("C1b 원안(익절15%)", 0.15)):
        ret, hold, xcol, valid = en.simulate(w, e15, tp, np.inf, 10, a.cost)
        out[nm] = (ret, xcol, valid)
        for g, gm in (("선별", scr), ("전체", np.ones(len(ev), bool))):
            res.append(dict(후보=nm, 그룹=g, **cstats(ret, valid & gm & inper, dates)))
        m = valid & scr & inper
        pf, _ = en.portfolio_trades(w.dates, (w.t0 + e15.col - P)[m], (w.t0 + xcol - P)[m], ret[m],
                                    slots=10, start=START, end=END, max_new_per_day=2)
        res.append(dict(후보=nm, 그룹="계좌(선별)", n=pf["trades"], wmean=pf["cagr"], median=pf["mdd"], win=np.nan, t=np.nan))
    retA, _, _, vA = en.simulate(w, eA, 0.15, np.inf, 10, a.cost)
    dose = {c: cstats(retA, vA & (cnt == c) & inper, dates)["wmean"] for c in range(5)}
    print("\n조건 개수별(D+1 종가 매수, 익절15%·10일) 거래당:", {k: round(v, 4) for k, v in dose.items()})

    # C2 / C3
    S = combo.setup(w)
    cand = capit.add_score(capit.candidates(w, panel, S["fomo"]))
    for nm, q in (("C2 점수63.3+매도소멸", (cand.score >= SCORE_THR) & ~cand.burn_in),
                  ("C3 점수무관+매도소멸", ~cand.burn_in)):
        tr = capit.trade_rule(w, cand, q.values, S["trapped"], "fade", a.cost)
        if tr is None:
            res.append(dict(후보=nm, 그룹="-", n=0))
            continue
        ed = pd.to_datetime(tr["date"])
        m = tr["valid"] & (ed >= START) & (ed < END)
        s = en.stats(tr["ret"][m], tr["hold"][m], int(m.sum()))
        res.append(dict(후보=nm, 그룹="-", n=s["n"], wmean=s["wmean"], median=s["median"], win=s["win"], t=s["t"]))

    # C4
    r1 = w.C[:, P + 1] / w.c0 - 1
    upper = w.c0 / w.p0 - 1 >= w.LIM[:, P] - 0.005
    gapfade = (w.O[:, P + 1] >= w.c0 * 1.03) & (w.C[:, P + 1] < w.O[:, P + 1]) & (r1 < 0.05)
    seg = (~upper) & (r1 >= 0.05) & (r1 < 0.15) & ~gapfade
    e25 = en.limit_entry(w, w.c0 * 0.75, 20, "C4", "v", start=2)
    ret, hold, xcol, valid = en.simulate(w, e25, 0.20, np.inf, 20, a.cost)
    res.append(dict(후보="C4 일반급등+다음날추가상승 -25%지정가", 그룹="-", **cstats(ret, valid & seg & inper, dates)))
    res.append(dict(후보="C4 비교: 전체 -25%지정가", 그룹="-", **cstats(ret, valid & inper, dates)))

    # C5 모델 (키움으로 학습)
    kp = data.load_pack(a.kiwoom_path or a.path, "")
    kev = en.find_events(kp, min_value=1e9)
    kw = en.build_windows(kp, kev)
    KX, KY = sc.features(kp, kw, kev), sc.targets(kw)
    ok = KY.y_bar.notna().values
    mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)
    mdl.fit(KX.values[ok], KY.y_bar.values[ok].astype(int))
    thr = np.quantile(mdl.predict_proba(KX.values[ok])[:, 1], 0.8)
    pr = mdl.predict_proba(X.values)[:, 1]
    top = pr >= thr
    for g, gm in (("모델 상위20%", top), ("전체", np.ones(len(ev), bool))):
        res.append(dict(후보="C5 반등모델 + D+1종가", 그룹=g, **cstats(retA, vA & gm & inper, dates)))
    R = pd.DataFrame(res)
    print("\n=== 2005~2014 결과 (거래당 절단평균, t=같은 날 묶음 / C2·C3 는 사전등록대로 단순 t) ===")
    print("    (계좌 행: n=거래수, wmean=연복리, median=최대낙폭)")
    print(R.round(4).to_string(index=False))

    T = R.set_index(["후보", "그룹"])
    c1s, c1a = T.loc[("C1 메인(익절10%)", "선별")], T.loc[("C1 메인(익절10%)", "전체")]
    c1p = T.loc[("C1 메인(익절10%)", "계좌(선별)")]
    checks = [
        ("C1 메인: 선별 거래당 > 0, 묶음 t >= 2", c1s["wmean"] > 0 and c1s["t"] >= 2),
        ("C1 메인: 선별이 전체보다 +0.5%p 이상", c1s["wmean"] - c1a["wmean"] >= 0.005),
        ("C1 메인: 조건 4개 만족 > 0개 만족", dose[4] > dose[0]),
        ("C1 메인: 계좌 연복리 > 0, 최대낙폭 -30% 이내", c1p["wmean"] > 0 and c1p["median"] > -0.30),
    ]
    for nm in ("C2 점수63.3+매도소멸", "C3 점수무관+매도소멸"):
        r = T.loc[(nm, "-")]
        checks.append((f"{nm}: 거래당 > 0, t >= 2, 150건 이상", r["n"] >= 150 and r["wmean"] > 0 and r["t"] >= 2))
    r = T.loc[("C4 일반급등+다음날추가상승 -25%지정가", "-")]
    checks.append(("C4: 거래당 > 0, 묶음 t >= 2", r["wmean"] > 0 and r["t"] >= 2))
    a5, b5 = T.loc[("C5 반등모델 + D+1종가", "모델 상위20%")], T.loc[("C5 반등모델 + D+1종가", "전체")]
    checks.append(("C5: 모델 상위20% 거래당 > 0, 전체보다 +0.5%p 이상", a5["wmean"] > 0 and a5["wmean"] - b5["wmean"] >= 0.005))
    print("\n=== 사전 판정 (2005~2014) ===")
    for k, v in checks:
        print(f"  [{'통과' if v else '실패'}] {k}")
    main_ok = all(bool(v) for _, v in checks[:4])
    print("  => 메인 후보(C1): " + ("채택" if main_ok else "기각"))


if __name__ == "__main__":
    main()
