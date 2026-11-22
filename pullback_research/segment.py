"""
급등주 세분화(D일 모양 × D+1일 흐름) + 분류별 매수 방법 선택.

분류 (D+1 장 마감 후 확정, 진입은 D+1 종가 또는 D+2 부터)
  D일   : 상한가마감 / 일반급등
  D+1일 : r1 = D+1 종가 / D 종가 - 1, O1 = D+1 시가
    연속급등     r1 >= +15%
    추가상승     +5% <= r1 < +15%
    갭업후밀림   O1 >= D종가*1.03 이고 D+1 종가 < O1 이고 r1 < +5%  ("시가 올랐다 종가 훅")
    강보합       0 <= r1 < +5%
    약보합       -5% <= r1 < 0
    급락         r1 < -5%
  -> 2 × 6 = 12 분류
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en
from psych import _cummin_after, _cummax_after, _prev

PRE = en.PRE
D1_TYPES = ["연속급등", "추가상승", "갭업후밀림", "강보합", "약보합", "급락"]


def classify(w: en.Win) -> pd.Series:
    P = PRE
    c0, p0 = w.c0, w.p0
    O1, C1 = w.O[:, P + 1], w.C[:, P + 1]
    r1 = C1 / c0 - 1
    gapfade = (O1 >= c0 * 1.03) & (C1 < O1) & (r1 < 0.05)
    t = np.select([r1 >= 0.15, r1 >= 0.05, gapfade, r1 >= 0, r1 >= -0.05],
                  ["연속급등", "추가상승", "갭업후밀림", "강보합", "약보합"], "급락")
    upper = w.c0 / w.p0 - 1 >= w.LIM[:, P] - 0.005
    dtype = np.where(upper, "상한가", "일반")
    lab = pd.Series([f"{a}·{b}" for a, b in zip(dtype, t)], index=w.ev.index)
    lab[~np.isfinite(C1)] = "데이터없음"
    return lab


def methods(w: en.Win) -> dict[str, en.Entry]:
    P = PRE
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    c0, p0 = w.c0, w.p0
    C1 = C[:, P + 1]
    n = len(c0)
    r = np.arange(n)
    M = {}
    lk = en.locked_up(w, r, np.full(n, P + 1))
    M["M0_D+1종가"] = en.Entry("M0", np.where(lk | ~np.isfinite(C1), -1, P + 1), np.where(lk, np.nan, C1), "close")
    o2 = O[:, P + 2]
    M["M1_D+2시가"] = en.Entry("M1", np.where(np.isfinite(o2), P + 2, -1), o2, "open")
    M["M2_D+1종가-5%지정가(5일)"] = en.limit_entry(w, C1 * 0.95, 6, "M2", "seg", start=2)
    M["M3_D+1종가-10%지정가(10일)"] = en.limit_entry(w, C1 * 0.90, 11, "M3", "seg", start=2)
    M["M4_D종가-15%지정가(10일)"] = en.limit_entry(w, c0 * 0.85, 11, "M4", "seg", start=2)
    M["M5_D종가-25%지정가(20일)"] = en.limit_entry(w, c0 * 0.75, 20, "M5", "seg", start=2)
    M["M6_장대봉절반지정가(10일)"] = en.limit_entry(w, p0 + 0.5 * (c0 - p0), 11, "M6", "seg", start=2)
    ma5p = en.prior_ma(C, 5)
    M["M7_5일선지지확인종가(10일)"] = en.signal_entry(w, (L <= ma5p) & (C > ma5p), 11, "M7", "seg", "close", start=2)
    minC = _cummin_after(C, P + 1)
    runH = _cummax_after(H, P)
    sig = (minC <= runH * 0.9) & (C > _prev(H))
    M["M8_고점-10%후전일고가돌파(다음날시가)"] = en.signal_entry(w, sig, 10, "M8", "seg", "open", start=2)
    minL_prev = _prev(_cummin_after(L, P + 1))
    newlow = (L < minL_prev).astype(float)
    sig = (_prev(newlow) == 1) & (V <= 0.6 * _prev(V)) & (L > _prev(L))
    M["M9_신저가후매도소멸(종가)"] = en.signal_entry(w, sig, 11, "M9", "seg", "close", start=3)
    return M


EXITS = {
    "X1_익절15%·손절없음·10일": dict(tp=0.15, sl=np.inf, H=10),
    "X2_익절10%·손절7%·5일": dict(tp=0.10, sl=0.07, H=5),
    "X3_익절20%·손절없음·20일": dict(tp=0.20, sl=np.inf, H=20),
}
