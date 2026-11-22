"""
눌림 확인 후 '상승 전환을 확인하고' 매수하는 방법.

준비(셋업): D+2 ~ D+20 사이에 D 이후 최고가 대비 x% 이상 눌림 (x = 10/15/20/25)
           -> 눌림이 처음 x% 에 닿은 날 이후로 확인 신호를 기다림 (최대 10거래일)
확인 신호 (모두 그날 종가에 진입, 진입 전 정보만 사용)
  C1 전일고가돌파   : 종가 > 전일 고가
  C2 2일연속상승    : 종가 > 전일 종가 > 전전일 종가
  C3 저점대비+5%    : 종가 >= 눌림 최저가 × 1.05
  C4 저점대비+10%   : 종가 >= 눌림 최저가 × 1.10
  C5 거래량양봉     : 양봉(종가>시가, 종가>전일종가) & 거래량 >= 눌림 기간 평균 × 1.5
  C6 5일선재돌파    : 전일 종가 < 전일 5일선, 오늘 종가 > 오늘 5일선
  C7 3일고가돌파    : 종가 > 직전 3일 최고가
  C8 저점높이기     : 오늘 저가 > 전일 저가 > 전전일 저가 & 양봉
비교(확인 없음)    : 눌림이 x% 에 처음 닿은 날 종가에 바로 매수
청산
  E1 : 손절 = 종가가 눌림 최저가(진입 시점까지) 아래 -> 다음날 시가, 익절 20%, 최대 20일
  E2 : 익절 = 급등일 종가의 98% (전고점 매물대 직전), 같은 손절, 최대 20일
  E3 : 익절 15%, 손절 없음, 최대 10일
"""
from __future__ import annotations

import numpy as np

import engine as en
from psych import _cummin_after, _cummax_after, _prev, _run_mean_after

PRE = en.PRE
DEPTHS = (0.10, 0.15, 0.20, 0.25)
WAIT = 10
CONFIRMS = ["C1_전일고가돌파", "C2_2일연속상승", "C3_저점대비+5%", "C4_저점대비+10%", "C5_거래량양봉",
            "C6_5일선재돌파", "C7_3일고가돌파", "C8_저점높이기"]


def build(w: en.Win):
    P = PRE
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    K = w.K
    cols = np.arange(K)
    runH = _cummax_after(H, P)                     # D~d 최고가
    minL = _cummin_after(L, P + 1)                 # D+1~d 최저가
    avgV = _run_mean_after(V, P + 1)               # D+1~d-1 평균 거래량
    ma5 = en.prior_ma(np.hstack([C[:, 1:], np.full((C.shape[0], 1), np.nan)]), 5)  # d일 포함 5일 평균
    Cp, Cpp, Hp, Lp, Lpp = _prev(C), _prev(_prev(C)), _prev(H), _prev(L), _prev(_prev(L))
    ma5p = _prev(ma5)
    hh3 = np.maximum(np.maximum(_prev(H), _prev(_prev(H))), _prev(_prev(_prev(H))))
    sig = {
        "C1_전일고가돌파": C > Hp,
        "C2_2일연속상승": (C > Cp) & (Cp > Cpp),
        "C3_저점대비+5%": C >= minL * 1.05,
        "C4_저점대비+10%": C >= minL * 1.10,
        "C5_거래량양봉": (C > O) & (C > Cp) & (V >= 1.5 * avgV),
        "C6_5일선재돌파": (Cp < ma5p) & (C > ma5),
        "C7_3일고가돌파": C > hh3,
        "C8_저점높이기": (L > Lp) & (Lp > Lpp) & (C > O),
    }
    out = {}
    stops = {}
    for x in DEPTHS:
        hit = (L <= runH * (1 - x)) & (cols[None, :] >= P + 2) & (cols[None, :] <= P + 20)
        first = np.where(hit.any(1), np.argmax(hit, 1), -1)        # 눌림이 처음 x% 에 닿은 날
        has = first >= 0
        f = np.where(has, first, 0)
        # 비교: 그날 종가에 바로 매수
        r = np.arange(len(f))
        px = np.where(has, C[r, f], np.nan)
        nm = f"눌림{int(x*100)}%_확인없음"
        out[nm] = en.Entry(nm, np.where(has & np.isfinite(px), f, -1), px, "close", "confirm")
        stops[nm] = minL[r, f]
        after = (cols[None, :] > f[:, None]) & (cols[None, :] <= f[:, None] + WAIT) & has[:, None] & w.known
        for cn, s in sig.items():
            m = s & after
            j = np.where(m.any(1), np.argmax(m, 1), -1)
            ok = j >= 0
            jj = np.where(ok, j, 0)
            px = np.where(ok, C[r, jj], np.nan)
            ok &= np.isfinite(px) & ~en.locked_up(w, r, jj)
            nm = f"눌림{int(x*100)}%_{cn}"
            out[nm] = en.Entry(nm, np.where(ok, jj, -1), np.where(ok, px, np.nan), "close", "confirm")
            stops[nm] = minL[r, jj]        # 진입 시점까지의 눌림 최저가
    return out, stops


def exits(w: en.Win, stop_level: np.ndarray) -> dict:
    n, K = len(w.c0), w.K
    ce = np.broadcast_to(stop_level[:, None], (n, K))
    return {
        "E1_저점이탈손절·익절20%·20일": dict(tp=0.20, sl=np.inf, H=20, close_exit=ce, tp_px=None),
        "E2_저점이탈손절·D종가직전익절·20일": dict(tp=0.20, sl=np.inf, H=20, close_exit=ce, tp_px=w.c0 * 0.98),
        "E3_익절15%·손절없음·10일": dict(tp=0.15, sl=np.inf, H=10, close_exit=None, tp_px=None),
    }
