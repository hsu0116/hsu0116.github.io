"""
눌림의 '질'로 판단하는 매수 (깊이 기준 아님).

조건 (D+2 ~ D+10 중 처음 만족하는 날 종가 매수, 이벤트당 1회)
  A 저가 높이기   : 오늘 저가 > 전일 저가
  B 거래량 감소   : 오늘 거래량 <= 급등일 거래량 × v  (v = 0.3 / 0.5 / 0.7 / 조건없음)
                    + (선택) 오늘 거래량 <= 직전 3일 평균 × 0.8  (계속 줄어드는 중)
  C 가격 상한     : 오늘 종가 <= 급등일 종가 × k  (k = 0.90 / 1.00 / 1.05 / 조건없음)
                    -> 급등 장대봉 위로 다시 올라간 비싼 가격에는 사지 않음
청산
  S1 지지 이탈 손절: 종가가 진입 전일 저가(지지선) 아래 -> 다음날 시가, 익절 15%, 최대 10일
  S2 지지 이탈 손절: 같은 손절, 익절 20%, 최대 20일
  S3 손절 없음     : 익절 15%, 최대 10일

사전 등록 메인 규칙: A + (B: v=0.5, 계속 감소) + (C: k=1.00) + S1
"""
from __future__ import annotations

from itertools import product

import numpy as np

import engine as en
from psych import _prev

PRE = en.PRE
VOLS = (0.3, 0.5, 0.7, None)
TRENDS = (True, False)
CAPS = (0.90, 1.00, 1.05, None)
HL = (True, False)
MAIN = ("저가높이기", 0.5, True, 1.00)


def name(hl, v, tr, k):
    a = "저가높이기" if hl else "저가무관"
    b = f"거래량≤급등일×{v}" if v is not None else "거래량무관"
    b += "&감소중" if (tr and v is not None) else ""
    c = f"종가≤급등일종가×{k}" if k is not None else "가격무관"
    return f"{a} | {b} | {c}"


def build(w: en.Win, first: int = 2, last: int = 10):
    P = PRE
    C, L, V = w.C, w.L, w.V
    n, K = len(w.c0), w.K
    cols = np.arange(K)
    win = (cols[None, :] >= P + first) & (cols[None, :] <= P + last) & w.known
    hlow = L > _prev(L)
    v3 = (_prev(V) + _prev(_prev(V)) + _prev(_prev(_prev(V)))) / 3
    r = np.arange(n)
    E, meta = {}, {}
    for hl, v, tr, k in product(HL, VOLS, TRENDS, CAPS):
        if v is None and tr:          # 거래량 조건이 없으면 '감소중'도 의미 없음
            continue
        m = win.copy()
        if hl:
            m &= hlow
        if v is not None:
            m &= V <= v * w.v0[:, None]
            if tr:
                m &= V <= 0.8 * v3
        if k is not None:
            m &= C <= k * w.c0[:, None]
        j = np.where(m.any(1), np.argmax(m, 1), -1)
        ok = j >= 0
        jj = np.where(ok, j, 0)
        px = np.where(ok, C[r, jj], np.nan)
        ok &= np.isfinite(px) & ~en.locked_up(w, r, jj)
        nm = name(hl, v, tr, k)
        E[nm] = en.Entry(nm, np.where(ok, jj, -1), np.where(ok, px, np.nan), "close", "quality")
        meta[nm] = dict(hl=hl, v=v, tr=tr, k=k, support=L[r, np.clip(jj - 1, 0, K - 1)])
    return E, meta


def exits(w: en.Win, support: np.ndarray) -> dict:
    n, K = len(w.c0), w.K
    ce = np.broadcast_to(support[:, None], (n, K))
    return {
        "S1_지지이탈손절·익절15%·10일": dict(tp=0.15, sl=np.inf, H=10, close_exit=ce),
        "S2_지지이탈손절·익절20%·20일": dict(tp=0.20, sl=np.inf, H=20, close_exit=ce),
        "S3_손절없음·익절15%·10일": dict(tp=0.15, sl=np.inf, H=10, close_exit=None),
    }
