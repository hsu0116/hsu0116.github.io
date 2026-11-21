"""
추격매수 실패 -> 강제매도(미수·신용 반대매매) -> 투매 소진 조합 눌림목 전략.

정의 (일봉, D=급등일, P=창 안 D의 열 위치)
  추격매수 실패 A형 : D+1 시가 >= D종가*1.03, D+1 종가 < D+1 시가, D+1 거래량 >= D 거래량*0.5
  추격매수 실패 B형 : D 고가가 상한가(제한폭-0.5%p) 도달, D 종가 < D 고가*0.97 (풀림)
  추격 평균가       : D+1 (고가+저가+종가)/3
  신용 반대매매선   : 추격 평균가*0.84 (보증금률 40% 가정, 담보비율 140%)  [변형 0.70]
  강제매도 구간     : D+3 ~ D+6  [변형 D+3 ~ D+10]
  투매 소진일 s     : 구간 안에서 거래량 >= (D+2~s-1 평균)*1.5, 저가가 D+1 이후 신저가,
                      저가 <= 신용선, (종가-저가)/(고가-저가) >= 0.5
  진입  : s 종가  [변형: s+1 시가가 s 저가 위에서 시작하면 s+1 시가]
  익절  : 추격 평균가*0.98
  손절  : 종가가 s 저가 아래로 마감 -> 다음날 시가
  시간  : 최대 10거래일
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en
from psych import _cummin_after, _prev, _run_mean_after

PRE = en.PRE
H_DAYS = 10


def setup(w: en.Win) -> dict:
    P = PRE
    O1, H1, L1, C1, V1 = (a[:, P + 1] for a in (w.O, w.H, w.L, w.C, w.V))
    c0, p0, h0, v0 = w.c0, w.p0, w.h0, w.v0
    fomoA = (O1 >= c0 * 1.03) & (C1 < O1) & (V1 >= 0.5 * v0)
    lim0 = w.LIM[:, P]
    fomoB = (h0 >= p0 * (1 + lim0 - 0.005)) & (c0 < h0 * 0.97)   # 상한가 도달 후 풀림 (제한폭 시기별)
    trapped = (H1 + L1 + C1) / 3
    return dict(fomoA=fomoA, fomoB=fomoB, fomo=fomoA | fomoB, trapped=trapped)


def capitulation_signal(w: en.Win, S: dict, line_k: float = 0.84) -> np.ndarray:
    P = PRE
    C, H, L, V = w.C, w.H, w.L, w.V
    avgV = _run_mean_after(V, P + 2)                  # D+2 ~ s-1 평균 거래량
    minL_prev = _prev(_cummin_after(L, P + 1))        # D+1 ~ s-1 최저가
    rng = np.maximum(H - L, 1e-9)
    line = (S["trapped"] * line_k)[:, None]
    return (V >= 1.5 * avgV) & (L < minL_prev) & (L <= line) & ((C - L) / rng >= 0.5)


def _close_entry(w, sig, last_day, name):
    return en.signal_entry(w, sig, last_day, name, "combo", "close", start=3)


def _confirm_entry(w, sig, last_day, name):
    e = en.signal_entry(w, sig, last_day, name, "combo", "open", start=3)
    r = np.arange(len(e.col))
    c = np.clip(e.col, 1, w.K - 1)
    ok = (e.col >= 0) & (w.O[r, c] > w.L[r, c - 1])
    return en.Entry(name, np.where(ok, e.col, -1), np.where(ok, e.price, np.nan), "open", "combo")


def stop_at_signal_low(w: en.Win, e: en.Entry) -> np.ndarray:
    """신호일(투매일) 저가를 종가로 이탈하면 청산하는 수준 (E,K)."""
    r = np.arange(len(e.col))
    s = np.where(e.kind == "open", e.col - 1, e.col)
    lv = np.where(e.col >= 0, w.L[r, np.clip(s, 0, w.K - 1)], np.nan)
    return np.broadcast_to(lv[:, None], (len(lv), w.K))


def strategies(w: en.Win, S: dict) -> list[dict]:
    """각 전략: name, entry, mask(이벤트 선택), exit(dict)."""
    P = PRE
    n = len(w.c0)
    allm = np.ones(n, bool)
    tp_px = S["trapped"] * 0.98
    out = []

    def add(name, entry, mask, stop="signal_low", tp=0.15, H=H_DAYS, tp_trapped=True, group="메인/변형"):
        if stop == "signal_low":
            ce = stop_at_signal_low(w, entry)
        elif stop == "pct10":
            ce = np.broadcast_to((entry.price * 0.9)[:, None], (n, w.K))
        elif stop == "entry_low":
            r = np.arange(n)
            lv = np.where(entry.col >= 0, w.L[r, np.clip(entry.col, 0, w.K - 1)], np.nan)
            ce = np.broadcast_to(lv[:, None], (n, w.K))
        else:
            ce = None
        out.append(dict(name=name, entry=entry, mask=mask, group=group,
                        exit=dict(tp=tp, sl=np.inf, H=H, close_exit=ce, tp_px=tp_px if tp_trapped else None)))

    sig84 = capitulation_signal(w, S, 0.84)
    sig70 = capitulation_signal(w, S, 0.70)
    main_e = _close_entry(w, sig84, 6, "투매소진_D3~6_종가")
    add("메인: 추격실패(A|B)+강제매도구간+투매소진", main_e, S["fomo"])
    add("변형1: 추격실패 A형만", main_e, S["fomoA"])
    add("변형2: 추격실패 B형만", main_e, S["fomoB"])
    add("변형3: 구간 D+3~D+10", _close_entry(w, sig84, 10, "투매소진_D3~10_종가"), S["fomo"])
    add("변형4: 다음날 시가 확인 진입", _confirm_entry(w, sig84, 6, "투매소진_확인"), S["fomo"])
    add("변형5: 신용선 0.70", _close_entry(w, sig70, 6, "투매소진70_종가"), S["fomo"])
    # 비교군
    add("비교①a: 투매소진만(전체 급등)", main_e, allm, group="비교군")
    add("비교①b: 투매소진, 추격실패 아닌 종목", main_e, ~S["fomo"], group="비교군")
    lim = en.limit_entry(w, S["trapped"] * 0.84, 6, "신용선지정가", "combo", start=3)
    add("비교②: 추격실패+신용선 지정가(투매확인 없음)", lim, S["fomo"], stop="pct10", group="비교군")
    r = np.arange(n)
    g3 = (w.O[:, P + 3] <= w.C[:, P + 2] * 0.97) & np.isfinite(w.O[:, P + 3])
    g4 = (w.O[:, P + 4] <= w.C[:, P + 3] * 0.97) & np.isfinite(w.O[:, P + 4]) & ~g3
    col = np.where(g3, P + 3, np.where(g4, P + 4, -1))
    px = np.where(col >= 0, w.O[r, np.clip(col, 0, w.K - 1)], np.nan)
    gap = en.Entry("반대매매시초갭", col, px, "open", "combo")
    add("비교③: 추격실패+D+3·4 시초 -3% 갭하락 매수", gap, S["fomo"], stop="entry_low", group="비교군")
    bench = en.limit_entry(w, w.c0 * 0.75, 20, "종가-25%지정가", "bench")
    out.append(dict(name="비교④: 종가-25% 지정가(기존 최선)", entry=bench, mask=None, group="비교군",
                    exit=dict(tp=0.15, sl=np.inf, H=20, close_exit=None, tp_px=None)))
    return out


def evaluate(w: en.Win, strats: list[dict], periods: dict, cost: float, mkt_up20: np.ndarray) -> pd.DataFrame:
    """periods: {이름: (E,) bool 이벤트 마스크}"""
    rows = []
    for st in strats:
        x = st["exit"]
        mask = st["mask"] if st["mask"] is not None else mkt_up20[w.ev["t"].values]
        ret, hold, _, valid = en.simulate(w, st["entry"], x["tp"], x["sl"], x["H"], cost, None,
                                          x["close_exit"], x["tp_px"])
        st["_ret"], st["_valid"], st["_mask"] = ret, valid, mask
        row = dict(전략=st["name"], 구분=st["group"])
        for tag, m in periods.items():
            mm = valid & mask & m
            s = en.stats(ret[mm], hold[mm], int((mask & m).sum()))
            row.update({f"{tag}_{k}": v for k, v in s.items()})
        rows.append(row)
    return pd.DataFrame(rows)
