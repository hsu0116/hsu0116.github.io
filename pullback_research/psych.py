"""
심리 기반 급등주 눌림목 전략 (가설별 1~3개 변형만, 사전에 정한 통합 전략 포함).

가설 -> 일봉에서 측정하는 흔적
 H1 관심 소멸 + 가격 유지 : 거래량이 D일의 20% 이하로 식었는데 종가가 장대봉 절반 위
 H2 재점화(두 번째 파동)   : 한산해진 뒤(최저 거래량 <= D일 40%) 거래량이 D일의 30/50% 이상으로 다시 붙은 양봉
 H3 투매 소진(항복)        : 고점 대비 -15/-25% 이상 눌린 상태에서 눌림 중 평균의 2배 거래량, 눌림 신저가, 아래꼬리(종가가 일중 범위 위쪽 절반)
 H4 추격매수 실패(FOMO 꼭지): D+1 갭상승(+3%) 출발 후 시가 아래 마감, 또는 장중 상한가 찍고 풀림 -> 후보 제외(게이트)
 H5 본전 심리(처분효과)    : D, D+1 거래량가중 평균가 = 물린 사람들 본전 -> 그 바로 아래(99%)를 익절가로
 H6 기준점 고정(앵커링)    : 급등 전 가격(p0) 바로 위 지정가, 라운드 넘버 바로 위 지정가 / p0 종가 이탈 시 손절
 H7 강제 매도(미수 반대매매): D+3, D+4 시초가가 전일 종가보다 3/5% 이상 갭하락하면 시가 매수
 H8 시장 공포              : 전종목 중간값 지수가 20일선 아래면 진입 보류(게이트)
 H9 테마 군집              : 같은 날 같은 업종 급등주 중 거래대금 1위(대장주)만 (업종 정보 없으면 그날 전체 1위)

사전 등록 통합 전략 (PSY_통합):
  게이트 = FOMO 실패 아님 & 시장 공포 아님
  진입   = 급등 전 가격(p0) 종가 이탈 없이 살아 있는 동안, H2(재점화 r30) 또는 H3(투매 소진 -15%) 중 먼저 나온 신호의 종가
  청산   = 익절 H5(본전가 99%), 손절 p0 종가 이탈 -> 다음날 시가, 최대 20거래일
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en

PRE = en.PRE


def _prev(a):
    return np.hstack([np.full((a.shape[0], 1), np.nan), a[:, :-1]])


def _cummin_after(a, start_col):
    cols = np.arange(a.shape[1])
    return np.minimum.accumulate(np.where(cols[None, :] >= start_col, a, np.inf), axis=1)


def _cummax_after(a, start_col):
    cols = np.arange(a.shape[1])
    return np.maximum.accumulate(np.where(cols[None, :] >= start_col, a, -np.inf), axis=1)


def _run_mean_after(a, start_col):
    """start_col ~ d-1 까지의 평균 (d 열에 기록)."""
    cols = np.arange(a.shape[1])
    x = np.where(cols[None, :] >= start_col, np.nan_to_num(a), 0.0)
    cnt = (cols[None, :] >= start_col).astype(float)
    cs, cc = np.cumsum(x, axis=1), np.cumsum(cnt, axis=1)
    m = cs / np.where(cc > 0, cc, np.nan)
    return _prev(m)


# --------------------------------------------------------------------------- 진입
def make_psych_entries(w: en.Win, W: int = 20) -> dict[str, en.Entry]:
    P = PRE
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    c0, p0, o0, v0 = w.c0, w.p0, w.o0, w.v0
    n, K = len(c0), w.K
    cols = np.arange(K)
    after = cols[None, :] >= P + 1
    mid = (p0 + 0.5 * (c0 - p0))[:, None]
    minC = _cummin_after(C, P + 1)                # D+1~d 최저 종가
    alive = minC >= p0[:, None]                   # 급등 전 가격 종가 이탈 없음
    minV_prev = _prev(_cummin_after(V, P + 1))    # D+1~d-1 최저 거래량
    avgV_prev = _run_mean_after(V, P + 1)         # D+1~d-1 평균 거래량
    runH = _cummax_after(H, P)                    # D~d 최고가
    minL_prev = _prev(_cummin_after(L, P + 1))    # D+1~d-1 최저가
    bull = (C > O) & (C > _prev(C))
    E = {}

    # H1 관심 소멸 + 가격 유지
    sig = (V <= 0.2 * v0[:, None]) & (C >= mid) & after & (cols[None, :] >= P + 3)
    E["PSY_H1_관심소멸_가격유지"] = en.signal_entry(w, sig, W, "PSY_H1_관심소멸_가격유지", "psy", "close")

    # H2 재점화
    pulled = minC <= c0[:, None] * 0.9
    for r in (0.3, 0.5):
        sig = (minV_prev <= 0.4 * v0[:, None]) & (V >= r * v0[:, None]) & bull & pulled & alive & after
        nm = f"PSY_H2_재점화{int(r*100)}%_종가"
        E[nm] = en.signal_entry(w, sig, W, nm, "psy", "close", start=2)
    sig = (minV_prev <= 0.4 * v0[:, None]) & (V >= 0.3 * v0[:, None]) & bull & pulled & alive & after
    E["PSY_H2_재점화30%_익일시가"] = en.signal_entry(w, sig, W, "PSY_H2_재점화30%_익일시가", "psy", "open", start=2)

    # H3 투매 소진
    rng = np.maximum(H - L, 1e-9)
    for dd in (0.15, 0.25):
        sig = (L <= runH * (1 - dd)) & (V >= 2 * avgV_prev) & (L < minL_prev) & ((C - L) / rng >= 0.5) & after
        nm = f"PSY_H3_투매소진-{int(dd*100)}%_종가"
        E[nm] = en.signal_entry(w, sig, W, nm, "psy", "close", start=2)

    # H6 앵커 지정가
    E["PSY_H6_급등전가격+1%_지정가"] = en.limit_entry(w, p0 * 1.01, W, "PSY_H6_급등전가격+1%_지정가", "psy")
    step = 5 * 10 ** (np.floor(np.log10(np.maximum(c0, 1))) - 1)
    rnd = np.floor(c0 * 0.85 / step) * step
    E["PSY_H6_라운드넘버+0.5%_지정가"] = en.limit_entry(w, rnd * 1.005, W, "PSY_H6_라운드넘버+0.5%_지정가", "psy")

    # H7 미수 반대매매 시초 갭하락
    r_ = np.arange(n)
    for day in (3, 4):
        for g in (0.03, 0.05):
            col = P + day
            ok = (O[:, col] <= C[:, col - 1] * (1 - g)) & np.isfinite(O[:, col]) & w.known[:, col]
            nm = f"PSY_H7_D+{day}시초갭-{int(g*100)}%"
            E[nm] = en.Entry(nm, np.where(ok, col, -1), np.where(ok, O[r_, col], np.nan), "open", "psy")

    # 통합 전략: H2(30%) 또는 H3(-15%) 중 먼저 (둘 다 살아있는 상태에서)
    a = E["PSY_H2_재점화30%_종가"]
    sig3 = (L <= runH * 0.85) & (V >= 2 * avgV_prev) & (L < minL_prev) & ((C - L) / rng >= 0.5) & after & alive
    b = en.signal_entry(w, sig3, W, "tmp", "psy", "close", start=2)
    ca = np.where(a.col >= 0, a.col, 10 ** 6)
    cb = np.where(b.col >= 0, b.col, 10 ** 6)
    use_a = ca <= cb
    col = np.where(use_a, a.col, b.col)
    px = np.where(use_a, a.price, b.price)
    E["PSY_통합"] = en.Entry("PSY_통합", np.where(np.minimum(ca, cb) < 10 ** 6, col, -1), px, "close", "psy")
    return E


# --------------------------------------------------------------------------- 게이트(후보 필터)
def gates(w: en.Win, panel: dict, ev: pd.DataFrame, mkt_up20: np.ndarray) -> pd.DataFrame:
    P = PRE
    O1, C1 = w.O[:, P + 1], w.C[:, P + 1]
    c0, p0, h0 = w.c0, w.p0, w.h0
    fomo = ((O1 >= c0 * 1.03) & (C1 < O1)) | ((h0 >= p0 * 1.295) & (c0 < h0 * 0.99))
    g = pd.DataFrame(index=ev.index)
    g["FOMO실패아님"] = ~fomo
    g["시장공포아님"] = mkt_up20[ev["t"].values]
    # 대장주: 같은 날 같은 업종(없으면 전체) 급등주 중 거래대금 1위
    if "sector" in ev:
        sec = ev["sector"].astype(object).where(ev["sector"].fillna("").astype(str).str.len() > 0)
    else:
        sec = pd.Series(np.nan, index=ev.index, dtype=object)
    key = ev["t"].astype(str) + "|" + sec.fillna("ALL")
    rank = ev.groupby(key)["value"].rank(ascending=False, method="first")
    cnt = ev.groupby(key)["value"].transform("size")
    g["대장주"] = rank.values == 1
    g["테마대장(동종2+)"] = (rank.values == 1) & (cnt.values >= 2) & sec.notna().values
    g["통합게이트"] = g["FOMO실패아님"] & g["시장공포아님"]
    return g


# --------------------------------------------------------------------------- 청산
def exit_specs(w: en.Win) -> dict[str, dict]:
    P = PRE
    v0, v1 = w.v0, w.V[:, P + 1]
    tp0 = (w.h0 + w.l0 + w.c0) / 3
    tp1 = (w.H[:, P + 1] + w.L[:, P + 1] + w.C[:, P + 1]) / 3
    trapped = (tp0 * v0 + tp1 * v1) / np.maximum(v0 + v1, 1)       # 물린 사람들 평균 단가
    p0b = np.broadcast_to(w.p0[:, None], (len(w.p0), w.K))
    T = trapped * 0.99
    return {
        "본전익절+급등전가격손절_20일": dict(tp=0.15, sl=np.inf, H=20, tp_px=T, close_exit=p0b),
        "본전익절_손절없음_20일": dict(tp=0.15, sl=np.inf, H=20, tp_px=T),
        "15%익절+급등전가격손절_20일": dict(tp=0.15, sl=np.inf, H=20, close_exit=p0b),
        "15%익절_손절없음_20일": dict(tp=0.15, sl=np.inf, H=20),
        "10%익절+급등전가격손절_10일": dict(tp=0.10, sl=np.inf, H=10, close_exit=p0b),
    }


def run(w, entries: dict, exits: dict, gate_df: pd.DataFrame, split: str, cost: float) -> pd.DataFrame:
    is_m = (w.ev["date"] < pd.Timestamp(split)).values
    gate_sets = {"게이트없음": np.ones(len(is_m), bool), "FOMO실패제외": gate_df["FOMO실패아님"].values,
                 "시장공포제외": gate_df["시장공포아님"].values, "대장주만": gate_df["대장주"].values,
                 "통합게이트": gate_df["통합게이트"].values}
    rows = []
    for en_name, e in entries.items():
        for xn, x in exits.items():
            ret, hold, _, valid = en.simulate(w, e, x["tp"], x["sl"], x["H"], cost, None,
                                              x.get("close_exit"), x.get("tp_px"))
            for gn, gm in gate_sets.items():
                row = dict(entry=en_name, exit=xn, gate=gn)
                for tag, m in (("IS", is_m), ("OOS", ~is_m)):
                    mm = valid & gm & m
                    s = en.stats(ret[mm], hold[mm], int((gm & m).sum()))
                    row.update({f"{tag}_{k}": v for k, v in s.items()})
                rows.append(row)
    return pd.DataFrame(rows)
