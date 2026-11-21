"""
투매 소진 정량 점수 (일봉).

대상(투매 후보): 추격매수 실패(A|B) 종목에서 D+3 ~ D+10 중 저가가 D+1 이후 신저가인 날 s.

지표 (클수록 '투매 소진'에 가깝게 방향을 맞춤)
 ① vol_z    : (log V_s - 급등 전 60일 log 거래량 평균) / 표준편차
 ② turnover : V_s / 상장주식수 (KRX 데이터에만 있음, 없으면 제외)
 ③ depth    : (D 이후 최고가 - L_s) / 급등 전 20일 평균 진폭(고가-저가)
 ④ recov    : (C_s - L_s) / (전일종가 - L_s)  장중 하락폭 중 종가까지 되찾은 비율
 ⑤ absorb   : 시가가 전일종가 -3% 이하로 갭하락한 날만 (C_s - O_s) / (전일종가 - O_s), 아니면 결측
 ⑥ digest   : D+2 ~ s 누적 거래량 / (D + D+1 거래량)  물린 물량 소화율
확인 지표 (진입을 늦추는 변형)
 ⑦ fade     : V_{s+1} <= 0.6 V_s 이고 L_{s+1} > L_s  -> s+1 종가 진입
 ⑧ retest   : s+1 ~ s+3 저가가 모두 L_s 위        -> s+3 종가 진입

점수 = 각 지표의 '과거 후보 대비 백분위'(시간순 누적, 미래 정보 없음)의 단순 평균 (0~100).
가중치는 맞추지 않는다(동일 가중).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en
from psych import _cummin_after, _cummax_after, _prev

PRE = en.PRE
COMPONENTS = ["vol_z", "turnover", "depth", "recov", "absorb", "digest"]
LABEL = {"vol_z": "① 거래 폭발도", "turnover": "② 회전율", "depth": "③ 변동성 대비 하락 깊이",
         "recov": "④ 장중 회복률", "absorb": "⑤ 시초 흡수율", "digest": "⑥ 물린 물량 소화율"}


def panel_stats(panel: dict) -> dict:
    """급등 전 기준값 (D 전일까지): 60일 log 거래량 평균/표준편차, 20일 평균 진폭."""
    lv = np.log(panel["volume"].where(panel["volume"] > 0))
    rng = (panel["high"] - panel["low"]) / panel["close"]
    return {
        "lv_mu": lv.rolling(60, min_periods=30).mean().shift(1).values,
        "lv_sd": lv.rolling(60, min_periods=30).std().shift(1).values,
        "atr_pct": rng.rolling(20, min_periods=10).mean().shift(1).values,   # 가격 대비 평균 진폭
    }


def candidates(w: en.Win, panel: dict, fomo: np.ndarray, first: int = 3, last: int = 10) -> pd.DataFrame:
    P = PRE
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    t, i = w.ev["t"].values, w.ev["i"].values
    ps = panel_stats(panel)
    lv_mu, lv_sd, atr = ps["lv_mu"][t, i], ps["lv_sd"][t, i], ps["atr_pct"][t, i] * w.p0
    minL_prev = _prev(_cummin_after(L, P + 1))
    runH = _cummax_after(H, P)
    cols = np.arange(w.K)
    inwin = (cols[None, :] >= P + first) & (cols[None, :] <= P + last)
    cand = (L < minL_prev) & inwin & fomo[:, None] & w.known & np.isfinite(C)
    ei, s = np.nonzero(cand)
    Cp = C[ei, s - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        df = pd.DataFrame({"ev": ei, "col": s, "day": s - P})
        df["date"] = w.dates[np.clip(t[ei] + (s - P), 0, len(w.dates) - 1)]
        df["vol_z"] = (np.log(np.maximum(V[ei, s], 1)) - lv_mu[ei]) / lv_sd[ei]
        if "turnover" in panel:
            tv = panel["turnover"].values
            tt = np.clip(t[ei] + (s - P), 0, tv.shape[0] - 1)
            df["turnover"] = tv[tt, i[ei]]
        else:
            df["turnover"] = np.nan
        df["depth"] = (runH[ei, s] - L[ei, s]) / atr[ei]
        drop = Cp - L[ei, s]
        df["recov"] = np.clip(np.where(drop > 0, (C[ei, s] - L[ei, s]) / drop, np.nan), 0, 1.5)
        gap = O[ei, s] <= Cp * 0.97
        df["absorb"] = np.where(gap, np.clip((C[ei, s] - O[ei, s]) / (Cp - O[ei, s]), -1, 1.5), np.nan)
        cumV = np.cumsum(np.where(cols[None, :] >= P + 2, V, 0), axis=1)
        df["digest"] = cumV[ei, s] / (w.v0[ei] + V[ei, P + 1])
        # 확인 지표
        s1 = np.clip(s + 1, 0, w.K - 1)
        df["fade"] = (V[ei, s1] <= 0.6 * V[ei, s]) & (L[ei, s1] > L[ei, s])
        nxt = np.minimum(np.minimum(L[ei, np.clip(s + 1, 0, w.K - 1)], L[ei, np.clip(s + 2, 0, w.K - 1)]),
                         L[ei, np.clip(s + 3, 0, w.K - 1)])
        df["retest"] = nxt > L[ei, s]
        # 단순 선행 수익 (s 종가 진입)
        for h in (5, 10):
            df[f"r{h}"] = C[ei, np.clip(s + h, 0, w.K - 1)] / C[ei, s] - 1
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_score(df: pd.DataFrame, burn_in: int = 300) -> pd.DataFrame:
    """시간순 누적 백분위(과거+당일 후보 대비) -> 동일가중 평균 점수."""
    df = df.sort_values(["date", "ev"]).reset_index(drop=True)
    used = []
    for c in COMPONENTS:
        if df[c].notna().sum() < burn_in:
            df[f"p_{c}"] = np.nan
            continue
        df[f"p_{c}"] = df[c].expanding().rank(pct=True) * 100
        df.loc[df[c].isna(), f"p_{c}"] = np.nan
        used.append(c)
    pc = [f"p_{c}" for c in used]
    df["score"] = df[pc].mean(axis=1)
    df["burn_in"] = np.arange(len(df)) < burn_in
    df.attrs["used"] = used
    return df


def clustered_mean_se(r: pd.Series, groups: pd.Series):
    r = r.astype(float)
    n = len(r)
    if n < 2:
        return np.nan, np.nan
    m = r.mean()
    g = (r - m).groupby(groups.values).sum()
    se = np.sqrt((g ** 2).sum()) / n
    return m, se


def dose_response(df: pd.DataFrame, col: str, ret: str = "r10", q: int = 5) -> dict:
    x = df[[col, ret, "date"]].dropna()
    if len(x) < q * 20:
        return dict(n=len(x))
    x["q"] = pd.qcut(x[col].rank(method="first"), q, labels=False)
    means = x.groupby("q")[ret].mean()
    top, bot = x[x.q == q - 1], x[x.q == 0]
    mt, st = clustered_mean_se(top[ret], top["date"])
    mb, sb = clustered_mean_se(bot[ret], bot["date"])
    rho = x[col].rank().corr(x[ret].rank())   # 스피어만 순위상관 (scipy 불필요)
    return dict(n=len(x), rho=rho, top=mt, bottom=mb, spread=mt - mb,
                t=(mt - mb) / np.sqrt(st ** 2 + sb ** 2) if st and sb else np.nan,
                **{f"Q{k + 1}": v for k, v in means.items()})


# --------------------------------------------------------------------------- 매매 규칙
def subset_win(w: en.Win, rows: np.ndarray) -> en.Win:
    ev = w.ev.iloc[rows].reset_index(drop=True)
    w2 = en.Win(O=w.O[rows], H=w.H[rows], L=w.L[rows], C=w.C[rows], V=w.V[rows], known=w.known[rows],
                ev=ev, dates=w.dates)
    w2.LIM = w.LIM[rows]
    return w2


def trade_rule(w: en.Win, cand: pd.DataFrame, qualify: np.ndarray, trapped: np.ndarray,
               delay: str = "none", cost: float = 0.003, H: int = 10):
    """이벤트마다 조건을 만족한 첫 후보일에 진입. delay: none(s 종가) / fade(s+1 종가) / retest(s+3 종가)."""
    c = cand[qualify].sort_values(["ev", "col"]).drop_duplicates("ev")
    if delay == "fade":
        c = c[c["fade"]]
        shift = 1
    elif delay == "retest":
        c = c[c["retest"]]
        shift = 3
    else:
        shift = 0
    rows = c["ev"].values
    if len(rows) == 0:
        return None
    w2 = subset_win(w, rows)
    n = len(rows)
    r = np.arange(n)
    ecol = c["col"].values + shift
    px = w2.C[r, np.clip(ecol, 0, w2.K - 1)]
    ok = np.isfinite(px) & (ecol < w2.K)
    entry = en.Entry("capit", np.where(ok, ecol, -1), np.where(ok, px, np.nan), "close", "capit")
    lowS = w2.L[r, c["col"].values]
    ce = np.broadcast_to(lowS[:, None], (n, w2.K))
    tp_px = trapped[rows] * 0.98
    ret, hold, xcol, valid = en.simulate(w2, entry, 0.15, np.inf, H, cost, None, ce, tp_px)
    return dict(w=w2, entry=entry, ret=ret, hold=hold, valid=valid, close_exit=ce, tp_px=tp_px,
                date=w2.ev["date"].values)
