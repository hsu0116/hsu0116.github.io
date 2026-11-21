"""
유튜브·커뮤니티에서 흔히 소개되는 급등주 눌림 기법 + 질적(재료/수급/추세) 필터.

지표 기법 (진입 규칙, 계열 'yt')
- 엔벨로프(20, 10/15/20%) 하단, 엔벨로프(10, 10%) 하단 지정가
- 볼린저밴드(20, 2) 하단 지정가 / 하단 이탈 후 밴드 안 복귀 종가 / 중심선(20일선) 지지 확인 종가
- RSI(14) 30·40·50 이하 종가, RSI 30·40·50 상향 돌파 종가
- 스토캐스틱 슬로우(5,3,3) 25 이하 골든크로스 종가
- MACD(12,26,9) 오실레이터 하락→상승 전환 (고점 대비 -10% 이상 눌린 뒤) 종가
- 5일선·10일선 장중 터치 후 위에서 마감(지지 확인) 종가
- 기준봉(급등 장대양봉) 절반 가격 터치 후 위에서 마감 종가
- 수급 1음봉 / 2음봉: 거래량 줄어든 음봉 종가
- 3일차 매매: D+1, D+2 연속 하락 후 D+3 시가

구조적 청산 (유튜브식 손절)
- 기준봉 시가 종가 이탈, 기준봉 저가 장중 이탈, 5일선 종가 이탈, 20일선 종가 이탈

질적 필터 (D일 장 마감 시점에 알 수 있는 정보만 사용)
- 52주 신고가, 거래대금 300억/1000억 이상, 상한가 마감, 첫 급등(20일 내 재급등 아님),
  정배열(5>20>60일선), 20·60일선 우하향, 갭상승 출발(+5%), 고가권 마감(윗꼬리 짧음),
  거래량 폭증(20일 평균 10배), 저가주(2천원 미만), 시장 상승국면(전종목 중간값 지수 > 20일선),
  테마 장세(같은 날 +15% 종목 10개 이상)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en

PRE = en.PRE


# --------------------------------------------------------------------------- 패널 지표
def compute_panels(panel: dict) -> dict:
    C = panel["close"].astype("float32")
    H = panel["high"].astype("float32")
    L = panel["low"].astype("float32")
    V = panel["volume"].astype("float32")
    out = {}
    for n in (5, 10, 20, 60):
        out[f"ma{n}"] = C.rolling(n, min_periods=n).mean()
    out["sd20"] = C.rolling(20, min_periods=20).std()
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ll, hh = L.rolling(5).min(), H.rolling(5).max()
    fastk = (C - ll) / (hh - ll).replace(0, np.nan) * 100
    out["slowk"] = fastk.rolling(3).mean()
    out["slowd"] = out["slowk"].rolling(3).mean()
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    out["osc"] = macd - macd.ewm(span=9, adjust=False).mean()
    out["hh250"] = H.rolling(250, min_periods=120).max().shift(1)       # 전일까지 52주 최고가
    out["vavg20"] = V.rolling(20, min_periods=10).mean().shift(1)        # 전일까지 20일 평균 거래량
    out["ma20_5ago"] = out["ma20"].shift(5)
    out["ma60_5ago"] = out["ma60"].shift(5)
    # 시장 국면: 전종목 일간수익률 중간값으로 만든 동일가중 지수
    ret = (C / C.ffill().shift(1) - 1).where(lambda x: x.abs() < 0.35)
    idx = (1 + ret.median(axis=1).fillna(0)).cumprod()
    out["mkt_up20"] = (idx > idx.rolling(20).mean()).values
    out["mkt_up60"] = (idx > idx.rolling(60).mean()).values
    out["n_surge"] = (panel["chg"] >= 15).sum(axis=1).values
    return {k: (v.values.astype("float32") if isinstance(v, pd.DataFrame) else v) for k, v in out.items()}


def gather(arr: np.ndarray, ev: pd.DataFrame, K: int) -> np.ndarray:
    T = arr.shape[0]
    offs = np.arange(-PRE, K - PRE)
    tt = ev["t"].values[:, None] + offs[None, :]
    ok = (tt >= 0) & (tt < T)
    a = arr[np.clip(tt, 0, T - 1), ev["i"].values[:, None]].astype(float)
    a[~ok] = np.nan
    return a


def prev(a: np.ndarray) -> np.ndarray:
    return np.hstack([np.full((a.shape[0], 1), np.nan), a[:, :-1]])


# --------------------------------------------------------------------------- 이벤트 특성 (질적 필터)
def event_features(panel: dict, G: dict, ev: pd.DataFrame) -> pd.DataFrame:
    t, i = ev["t"].values, ev["i"].values
    at = lambda k: G[k][t, i]
    c0, o0, h0, l0 = (panel[k].values[t, i] for k in ("close", "open", "high", "low"))
    v0 = panel["volume"].values[t, i]
    p0 = c0 / (1 + ev["chg"].values / 100)
    f = pd.DataFrame(index=ev.index)
    f["52주신고가"] = c0 >= at("hh250")
    f["거래대금300억+"] = ev["value"].values >= 3e10
    f["거래대금1000억+"] = ev["value"].values >= 1e11
    f["상한가마감"] = ev["chg"].values >= 29.5
    f["첫급등"] = ev["prior20"].values == 0
    f["정배열"] = (at("ma5") > at("ma20")) & (at("ma20") > at("ma60"))
    f["20·60선우하향"] = (at("ma20") < at("ma20_5ago")) & (at("ma60") < at("ma60_5ago"))
    f["갭상승출발+5%"] = o0 / p0 - 1 >= 0.05
    f["고가권마감"] = (h0 - c0) / np.maximum(h0 - l0, 1e-9) <= 0.2
    f["거래량10배+"] = v0 >= 10 * at("vavg20")
    f["저가주<2천원"] = c0 < 2000
    f["시장상승국면20"] = G["mkt_up20"][t]
    f["시장상승국면60"] = G["mkt_up60"][t]
    f["테마장세(동시급등10+)"] = G["n_surge"][t] >= 10
    return f


# --------------------------------------------------------------------------- 유튜브식 진입 규칙
def make_yt_entries(w: en.Win, X: dict, Ws=(10, 20)) -> list[en.Entry]:
    """X: gather 된 지표 (E,K)."""
    P = PRE
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    c0, p0, v0 = w.c0, w.p0, w.v0
    n = len(c0)
    cols = np.arange(w.K)
    after = cols[None, :] >= P + 1
    C_prev = prev(C)
    E = []
    lower = X["ma20"] - 2 * X["sd20"]
    run_high = np.maximum.accumulate(np.where(cols[None, :] >= P, H, -np.inf), axis=1)
    min_close = np.minimum.accumulate(np.where(after, C, np.inf), axis=1)
    rsi, rsi_p = X["rsi"], prev(X["rsi"])
    k, d = X["slowk"], X["slowd"]
    osc = X["osc"]
    bear = C < O
    mid = (p0 + 0.5 * (w.c0 - p0))[:, None]
    for W in Ws:
        tag = f"_W{W}"
        for x in (10, 15, 20):
            E.append(en.limit_entry(w, prev(X["ma20"]) * (1 - x / 100), W, f"YT_엔벨로프20-{x}%{tag}", "yt"))
        E.append(en.limit_entry(w, prev(X["ma10"]) * 0.9, W, f"YT_엔벨로프10-10%{tag}", "yt"))
        E.append(en.limit_entry(w, prev(lower), W, f"YT_볼린저하단{tag}", "yt"))
        sig = (prev(C) < prev(lower)) & (C > lower) & after
        E.append(en.signal_entry(w, sig, W, f"YT_볼린저하단이탈후복귀{tag}", "yt", "close"))
        sig = (L <= X["ma20"]) & (C > X["ma20"]) & after
        E.append(en.signal_entry(w, sig, W, f"YT_볼린저중심선지지{tag}", "yt", "close"))
        for lv in (30, 40, 50):
            E.append(en.signal_entry(w, (rsi <= lv) & after, W, f"YT_RSI{lv}이하{tag}", "yt", "close"))
            E.append(en.signal_entry(w, (rsi_p < lv) & (rsi >= lv) & after, W,
                                     f"YT_RSI{lv}상향돌파{tag}", "yt", "close"))
        sig = (prev(k) <= prev(d)) & (k > d) & (k <= 25) & after
        E.append(en.signal_entry(w, sig, W, f"YT_스토캐스틱25이하골든{tag}", "yt", "close"))
        sig = (min_close <= run_high * 0.9) & (osc > prev(osc)) & (prev(osc) < prev(prev(osc))) & after
        E.append(en.signal_entry(w, sig, W, f"YT_MACD오실상승전환{tag}", "yt", "close"))
        for m in (5, 10):
            lv = prev(X[f"ma{m}"])
            sig = (L <= lv) & (C > lv) & after
            E.append(en.signal_entry(w, sig, W, f"YT_{m}일선지지확인{tag}", "yt", "close"))
        sig = (L <= mid) & (C >= mid) & after
        E.append(en.signal_entry(w, sig, W, f"YT_기준봉절반지지확인{tag}", "yt", "close"))
        sig = bear & (V < 0.5 * v0[:, None]) & after
        E.append(en.signal_entry(w, sig, W, f"YT_수급1음봉{tag}", "yt", "close"))
        sig = bear & (prev(bear.astype(float)) == 1) & (V < 0.5 * v0[:, None]) & (cols[None, :] >= P + 2)
        E.append(en.signal_entry(w, sig, W, f"YT_수급2음봉{tag}", "yt", "close"))
    # 3일차: D+1, D+2 연속 하락 -> D+3 시가 (신호일 D+2 고정)
    sig = (C < C_prev) & (prev(C) < prev(C_prev)) & (cols[None, :] == P + 2)
    E.append(en.signal_entry(w, sig, 2, "YT_3일차(2일연속하락후)", "yt", "open", start=2))
    return E


# --------------------------------------------------------------------------- 구조적 청산
STRUCT = ("기준봉시가종가이탈", "기준봉저가이탈", "5일선종가이탈", "20일선종가이탈")


def struct_exits() -> list[dict]:
    return [dict(tp=tp, sl=np.inf, H=H, xs=xs)
            for tp in (0.10, 0.15, 0.20, np.inf) for xs in STRUCT for H in (10, 20)]


def exit_arrays_factory(w: en.Win, X: dict):
    K = w.K
    o0b = np.broadcast_to(w.o0[:, None], (len(w.o0), K))

    def f(xs: str):
        if xs == "기준봉시가종가이탈":
            return None, o0b
        if xs == "기준봉저가이탈":
            return w.l0, None
        if xs == "5일선종가이탈":
            return None, X["ma5"]
        if xs == "20일선종가이탈":
            return None, X["ma20"]
        raise KeyError(xs)
    return f


# --------------------------------------------------------------------------- 필터 연구
def filter_study(w: en.Win, emap: dict, picks: pd.DataFrame, feats: pd.DataFrame, split: str,
                 cost: float, exit_arrays, min_n: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    """picks: entry,tp,sl,H,xs 행들. 각 전략 × 각 필터(참/거짓) 성과, 그리고 IS 로 고른 2개 필터 조합의 OOS."""
    is_m = (w.ev["date"] < pd.Timestamp(split)).values
    rows, combo = [], []
    for r in picks.itertuples():
        e = emap[r.entry]
        sp, ce = exit_arrays(r.xs) if isinstance(r.xs, str) and r.xs else (None, None)
        ret, hold, _, valid = en.simulate(w, e, r.tp, r.sl, int(r.H), cost, sp, ce)
        base = dict(entry=r.entry, tp=r.tp, sl=r.sl, H=r.H, xs=r.xs)

        def st(m):  # 평균은 윈저화 평균(상하위 1% 절단) 사용
            a, b = valid & m & is_m, valid & m & ~is_m
            return (a.sum(), en.wmean(ret[a]), b.sum(), en.wmean(ret[b]),
                    (ret[b] > 0).mean() if b.any() else np.nan)
        n0, m0, n1, m1, w1 = st(np.ones(len(ret), bool))
        rows.append(dict(**base, filter="(전체)", value=True, IS_n=n0, IS_mean=m0, OOS_n=n1, OOS_mean=m1, OOS_win=w1))
        gains = []
        for col in feats.columns:
            for val in (True, False):
                m = (feats[col].values == val)
                a = st(m)
                rows.append(dict(**base, filter=col, value=val, IS_n=a[0], IS_mean=a[1], OOS_n=a[2], OOS_mean=a[3],
                                 OOS_win=a[4]))
                if a[0] >= min_n and np.isfinite(a[1]):
                    gains.append((a[1] - m0, col, val))
        gains.sort(reverse=True)
        chosen, m = [], np.ones(len(ret), bool)
        for g, col, val in gains:
            if col in [c for c, _ in chosen] or g <= 0:
                continue
            m2 = m & (feats[col].values == val)
            if (valid & m2 & is_m).sum() < min_n:
                continue
            chosen.append((col, val))
            m = m2
            if len(chosen) == 2:
                break
        a = st(m)
        combo.append(dict(**base, filters=" & ".join(f"{c}={'예' if v else '아니오'}" for c, v in chosen) or "(없음)",
                          IS_n=a[0], IS_mean=a[1], OOS_n=a[2], OOS_mean=a[3], OOS_win=a[4],
                          base_IS=m0, base_OOS=m1))
    return pd.DataFrame(rows), pd.DataFrame(combo)
