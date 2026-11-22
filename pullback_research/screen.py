"""
반등 가능성 높은 급등주 선별 (D+1 장 마감 시점, 그때까지의 정보만 사용).

특성(feature)
  급등일: 등락률, 상한가 여부, 거래대금(log), 평소 대비 거래량 배수, 시가 갭, 윗꼬리·몸통 비율, 가격대(log)
  급등 전 흐름: 직전 5·20·60일 수익률, 20일선 이격도, 52주 최고가 대비 위치, 평소 변동성, 최근 20일 급등 횟수
  다음날(D+1): 등락률, 시가 갭, 윗꼬리, 거래량(급등일 대비), 일중 범위
  시장: 시장 상승국면(전종목 중간값 지수 > 20일선), 시장 5일 수익률, 그날 +15% 급등 종목 수, 코스닥 여부
목표(target) — D+1 종가 매수 기준
  y_bar  : 10거래일 안에 +10% 가 -10% 보다 먼저 오면 1, -10% 가 먼저면 0 (같은 날 둘 다면 0, 둘 다 안 닿으면 제외)
  y_ret10: 10거래일 뒤 종가 수익률
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as en

PRE = en.PRE

FEATURES = ["chg", "upper_limit", "log_value", "vol_mult60", "gap0", "upper_shadow0", "body0", "log_price",
            "ret5_pre", "ret20_pre", "ret60_pre", "disp20_pre", "pos52", "atr_pct", "prior20",
            "r1", "gap1", "upper_shadow1", "v1_ratio", "range1",
            "mkt_up20", "mkt_ret5", "n_surge", "kosdaq"]
LABEL = {
    "chg": "급등일 등락률", "upper_limit": "상한가 마감", "log_value": "급등일 거래대금", "vol_mult60": "평소 대비 거래량 배수",
    "gap0": "급등일 시가 갭", "upper_shadow0": "급등일 윗꼬리", "body0": "급등일 몸통 비율", "log_price": "주가 수준",
    "ret5_pre": "급등 전 5일 수익률", "ret20_pre": "급등 전 20일 수익률", "ret60_pre": "급등 전 60일 수익률",
    "disp20_pre": "급등 전 20일선 이격도", "pos52": "52주 최고가 대비 위치", "atr_pct": "평소 변동성",
    "prior20": "최근 20일 급등 횟수", "r1": "다음날 등락률", "gap1": "다음날 시가 갭", "upper_shadow1": "다음날 윗꼬리",
    "v1_ratio": "다음날 거래량(급등일 대비)", "range1": "다음날 일중 범위", "mkt_up20": "시장 상승국면",
    "mkt_ret5": "시장 5일 수익률", "n_surge": "그날 급등 종목 수", "kosdaq": "코스닥",
}


def market_index(panel):
    C = panel["close"]
    ret = (C / C.ffill().shift(1) - 1).where(lambda x: x.abs() < 0.35)
    return (1 + ret.median(axis=1).fillna(0)).cumprod()


def features(panel: dict, w: en.Win, ev: pd.DataFrame) -> pd.DataFrame:
    P = PRE
    t, i = ev["t"].values, ev["i"].values
    C, H, L, V = panel["close"], panel["high"], panel["low"], panel["volume"]
    Cs = C.ffill()

    def at(df, lag=1):
        a = df.values
        return a[np.clip(t - lag, 0, len(a) - 1), i]
    c0, p0, o0, h0, l0, v0 = w.c0, w.p0, w.o0, w.h0, w.l0, w.v0
    O1, H1, L1, C1, V1 = (a[:, P + 1] for a in (w.O, w.H, w.L, w.C, w.V))
    idx = market_index(panel)
    up20 = (idx > idx.rolling(20).mean()).values
    f = pd.DataFrame(index=ev.index)
    f["chg"] = ev["chg"].values
    f["upper_limit"] = (c0 / p0 - 1 >= w.LIM[:, P] - 0.005).astype(float)
    f["log_value"] = np.log10(np.maximum(ev["value"].values, 1))
    f["vol_mult60"] = np.log10(np.maximum(v0, 1) / np.maximum(at(V.rolling(60, min_periods=20).mean()), 1))
    f["gap0"] = o0 / p0 - 1
    rng0 = np.maximum(h0 - l0, 1e-9)
    f["upper_shadow0"] = (h0 - c0) / rng0
    f["body0"] = (c0 - o0) / rng0
    f["log_price"] = np.log10(np.maximum(c0, 1))
    f["ret5_pre"] = at(Cs) / at(Cs, 6) - 1
    f["ret20_pre"] = at(Cs) / at(Cs, 21) - 1
    f["ret60_pre"] = at(Cs) / at(Cs, 61) - 1
    f["disp20_pre"] = at(Cs) / at(Cs.rolling(20, min_periods=10).mean()) - 1
    f["pos52"] = c0 / at(H.rolling(250, min_periods=120).max()) - 1
    f["atr_pct"] = at(((H - L) / C).rolling(20, min_periods=10).mean())
    f["prior20"] = ev["prior20"].values
    f["r1"] = C1 / c0 - 1
    f["gap1"] = O1 / c0 - 1
    f["upper_shadow1"] = (H1 - np.maximum(O1, C1)) / np.maximum(H1 - L1, 1e-9)
    f["v1_ratio"] = np.log10(np.maximum(V1, 1) / np.maximum(v0, 1))
    f["range1"] = (H1 - L1) / c0
    t1 = np.clip(t + 1, 0, len(idx) - 1)
    f["mkt_up20"] = up20[t1].astype(float)
    f["mkt_ret5"] = idx.values[t1] / idx.values[np.clip(t1 - 5, 0, None)] - 1
    ns = (panel["chg"] >= 15).sum(axis=1).values
    f["n_surge"] = ns[t]
    f["kosdaq"] = (ev["market"].astype(str).str.upper().str.contains("KOSDAQ")).astype(float).values
    return f.replace([np.inf, -np.inf], np.nan)


def targets(w: en.Win, up=0.10, dn=0.10, H=10) -> pd.DataFrame:
    P = PRE
    C1 = w.C[:, P + 1]
    n = len(C1)
    res = np.full(n, np.nan)
    done = np.zeros(n, bool)
    for k in range(1, H + 1):
        c = P + 1 + k
        hi = w.H[:, c] >= C1 * (1 + up)
        lo = w.L[:, c] <= C1 * (1 - dn)
        fin = np.isfinite(w.C[:, c])
        lose = ~done & fin & lo
        win = ~done & fin & hi & ~lo
        res[lose] = 0
        res[win] = 1
        done |= lose | win
    # 10일 안에 어느 쪽에도 닿지 않으면 결측(학습·평가 제외).
    # 이를 실패로 세면 모델이 '반등'이 아니라 '변동성 크기'를 맞히게 됨 (합성 데이터 점검에서 확인)
    return pd.DataFrame({"y_bar": res, "y_ret10": w.C[:, P + 1 + H] / C1 - 1})
