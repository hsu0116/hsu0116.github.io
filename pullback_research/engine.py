"""
급등(종가 +15% 이상) 다음날부터 추적 -> 눌림 매수 전략 백테스트 엔진 (벡터화).

용어
- D      : 급등일 (종가 등락률 >= threshold). D 장 마감 후 알 수 있음.
- c0, p0 : D 종가, D 전일 종가 (수정주가)
- o0,h0,l0,v0 : D 시가/고가/저가/거래량
- 관찰일 d = 1..W : D 다음날부터 W 거래일 동안 진입 조건 감시. 미체결이면 거래 없음.

진입 체결 규칙 (보수적)
- 지정가(limit): 그날 시가가 지정가 이하면 시가 체결(갭하락), 아니면 저가<=지정가일 때 지정가 체결.
- 'open'  : 신호 다음날 시가 체결
- 'close' : 신호 당일 종가 체결(종가 단일가)
청산 규칙 (보수적)
- 같은 날 손절/익절 둘 다 닿으면 손절 우선.
- 지정가로 장중 체결된 당일: 저가가 손절가 이하면 손절로 처리(익절은 다음날부터).
- 시가가 손절가 아래/익절가 위로 갭이면 시가 청산.
- 보유 H 거래일째 종가 청산.
- 비용: 왕복 cost (수수료+거래세+슬리피지) 차감.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

PRE = 20      # 급등일 이전 확보 일수 (이평 계산용)
W_MAX = 20    # 최대 관찰일
H_MAX = 20    # 최대 보유일
INF = np.inf


# =========================================================================== events
def find_events(panel: dict, thr: float = 15.0, min_value: float = 1e9, min_price: float = 1000,
                exclude_pref: bool = True, exclude_spac: bool = True, upper_only: bool = False,
                max_chg: float = 1e9, exclude_anomaly: bool = True) -> pd.DataFrame:
    chg, val, cls = panel["chg"], panel["value"], panel["close"]
    meta = panel["meta"]
    m = (chg >= thr) & (val >= min_value) & (cls >= min_price)
    # 데이터 이상: 가격제한폭(2015-06-15 이전 15%, 이후 30%)을 넘는 하루 변동이 이벤트 전후 창 안에 있으면 제외
    #  (수정주가 미반영 감자·병합, 장기 거래정지 후 재개, 정리매매 등 -> 평균을 크게 왜곡)
    r = (cls / cls.ffill().shift(1) - 1).abs()
    lim = np.where(cls.index < pd.Timestamp("2015-06-15"), 0.16, 0.31)[:, None]
    bad = (r.values > lim).astype(np.int32)
    cb = np.vstack([np.zeros((1, bad.shape[1]), np.int32), np.cumsum(bad, axis=0)])
    T = len(cls)
    tt = np.arange(T)
    lo_, hi_ = np.clip(tt - PRE, 0, T), np.clip(tt + W_MAX + H_MAX + 2, 0, T)
    anom = (cb[hi_] - cb[lo_]) > 0
    if exclude_anomaly:
        m &= ~anom
    if upper_only:
        m &= chg >= 29.5
    m &= chg < max_chg
    cols = cls.columns
    ok = pd.Series(True, index=cols)
    if exclude_pref:  # 우선주: 코드 끝자리 0 아님 (구형 코드 기준)
        ok &= pd.Series([c[-1] == "0" if c.isdigit() else True for c in cols], index=cols)
    if exclude_spac and "name" in meta:
        nm = meta["name"].reindex(cols).fillna("")
        ok &= ~nm.str.contains("스팩|SPAC")
    m &= ok.values[None, :]
    ti, ii = np.nonzero(m.values)
    ev = pd.DataFrame({"t": ti, "i": ii})
    ev["date"] = cls.index[ti]
    ev["code"] = cols[ii]
    ev["chg"] = chg.values[ti, ii]
    ev["value"] = val.values[ti, ii]
    ev["market"] = meta["market"].reindex(cols[ii]).values if "market" in meta else ""
    ev["name"] = meta["name"].reindex(cols[ii]).values if "name" in meta else ""
    # 최근 20일 내 이전 급등 횟수 (첫 급등 vs 연속 급등)
    hit = (chg >= thr).astype(np.int8).values
    cs = np.vstack([np.zeros((1, hit.shape[1]), np.int32), np.cumsum(hit, axis=0)])
    lo = np.clip(ti - 20, 0, None)
    ev["prior20"] = cs[ti, ii] - cs[lo, ii]
    find_events.n_anomaly = int((((chg >= thr) & (val >= min_value)).values & anom).sum())
    return ev.sort_values(["t", "i"]).reset_index(drop=True)


# =========================================================================== windows
@dataclass
class Win:
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    known: np.ndarray       # 데이터 기간 안에 있는 날인지
    ev: pd.DataFrame
    dates: pd.DatetimeIndex
    K: int = field(init=False)

    def __post_init__(self):
        self.K = self.C.shape[1]
        P = PRE
        self.c0 = self.C[:, P]
        self.p0 = self.C[:, P - 1]
        self.o0 = self.O[:, P]
        self.h0 = self.H[:, P]
        self.l0 = self.L[:, P]
        self.v0 = self.V[:, P]
        self.t0 = self.ev["t"].values


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(a.shape[0])[:, None], idx]
    return out


def build_windows(panel: dict, ev: pd.DataFrame) -> Win:
    T = len(panel["close"])
    K = PRE + 1 + W_MAX + 1 + H_MAX
    offs = np.arange(-PRE, K - PRE)
    tt = ev["t"].values[:, None] + offs[None, :]
    known = (tt >= 0) & (tt < T)
    ttc = np.clip(tt, 0, T - 1)
    ii = ev["i"].values[:, None]
    g = {}
    for k in ("open", "high", "low", "close", "volume"):
        a = panel[k].values[ttc, ii].astype(float)
        a[~known] = np.nan
        g[k] = a
    # 상폐/정지로 비는 날은 직전 종가로 채움 (상폐 후엔 마지막 가격 유지 = 정리매매가)
    C = _ffill_rows(g["close"])
    for k in ("open", "high", "low"):
        g[k] = np.where(np.isnan(g[k]), C, g[k])
    V = np.nan_to_num(g["volume"])
    w = Win(O=g["open"], H=g["high"], L=g["low"], C=C, V=V, known=known, ev=ev, dates=panel["close"].index)
    # 데이터 끝 이후는 NaN 유지 (거래 무효 처리용)
    for a in (w.O, w.H, w.L, w.C):
        a[~known] = np.nan
    return w


def prior_ma(C: np.ndarray, n: int) -> np.ndarray:
    """d일에 쓸 수 있는 이평 = d-1일까지의 n일 종가 평균 (shape 동일, 앞부분 NaN)."""
    cs = np.nancumsum(np.nan_to_num(C), axis=1)
    cs = np.hstack([np.zeros((C.shape[0], 1)), cs])
    ma = np.full_like(C, np.nan)
    # ma[:, d] = mean(C[:, d-n : d])
    ma[:, n:] = (cs[:, n:-1] - cs[:, :-n - 1]) / n
    return ma


# =========================================================================== entries
@dataclass
class Entry:
    name: str
    col: np.ndarray     # 진입 열 (-1 = 미진입)
    price: np.ndarray
    kind: str           # 'limit' | 'open' | 'close'
    family: str = ""


def _first(mask: np.ndarray) -> np.ndarray:
    any_ = mask.any(axis=1)
    j = np.argmax(mask, axis=1)
    return np.where(any_, j, -1)


def limit_entry(w: Win, level, W: int, name: str, family: str, start: int = 1) -> Entry:
    P = PRE
    cols = np.arange(P + start, P + W + 1)
    lv = np.asarray(level, float)
    lv = np.broadcast_to(lv[:, None], (len(w.c0), w.K)) if lv.ndim == 1 else lv
    L, O, lvs = w.L[:, cols], w.O[:, cols], lv[:, cols]
    hit = (L <= lvs) & np.isfinite(lvs) & (lvs > 0) & w.known[:, cols]
    j = _first(hit)
    ok = j >= 0
    col = np.where(ok, cols[np.clip(j, 0, None)], -1)
    r = np.arange(len(j))
    px = np.minimum(O[r, np.clip(j, 0, None)], lvs[r, np.clip(j, 0, None)])
    return Entry(name, col, np.where(ok, px, np.nan), "limit", family)


def signal_entry(w: Win, sig: np.ndarray, W: int, name: str, family: str, kind: str, start: int = 1) -> Entry:
    """sig: (E,K) bool. kind='open' -> 다음날 시가, 'close' -> 당일 종가."""
    P = PRE
    cols = np.arange(P + start, P + W + 1)
    j = _first(sig[:, cols] & w.known[:, cols])
    ok = j >= 0
    scol = cols[np.clip(j, 0, None)]
    r = np.arange(len(j))
    if kind == "open":
        ecol = scol + 1
        px = w.O[r, ecol]
    else:
        ecol = scol
        px = w.C[r, ecol]
        ok &= ~locked_up(w, r, ecol)
    ok &= np.isfinite(px)
    return Entry(name, np.where(ok, ecol, -1), np.where(ok, px, np.nan), kind, family)


def locked_up(w: Win, r, col) -> np.ndarray:
    """그날 상한가(전일 대비 +29.5% 이상)로 고가=종가 마감 -> 종가 매수 체결 불가로 간주."""
    c, h, pc = w.C[r, col], w.H[r, col], w.C[r, np.clip(col - 1, 0, None)]
    return (c / pc - 1 >= 0.295) & (c >= h)


def make_entries(w: Win, Ws=(5, 10, 20)) -> list[Entry]:
    P = PRE
    c0, p0, o0, h0, v0 = w.c0, w.p0, w.o0, w.h0, w.v0
    E = []
    n = len(c0)
    r = np.arange(n)

    # --- 0. 기준선 (눌림 없이 그냥 사는 경우)
    lk = locked_up(w, r, np.full(n, P))
    E.append(Entry("BASE_D종가", np.where(lk, -1, P), np.where(lk, np.nan, c0), "close", "base"))
    E.append(Entry("BASE_D+1시가", np.full(n, P + 1), w.O[:, P + 1].copy(), "open", "base"))
    for k in (1, 2, 3, 5, 10):
        E.append(Entry(f"BASE_D+{k}종가", np.full(n, P + k), w.C[:, P + k].copy(), "close", "base"))

    for W in Ws:
        # --- 1. 급등일 종가 대비 고정 % 눌림 지정가
        for x in (3, 5, 7, 10, 15, 20, 25):
            E.append(limit_entry(w, c0 * (1 - x / 100), W, f"PCT_종가-{x}%_W{W}", "pct"))
        # --- 2. 급등 장대봉(전일종가~D종가) 되돌림 비율 지정가 (피보나치)
        for f in (0.75, 0.618, 0.5, 0.382, 0.25, 0.0):
            lv = p0 + f * (c0 - p0)
            E.append(limit_entry(w, lv, W, f"FIB_장대봉{int(f*1000)/10:g}%지점_W{W}", "fib"))
        # --- 3. 급등일 시가 / 몸통 중간 지지
        E.append(limit_entry(w, o0, W, f"CANDLE_D시가_W{W}", "candle"))
        E.append(limit_entry(w, (o0 + c0) / 2, W, f"CANDLE_D몸통중간_W{W}", "candle"))
        # --- 4. 이동평균 터치 (전일까지 이평값으로 지정가)
        for m in (3, 5, 10, 20):
            E.append(limit_entry(w, prior_ma(w.C, m), W, f"MA{m}터치_W{W}", "ma"))

    # 이후 규칙은 신호 기반
    C, H, L, O, V = w.C, w.H, w.L, w.O, w.V
    cols_all = np.arange(w.K)
    after = cols_all[None, :] >= P + 1
    Cd = np.where(after, C, np.inf)
    min_close = np.minimum.accumulate(Cd, axis=1)               # D+1~d 최저 종가
    Hd = np.where(cols_all[None, :] >= P, H, -np.inf)
    run_high = np.maximum.accumulate(Hd, axis=1)                 # D~d 최고가
    H_prev = np.hstack([np.full((n, 1), np.nan), H[:, :-1]])
    C_prev = np.hstack([np.full((n, 1), np.nan), C[:, :-1]])
    for W in Ws:
        # --- 5. 눌림 후 반등 확인 (전일 고가 돌파 종가) -> 다음날 시가
        for x in (5, 10, 15, 20):
            pulled = min_close <= run_high * (1 - x / 100)
            for hold_base in (False, True):
                sig = pulled & (C > H_prev) & after
                if hold_base:
                    sig &= min_close >= p0[:, None]            # 장대봉 시작점(전일종가) 이탈 없음
                tag = "_장대봉하단유지" if hold_base else ""
                E.append(signal_entry(w, sig, W, f"REV_고점-{x}%후_전일고가돌파{tag}_W{W}", "rev", "open"))
            # 반등 양봉 (종가>시가 & 종가>전일종가) -> 당일 종가
            sig = pulled & (C > O) & (C > C_prev) & after
            E.append(signal_entry(w, sig, W, f"REV_고점-{x}%후_양봉종가_W{W}", "rev", "close"))
        # --- 6. 거래량 급감 + 장대봉 절반 지지 -> 당일 종가
        for vr in (0.15, 0.25, 0.4):
            for x in (0, 5, 10):
                sig = (V <= vr * v0[:, None]) & (C >= (p0 + 0.5 * (c0 - p0))[:, None]) & \
                      (C <= c0[:, None] * (1 - x / 100)) & after
                E.append(signal_entry(w, sig, W, f"VOL_거래량{int(vr*100)}%이하_종가-{x}%이하_반지지_W{W}",
                                      "vol", "close"))
        # --- 7. 눌림 + 거래량 급감 + 반등 신호 (복합)
        for x in (5, 10):
            for vr in (0.25, 0.4):
                pulled = min_close <= c0[:, None] * (1 - x / 100)
                vdry = np.minimum.accumulate(np.where(after, V, np.inf), axis=1) <= vr * v0[:, None]
                sig = pulled & vdry & (C > H_prev) & (min_close >= p0[:, None]) & after
                E.append(signal_entry(w, sig, W, f"COMBO_종가-{x}%&거래량{int(vr*100)}%&전일고가돌파_W{W}",
                                      "combo", "open"))
    return E


# =========================================================================== exits
def simulate(w: Win, en: Entry, tp: float, sl: float, H: int, cost: float,
             sl_px: np.ndarray | None = None, close_exit: np.ndarray | None = None,
             tp_px: np.ndarray | None = None):
    """반환: ret, hold(일), exit_col, valid(bool)
    sl_px     : (E,) 종목별 절대 손절가(장중 터치). 진입가보다 낮을 때만 적용, 아니면 sl(%) 사용.
    close_exit: (E,K) 종가가 이 값 아래로 마감하면 다음날 시가 청산 (예: 5일선 종가 이탈).
    """
    n = len(en.col)
    has = en.col >= 0
    e = np.where(has, en.col, 0)
    p = en.price
    tpp, slp = p * (1 + tp), p * (1 - sl)
    if tp_px is not None:   # 종목별 절대 익절가 (진입가보다 높을 때만, 아니면 tp% 사용)
        ut = np.isfinite(tp_px) & (tp_px > p * 1.01)
        tpp = np.where(ut, tp_px, tpp)
    if sl_px is not None:
        use = np.isfinite(sl_px) & (sl_px < p)
        slp = np.where(use, sl_px, slp)   # 구조적 손절가가 있으면 그것을 사용
    done = ~has
    pending = np.zeros(n, bool)
    exit_px = np.full(n, np.nan)
    exit_col = np.full(n, -1)
    valid = has.copy()
    r = np.arange(n)
    for k in range(0, H + 1):
        c = np.clip(e + k, 0, w.K - 1)
        o, h, l, cl = w.O[r, c], w.H[r, c], w.L[r, c], w.C[r, c]
        inrange = (e + k < w.K) & np.isfinite(cl)
        act = ~done
        valid &= ~(act & ~inrange)   # 데이터 끝을 넘어가면 무효
        act &= inrange
        hit = np.zeros(n, bool)
        if k == 0:
            if en.kind != "close":
                if en.kind == "open":
                    hs = act & (l <= slp)
                    ht = act & ~hs & (h >= tpp)
                    exit_px = np.where(hs, slp, np.where(ht, tpp, exit_px))
                else:  # limit
                    hs = act & (l <= slp) & (slp < p)
                    ht = np.zeros(n, bool)
                    exit_px = np.where(hs, slp, exit_px)
                hit = hs | ht
        else:
            pe = act & pending                      # 전일 종가 이탈 -> 오늘 시가 청산
            exit_px = np.where(pe, o, exit_px)
            a2 = act & ~pe
            gs = a2 & (o <= slp)
            gt = a2 & ~gs & (o >= tpp)
            hs = a2 & ~gs & ~gt & (l <= slp)
            ht = a2 & ~gs & ~gt & ~hs & (h >= tpp)
            exit_px = np.where(gs | gt, o, exit_px)
            exit_px = np.where(hs, slp, exit_px)
            exit_px = np.where(ht, tpp, exit_px)
            hit = pe | gs | gt | hs | ht
        if k == H:
            tm = act & ~hit
            exit_px = np.where(tm, cl, exit_px)
            hit = hit | tm
        elif close_exit is not None and (k >= 1 or en.kind != "close"):
            lv = close_exit[r, c]
            pending |= act & ~hit & np.isfinite(lv) & (cl < lv)
        exit_col = np.where(hit & ~done, c, exit_col)
        done |= hit
    valid &= done & np.isfinite(exit_px) & np.isfinite(p)
    ret = exit_px / p - 1 - cost
    hold = exit_col - e
    return ret, hold, exit_col, valid


def wmean(ret: np.ndarray) -> float:
    """상·하위 1% 를 잘라 붙인(윈저화) 평균. 극단값 몇 건에 끌려가지 않는 평균."""
    if len(ret) == 0:
        return np.nan
    if len(ret) < 50:
        return float(np.mean(ret))
    lo, hi = np.quantile(ret, [0.01, 0.99])
    return float(np.clip(ret, lo, hi).mean())


def stats(ret: np.ndarray, hold: np.ndarray, n_events: int) -> dict:
    n = len(ret)
    if n == 0:
        return dict(n=0, fill=0, mean=np.nan, wmean=np.nan, median=np.nan, max=np.nan, win=np.nan, pf=np.nan,
                    t=np.nan, hold=np.nan, per_event=np.nan, per_day=np.nan)
    g, b = ret[ret > 0].sum(), -ret[ret < 0].sum()
    sd = ret.std(ddof=1) if n > 1 else np.nan
    mh = hold.mean()
    return dict(n=n, fill=n / max(n_events, 1), mean=ret.mean(), wmean=wmean(ret), median=np.median(ret),
                max=ret.max(),
                win=(ret > 0).mean(), pf=g / b if b > 0 else np.inf,
                t=ret.mean() / sd * np.sqrt(n) if sd and sd > 0 else np.nan,
                hold=mh, per_event=ret.sum() / max(n_events, 1),
                per_day=ret.mean() / max(mh, 1))


TP_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, INF)
SL_GRID = (0.03, 0.05, 0.07, 0.10, 0.15, INF)
H_GRID = (1, 3, 5, 10, 20)


def basic_exits(tps=TP_GRID, sls=SL_GRID, hs=H_GRID) -> list[dict]:
    return [dict(tp=tp, sl=sl, H=H, xs="") for tp, sl, H in product(tps, sls, hs)]


def run_grid(w: Win, entries: list[Entry], split_date: str, cost: float = 0.003,
             exits: list[dict] | None = None, exit_arrays=None) -> pd.DataFrame:
    """exits: dict(tp, sl, H, xs). xs 는 구조적 청산 이름 -> exit_arrays(xs) 가 (sl_px, close_exit) 반환."""
    exits = exits or basic_exits()
    is_mask = (w.ev["date"] < pd.Timestamp(split_date)).values
    n_is, n_oos = is_mask.sum(), (~is_mask).sum()
    rows = []
    for en in entries:
        for x in exits:
            sp, ce = exit_arrays(x["xs"]) if x["xs"] else (None, None)
            ret, hold, _, valid = simulate(w, en, x["tp"], x["sl"], x["H"], cost, sp, ce)
            row = dict(entry=en.name, family=en.family, kind=en.kind, tp=x["tp"], sl=x["sl"], H=x["H"], xs=x["xs"])
            for tag, m, ne in (("IS", is_mask, n_is), ("OOS", ~is_mask, n_oos)):
                mm = valid & m
                s_ = stats(ret[mm], hold[mm], ne)
                row.update({f"{tag}_{k}": v for k, v in s_.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== portfolio
def portfolio(w: Win, en: Entry, tp, sl, H, cost, slots: int = 10, start=None, end=None, seed: int = 0,
              mask=None, max_new_per_day: int = 0, sl_px=None, close_exit=None, tp_px=None):
    """동시 최대 slots 종목, 진입 시 (실현 자산/slots) 투입. 실현 기준 자산곡선.
    mask: (E,) 이 이벤트만 거래. max_new_per_day: 하루 신규 진입 상한(0=무제한)."""
    ret, hold, xcol, valid = simulate(w, en, tp, sl, H, cost, sl_px, close_exit, tp_px)
    if mask is not None:
        valid = valid & mask
    t_in = w.t0 + (en.col - PRE)
    t_out = w.t0 + (xcol - PRE)
    df = pd.DataFrame({"t_in": t_in, "t_out": t_out, "ret": ret,
                       "kind": en.kind, "chg": w.ev["chg"].values})[valid]
    d = w.dates
    if start:
        df = df[d[df.t_in.values] >= pd.Timestamp(start)]
    if end:
        df = df[d[df.t_in.values] < pd.Timestamp(end)]
    # 같은 날 신호 여럿이면 무작위 순서 (등락률 큰 순 우선은 상한가 쏠림 왜곡을 만든다)
    df["rnd"] = np.random.default_rng(seed).random(len(df))
    df = df.sort_values(["t_in", "rnd"])
    eq = 1.0
    active = []  # (t_out, size, ret)
    curve = []
    taken = 0
    day_cnt = {}
    for t_in, t_out, r in df[["t_in", "t_out", "ret"]].itertuples(index=False):
        still = []
        for a in sorted(active):
            if a[0] < t_in or (a[0] == t_in):
                eq += a[1] * a[2]
                curve.append((a[0], eq))
            else:
                still.append(a)
        active = still
        if len(active) < slots and (not max_new_per_day or day_cnt.get(t_in, 0) < max_new_per_day):
            active.append((t_out, eq / slots, r))
            taken += 1
            day_cnt[t_in] = day_cnt.get(t_in, 0) + 1
    for a in sorted(active):
        eq += a[1] * a[2]
        curve.append((a[0], eq))
    if not curve:
        return dict(trades=0, cagr=np.nan, mdd=np.nan, final=1.0), pd.Series(dtype=float)
    cs = pd.Series([c[1] for c in curve], index=d[[c[0] for c in curve]])
    cs = cs.groupby(level=0).last()
    yrs = max((cs.index[-1] - d[df.t_in.values[0]]).days / 365.25, 1e-9)
    mdd = (cs / cs.cummax() - 1).min()
    return dict(trades=taken, cagr=cs.iloc[-1] ** (1 / yrs) - 1, mdd=mdd, final=cs.iloc[-1]), cs
