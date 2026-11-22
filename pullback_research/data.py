"""
데이터 로더: (날짜 x 종목) 패널을 만든다.

panel = dict(
    open, high, low, close, volume, value, chg   -> DataFrame (index=날짜, columns=종목코드)
    meta                                          -> DataFrame (index=종목코드, columns=[name, market])
)
- 가격은 수정주가(액면분할/병합/감자 등 보정)여야 한다.
- chg 는 '전일 대비 등락률(%)'. KRX 원천이면 기준가 기준이라 권리락 등이 반영된 정확한 값.

지원 소스
1) KRX 전종목 일별 스냅샷 (pykrx, KRX 로그인 필요: 환경변수 KRX_ID / KRX_PW)
   - 상장폐지 종목까지 포함 -> 생존편향 없음 (권장)
2) 네이버 (FinanceDataReader, 종목별 일봉)
   - 현재 상장 종목만 -> 생존편향 있음 (상폐 종목 누락 => 결과가 낙관적으로 나옴)
3) CSV 폴더: 종목별 CSV (Date,Open,High,Low,Close,Volume[,Value]) - 수정주가
4) synthetic: 엔진 검증용 랜덤워크 (엣지 0 이어야 정상)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "value", "chg"]


# --------------------------------------------------------------------------- utils
def _finish(panel: dict) -> dict:
    """거래정지일(거래량 0, 시가 0) 처리 + 정렬."""
    c = panel["close"]
    for f in ("open", "high", "low"):
        x = panel[f]
        bad = (x <= 0) | x.isna()
        panel[f] = x.mask(bad & c.notna(), c)
    for f in FIELDS:
        panel[f] = panel[f].sort_index().astype("float32")
    return panel


def adjust_from_krx_raw(panel: dict) -> dict:
    """KRX 원주가 + 등락률(기준가 기준) -> 수정주가 (역방향 누적 보정)."""
    c = panel["close"]
    prev = c.ffill().shift(1)
    base = c / (1.0 + panel["chg"] / 100.0)
    f = (base / prev).where(lambda s: (s - 1).abs() > 0.005, 1.0).fillna(1.0)
    # t 시점 가격에 곱할 배수 = prod_{s>t} f_s
    mult = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    for k in ("open", "high", "low", "close"):
        panel[k] = panel[k] * mult
    panel["volume"] = panel["volume"] / mult
    return panel


# --------------------------------------------------------------------------- 1) KRX
def load_krx(start: str, end: str, cache_dir: str = "cache_krx", sleep: float = 0.4) -> dict:
    """KRX 전종목 시세(MDCSTAT01501)를 날짜별로 받아 캐시. 상장폐지 종목 포함.
    KRX 데이터포털 계정 필요: 환경변수 KRX_ID / KRX_PW. 중간에 끊겨도 다시 실행하면 이어서 받음.
    메모리 절약을 위해 float32 배열로 직접 조립."""
    from pykrx import stock
    from pykrx.website.krx.market.core import 전종목시세

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    days = stock.get_index_ohlcv(start, end, "1001").index  # KOSPI 지수로 영업일 확보
    mk_ok = {"KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"}
    files = []
    for n, d in enumerate(days, 1):
        ds = d.strftime("%Y%m%d")
        fp = cache / f"{ds}.csv.gz"
        if not fp.exists():
            for attempt in range(4):
                try:
                    df = 전종목시세().fetch(ds, "ALL")
                    break
                except Exception as e:  # noqa: BLE001
                    print("retry", ds, e)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"{ds} 수집 실패 (KRX 로그인/접속 확인)")
            df.to_csv(fp, index=False)
            time.sleep(sleep)
        files.append((d, fp))
        if n % 250 == 0:
            print(f"KRX 일별 시세 {n}/{len(days)}일")
    # 1차: 종목 목록
    names, markets = {}, {}
    for d, fp in files:
        df = pd.read_csv(fp, dtype={"ISU_SRT_CD": str}, usecols=["ISU_SRT_CD", "ISU_ABBRV", "MKT_NM"])
        df = df[df["MKT_NM"].isin(mk_ok)]
        names.update(zip(df["ISU_SRT_CD"], df["ISU_ABBRV"]))
        markets.update(zip(df["ISU_SRT_CD"], df["MKT_NM"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})))
    codes = sorted(names)
    pos = {c: i for i, c in enumerate(codes)}
    T, N = len(files), len(codes)
    col = dict(open="TDD_OPNPRC", high="TDD_HGPRC", low="TDD_LWPRC", close="TDD_CLSPRC",
               volume="ACC_TRDVOL", value="ACC_TRDVAL", chg="FLUC_RT", shares="LIST_SHRS")
    arr = {k: np.full((T, N), np.nan, dtype="float32") for k in col}
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    # 2차: 값 채우기
    for t, (d, fp) in enumerate(files):
        df = pd.read_csv(fp, dtype={"ISU_SRT_CD": str}, usecols=["ISU_SRT_CD", "MKT_NM"] + list(col.values()))
        df = df[df["MKT_NM"].isin(mk_ok)]
        idx = df["ISU_SRT_CD"].map(pos).values
        for k, v in col.items():
            arr[k][t, idx] = num(df[v]).values.astype("float32")
    dates = pd.DatetimeIndex([d for d, _ in files])
    # 회전율 = 거래량 / 상장주식수 (같은 날 원자료끼리 나눠 액면분할 등에 영향 없음)
    with np.errstate(divide="ignore", invalid="ignore"):
        arr["turnover"] = np.where(arr["shares"] > 0, arr["volume"] / arr["shares"], np.nan).astype("float32")
    del arr["shares"]
    panel = {k: pd.DataFrame(v, index=dates, columns=codes) for k, v in arr.items()}
    panel["meta"] = pd.DataFrame({"name": pd.Series(names), "market": pd.Series(markets)}).loc[codes]
    return _finish(adjust_from_krx_raw(panel))


# --------------------------------------------------------------------------- 2) Naver
def load_naver(start: str, end: str, cache_dir: str = "cache_naver", sleep: float = 0.15) -> dict:
    """FinanceDataReader 로 현재 상장 종목 일봉(수정주가) 수집. 생존편향 주의."""
    import FinanceDataReader as fdr

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    lst_fp = cache / "_listing.csv"
    if lst_fp.exists():
        lst = pd.read_csv(lst_fp, dtype={"Code": str})
    else:
        lst = fdr.StockListing("KRX")
        lst.to_csv(lst_fp, index=False)
    lst = lst[lst["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    data = {}
    for code in lst["Code"]:
        fp = cache / f"{code}.csv.gz"
        if fp.exists():
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
        else:
            try:
                df = fdr.DataReader(code, start, end)
            except Exception as e:  # noqa: BLE001
                print("fail", code, e)
                continue
            df.to_csv(fp)
            time.sleep(sleep)
        if len(df):
            data[code] = df
    panel = {}
    for k, v in dict(open="Open", high="High", low="Low", close="Close", volume="Volume").items():
        panel[k] = pd.DataFrame({c: d[v] for c, d in data.items()})
    panel["value"] = panel["close"] * panel["volume"]  # 근사 거래대금
    prev = panel["close"].ffill().shift(1)
    panel["chg"] = (panel["close"] / prev - 1) * 100
    meta = lst.set_index("Code")[["Name", "Market"]].rename(columns={"Name": "name", "Market": "market"})
    meta["market"] = meta["market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})
    panel["meta"] = meta.loc[meta.index.intersection(panel["close"].columns)]
    return _finish(panel)


# --------------------------------------------------------------------------- 3) CSV
def load_csv_dir(path: str, meta_csv: str | None = None) -> dict:
    data = {}
    for fp in Path(path).glob("*.csv*"):
        if fp.name.startswith("_"):
            continue
        code = fp.name.split(".")[0]
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        data[code] = df
    panel = {k: pd.DataFrame({c: d[k] for c, d in data.items()})
             for k in ("open", "high", "low", "close", "volume")}
    panel["value"] = (pd.DataFrame({c: d["value"] for c, d in data.items()})
                      if all("value" in d for d in data.values()) else panel["close"] * panel["volume"])
    prev = panel["close"].ffill().shift(1)
    panel["chg"] = (panel["close"] / prev - 1) * 100
    if meta_csv:
        mt = pd.read_csv(meta_csv, dtype={"code": str}).set_index("code")
        panel["meta"] = mt[[c for c in ("name", "market", "sector") if c in mt.columns]]
    else:
        panel["meta"] = pd.DataFrame({"name": list(data), "market": "UNKNOWN"}, index=list(data))
    return _finish(panel)


# --------------------------------------------------------------------------- 4) synthetic
def make_synthetic(n_days: int = 2000, n_stocks: int = 1000, seed: int = 0, sub: int = 12) -> dict:
    """장중 경로(sub 스텝)를 가진 마팅게일 랜덤워크 + 점프.
    시가/고가/저가/종가가 하나의 연속 경로에서 나오므로, 어떤 진입/청산 규칙도
    비용 0 기준 기대수익이 0 이하(보수적 체결 가정 때문에 약간 음수)여야 정상이다."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.015, 0.05, n_stocks)
    sv = vol / np.sqrt(sub)
    O = np.empty((n_days, n_stocks)); H = np.empty_like(O); L = np.empty_like(O); C = np.empty_like(O)
    jump = rng.random((n_days, n_stocks)) < 0.004
    jsize = rng.uniform(0.12, 0.28, (n_days, n_stocks))
    x = np.full(n_stocks, np.log(10000.0))
    for t in range(n_days):
        # 오버나잇 갭
        x = x + rng.standard_t(4, n_stocks) * sv * 1.5 / np.sqrt(2) - 0.5 * (sv * 1.5) ** 2
        steps = rng.standard_t(4, (sub, n_stocks)) * sv / np.sqrt(2) - 0.5 * sv ** 2
        js = rng.integers(0, sub, n_stocks)
        jm = np.zeros((sub, n_stocks))
        jm[js, np.arange(n_stocks)] = np.where(jump[t], np.log1p(jsize[t]), 0.0)
        # 점프는 기대값 0 유지 위해 비점프일 드리프트로 보정하지 않음(사후 경로는 독립 => 점프 이후 엣지 0)
        path = x + np.cumsum(steps + jm, axis=0)
        path = np.vstack([x[None, :], path])
        px = np.exp(path)
        O[t], C[t], H[t], L[t] = px[0], px[-1], px.max(0), px.min(0)
        x = path[-1]
    prevC = np.vstack([C[:1], C[:-1]])
    chg = (C / prevC - 1) * 100
    volume = rng.lognormal(12, 0.5, (n_days, n_stocks)) * (1 + 8 * jump)
    idx = pd.bdate_range("2014-01-02", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_stocks)]
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    panel = dict(open=mk(O), high=mk(H), low=mk(L), close=mk(C), volume=mk(volume))
    panel["value"] = panel["close"] * panel["volume"]
    panel["chg"] = mk(chg)
    panel["meta"] = pd.DataFrame({"name": cols, "market": "SYN"}, index=cols)
    return _finish(panel)


def load_pack(path: str, prefix: str = "") -> dict:
    """pack_data.py / pack_krx.py 로 만든 연도별 npz 묶음을 패널로 되읽기.
    prefix="krx_" 이면 KRX 묶음(krx_YYYY.npz, krx_meta.csv), 기본은 키움 묶음(YYYY.npz, meta.csv)."""
    files = sorted(f for f in Path(path).glob(f"{prefix}*.npz") if prefix or not f.name.startswith("krx_"))
    if not files:
        raise FileNotFoundError(f"{path} 에 npz 파일이 없습니다")
    parts = {f: [] for f in ("open", "high", "low", "close", "volume")}
    idx, codes = [], None
    for fp in files:
        z = np.load(fp)
        c = z["codes"]
        if codes is None:
            codes = c
        elif not np.array_equal(codes, c):
            raise ValueError("연도별 파일의 종목 목록이 다릅니다")
        idx.append(pd.to_datetime(z["dates"]))
        for f in parts:
            parts[f].append(z[f])
    dates = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    panel = {f: pd.DataFrame(np.vstack(v), index=dates, columns=codes) for f, v in parts.items()}
    panel["value"] = panel["close"] * panel["volume"]
    prev = panel["close"].ffill().shift(1)
    panel["chg"] = (panel["close"] / prev - 1) * 100
    mp = Path(path) / f"{prefix}meta.csv"
    if mp.exists():
        mt = pd.read_csv(mp, dtype={"code": str}).set_index("code")
        panel["meta"] = mt[[c for c in ("name", "market", "sector") if c in mt.columns]].reindex(codes)
    else:
        panel["meta"] = pd.DataFrame({"name": codes, "market": "UNKNOWN"}, index=codes)
    return _finish(panel)


def save_panel(panel: dict, fp: str):
    pd.to_pickle(panel, fp)


def load_panel(fp: str) -> dict:
    return pd.read_pickle(fp)
