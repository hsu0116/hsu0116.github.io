"""
키움 REST API 로 코스피/코스닥 전종목 일봉(수정주가)을 받아 CSV 로 저장.

준비
  1) 키움 OpenAPI 사이트(openapi.kiwoom.com)에서 REST API 사용 신청 -> 앱키/시크릿키 발급
  2) 이 스크립트를 실행할 PC의 공인 IP 를 허용 IP 로 등록
  3) 환경변수 설정 (키는 코드/채팅에 적지 말 것)
       Windows PowerShell:  $env:KIWOOM_APPKEY="..."; $env:KIWOOM_SECRETKEY="..."
       macOS/Linux:         export KIWOOM_APPKEY=... KIWOOM_SECRETKEY=...
실행
  python kiwoom_fetch.py --start 20150101 --out kiwoom_csv
  python run_research.py --source csv --path kiwoom_csv --meta kiwoom_csv/_meta.csv
  (모의투자 서버로 시험: --mock)

주의
- 키움은 현재 상장 종목만 조회 가능 -> 상장폐지 종목 누락(생존편향, 결과가 낙관적).
- TR: ka10099(종목정보 리스트), ka10081(주식일봉차트, 수정주가). 응답 필드명이 바뀌면
  --debug 로 원문을 확인하고 FIELD 매핑을 고칠 것.
- 요청 제한을 피하려고 기본 0.25초 간격. 전종목 10년치 ≈ 2,700종목 × 약 4페이지 → 1시간 내외.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REAL = "https://api.kiwoom.com"
MOCK = "https://mockapi.kiwoom.com"

# 일봉 응답 필드 -> 표준 컬럼
FIELD = {"dt": "Date", "open_pric": "Open", "high_pric": "High", "low_pric": "Low",
         "cur_prc": "Close", "trde_qty": "Volume", "trde_prica": "Value"}


class Kiwoom:
    def __init__(self, base: str, sleep: float = 0.25, debug: bool = False):
        self.base, self.sleep, self.debug = base, sleep, debug
        self.s = requests.Session()
        self.token = self._token()

    def _token(self) -> str:
        ak, sk = os.getenv("KIWOOM_APPKEY"), os.getenv("KIWOOM_SECRETKEY")
        if not ak or not sk:
            sys.exit("환경변수 KIWOOM_APPKEY / KIWOOM_SECRETKEY 를 설정하세요.")
        r = self.s.post(f"{self.base}/oauth2/token",
                        json={"grant_type": "client_credentials", "appkey": ak, "secretkey": sk},
                        headers={"Content-Type": "application/json;charset=UTF-8"}, timeout=20)
        j = r.json()
        if "token" not in j:
            sys.exit(f"토큰 발급 실패: {j}  (허용 IP 등록 여부 확인)")
        return j["token"]

    def call(self, path: str, api_id: str, body: dict, cont: str = "N", key: str = ""):
        h = {"Content-Type": "application/json;charset=UTF-8", "authorization": f"Bearer {self.token}",
             "api-id": api_id, "cont-yn": cont, "next-key": key}
        for attempt in range(5):
            r = self.s.post(f"{self.base}{path}", json=body, headers=h, timeout=30)
            time.sleep(self.sleep)
            if r.status_code == 429 or (r.status_code >= 500):
                time.sleep(2 ** attempt)
                continue
            j = r.json()
            if self.debug:
                print(api_id, r.status_code, dict(r.headers).get("cont-yn"), json.dumps(j, ensure_ascii=False)[:800])
            return j, r.headers.get("cont-yn", "N"), r.headers.get("next-key", "")
        raise RuntimeError(f"{api_id} 반복 실패")

    def listing(self) -> pd.DataFrame:
        rows = []
        for mk, tp in (("KOSPI", "0"), ("KOSDAQ", "10")):
            cont, key = "N", ""
            while True:
                j, cont, key = self.call("/api/dostk/stkinfo", "ka10099", {"mrkt_tp": tp}, cont, key)
                for it in j.get("list", []):
                    rows.append({"code": it.get("code"), "name": it.get("name"), "market": mk,
                                 "sector": it.get("upName") or ""})
                if cont != "Y":
                    break
        df = pd.DataFrame(rows).drop_duplicates("code")
        return df[df.code.str.fullmatch(r"\d{5}[0-9A-Z]", na=False)]

    def daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        recs, cont, key = [], "N", ""
        while True:
            j, cont, key = self.call("/api/dostk/chart", "ka10081",
                                     {"stk_cd": code, "base_dt": end, "upd_stkpc_tp": "1"}, cont, key)
            part = j.get("stk_dt_pole_chart_qry") or []
            recs += part
            if cont != "Y" or not part or min(p.get("dt", "99999999") for p in part) <= start:
                break
        if not recs:
            return pd.DataFrame()
        df = pd.DataFrame(recs)[list(FIELD)].rename(columns=FIELD)
        for c in ("Open", "High", "Low", "Close", "Volume", "Value"):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[+,\-]", "", regex=True), errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df[(df.Date >= pd.Timestamp(start))].drop_duplicates("Date").sort_values("Date")
        # 거래대금 단위가 버전에 따라 다를 수 있어 종가×거래량 근사로 통일
        df["Value"] = df["Close"] * df["Volume"]
        return df.set_index("Date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--out", default="kiwoom_csv")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞 N종목만")
    ap.add_argument("--listing_only", action="store_true", help="종목목록(업종 포함)만 다시 받아 _meta.csv 갱신")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    kw = Kiwoom(MOCK if a.mock else REAL, debug=a.debug)
    meta = kw.listing()
    meta = meta[~meta.name.str.contains("스팩|ETN|선물|인버스|레버리지", na=False)]
    meta.to_csv(out / "_meta.csv", index=False, encoding="utf-8-sig")
    print(f"종목 {len(meta)}개")
    if a.listing_only:
        print("업종 포함 종목목록 저장 완료 ->", out / "_meta.csv")
        return
    codes = meta.code.tolist()[: a.limit or None]
    for n, code in enumerate(codes, 1):
        fp = out / f"{code}.csv"
        if fp.exists():
            continue
        try:
            df = kw.daily(code, a.start, a.end)
        except Exception as e:  # noqa: BLE001
            print("fail", code, e)
            continue
        if len(df):
            df.to_csv(fp)
        if n % 100 == 0:
            print(f"{n}/{len(codes)}")
    print("완료 ->", out)


if __name__ == "__main__":
    main()
