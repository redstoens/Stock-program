"""네이버증권 WiseReport — 증권사 컨센서스 목표주가·투자의견 크롤링."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.naver.com/",
}
_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}"


def _fetch_one(code: str) -> tuple[str, dict]:
    try:
        r = requests.get(_URL.format(code=code), headers=_HEADERS, timeout=10)
        if r.status_code != 200:
            return code, {}
        text = r.text

        # chartData2: target_price 시계열 — 마지막 유효값 = 최신 컨센서스
        target = None
        m2 = re.search(r"var chartData2\s*=\s*(\{.*?\})\s*;", text, re.DOTALL)
        if m2:
            d2 = json.loads(m2.group(1))
            valid = [p["y"] for p in d2.get("target_price", []) if p.get("y")]
            if valid:
                target = int(valid[-1])

        # chartData3: today 투자의견 분포
        buy = hold = sell = 0
        m3 = re.search(r"var chartData3\s*=\s*(\{.*?\})\s*;", text, re.DOTALL)
        if m3:
            d3 = json.loads(m3.group(1))
            dist = {it["name"]: (it["y"] or 0) for it in d3.get("today", [])}
            buy  = int(dist.get("강력매수", 0) + dist.get("매수", 0))
            hold = int(dist.get("중립", 0))
            sell = int(dist.get("매도", 0) + dist.get("강력매도", 0))

        total = buy + hold + sell
        if target is None and total == 0:
            return code, {}

        out: dict = {"analyst_count": total, "analyst_buy": buy,
                     "analyst_hold": hold, "analyst_sell": sell}
        if target:
            out["consensus_target"] = target
            out["consensus_target_str"] = f"{target:,}원"

        return code, out
    except Exception:
        return code, {}


def fetch_consensus(codes: list[str]) -> dict[str, dict]:
    """
    여러 종목코드의 증권사 컨센서스 조회.
    Returns: {code: {consensus_target, consensus_target_str,
                     analyst_count, analyst_buy, analyst_hold, analyst_sell}}
    """
    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(_fetch_one, codes))
    return {code: m for code, m in pairs if m}
