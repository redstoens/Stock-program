"""
DART(금융감독원 전자공시) API — 핵심 재무지표 조회
- 영업이익 YoY 성장률 및 트렌드 (개선/보합/악화)
- 부채비율
- 영업이익률
"""
import io
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

DART_API    = "https://opendart.fss.or.kr/api"
_CACHE_PATH = "/tmp/dart_corp_map.json"


def _load_corp_map() -> dict[str, str]:
    """종목코드 → DART 고유번호 매핑. /tmp에 캐시해 재요청 방지."""
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    api_key = os.getenv("DART_API_KEY", "")
    if not api_key:
        return {}

    try:
        res  = requests.get(f"{DART_API}/corpCode.xml",
                            params={"crtfc_key": api_key}, timeout=15)
        z    = zipfile.ZipFile(io.BytesIO(res.content))
        root = ET.fromstring(z.read("CORPCODE.xml"))

        mapping: dict[str, str] = {}
        for item in root.findall("list"):
            sc = (item.findtext("stock_code") or "").strip()
            cc = (item.findtext("corp_code") or "").strip()
            if sc and cc:
                mapping[sc] = cc

        try:
            with open(_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(mapping, f)
        except Exception:
            pass
        return mapping
    except Exception:
        return {}


def _fetch_annual(corp_code: str, year: int, reprt_code: str = "11011") -> dict:
    """재무제표 단일 조회. reprt_code: 11011=사업보고서, 11013=1Q, 11012=H1, 11014=3Q"""
    api_key = os.getenv("DART_API_KEY", "")
    if not api_key:
        return {}

    for fs_div in ("CFS", "OFS"):
        try:
            res  = requests.get(
                f"{DART_API}/fnlttSinglAcnt.json",
                params={
                    "crtfc_key":  api_key,
                    "corp_code":  corp_code,
                    "bsns_year":  str(year),
                    "reprt_code": reprt_code,
                    "fs_div":     fs_div,
                },
                timeout=8,
            )
            data = res.json()
            if data.get("status") != "000" or not data.get("list"):
                continue

            out: dict = {}
            for item in data["list"]:
                nm  = item.get("account_nm", "")
                raw = (item.get("thstrm_amount") or "").replace(",", "")
                try:
                    val = int(raw)
                except Exception:
                    continue
                if nm in ("매출액", "수익(매출액)", "영업수익"):
                    out.setdefault("revenue", val)
                elif nm in ("영업이익", "영업이익(손실)"):
                    out.setdefault("op_profit", val)
                elif nm == "부채총계":
                    out["total_debt"] = val
                elif nm == "자본총계":
                    out["total_equity"] = val

            if out:
                return out
        except Exception:
            continue
    return {}


def fetch_dart_quarter_metrics(stock_codes: list[str]) -> dict[str, dict]:
    """
    최근 분기 영업이익률 조회 (3Q→H1→1Q 순으로 시도).
    Returns: {code: {quarter_label, quarter_op_margin, quarter_debt_ratio}}
    """
    corp_map = _load_corp_map()
    if not corp_map:
        return {}

    cur_y   = datetime.now().year
    prev_y  = cur_y - 1
    # 가장 최근 분기부터 시도 (전년도 기준)
    attempts = [
        (prev_y, "11014", f"{prev_y} 3Q"),
        (prev_y, "11012", f"{prev_y} 반기"),
        (prev_y, "11013", f"{prev_y} 1Q"),
        (cur_y,  "11013", f"{cur_y} 1Q"),
    ]

    def _qmetrics(code: str) -> tuple[str, dict]:
        cc = corp_map.get(code)
        if not cc:
            return code, {}
        for year, reprt_code, label in attempts:
            d = _fetch_annual(cc, year, reprt_code)
            if not d:
                continue
            out: dict = {"quarter_label": label}
            if d.get("op_profit") is not None and d.get("revenue") and d["revenue"] > 0:
                out["quarter_op_margin"] = round(d["op_profit"] / d["revenue"] * 100, 1)
            if d.get("total_debt") and d.get("total_equity") and d["total_equity"] > 0:
                out["quarter_debt_ratio"] = round(d["total_debt"] / d["total_equity"] * 100, 1)
            if len(out) > 1:   # quarter_label 외 데이터 있을 때만
                return code, out
        return code, {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(_qmetrics, stock_codes))

    return {code: m for code, m in pairs if m}


def fetch_dart_metrics(stock_codes: list[str]) -> dict[str, dict]:
    """
    여러 종목코드의 DART 핵심 재무지표 반환.
    Returns: {code: {debt_ratio, op_margin, op_growth_pct, trend_label, year}}
    """
    corp_map = _load_corp_map()
    if not corp_map:
        return {}

    base_y = datetime.now().year - 1   # 가장 최근 사업보고서 연도
    prev_y = base_y - 1                 # 비교 연도

    def _metrics(code: str) -> tuple[str, dict]:
        cc = corp_map.get(code)
        if not cc:
            return code, {}

        d1 = _fetch_annual(cc, base_y)
        d2 = _fetch_annual(cc, prev_y)
        if not d1:
            return code, {}

        out: dict = {"year": base_y}

        # 부채비율
        if d1.get("total_debt") and d1.get("total_equity") and d1["total_equity"] > 0:
            out["debt_ratio"] = round(d1["total_debt"] / d1["total_equity"] * 100, 1)

        # 영업이익률
        if d1.get("op_profit") is not None and d1.get("revenue") and d1["revenue"] > 0:
            out["op_margin"] = round(d1["op_profit"] / d1["revenue"] * 100, 1)

        # 영업이익 YoY 성장률 + 트렌드
        if d1.get("op_profit") is not None and d2.get("op_profit") and d2["op_profit"] != 0:
            growth = (d1["op_profit"] - d2["op_profit"]) / abs(d2["op_profit"]) * 100
            out["op_growth_pct"] = round(growth, 1)
            if growth >= 10:
                out["trend_label"] = "개선"
            elif growth <= -10:
                out["trend_label"] = "악화"
            else:
                out["trend_label"] = "보합"

        return code, out

    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(_metrics, stock_codes))

    return {code: m for code, m in pairs if m}
