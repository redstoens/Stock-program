import io
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DART_KEY = os.getenv("DART_API_KEY", "")
_CORP_MAP_CACHE = "/tmp/dart_corp_map.json"
_CORP_MAP: dict[str, str] | None = None

# KOSPI 주요 종목 (시가총액 상위, .KS = KOSPI)
KOSPI_TICKERS = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "207940.KS",  # 삼성바이오로직스
    "005380.KS",  # 현대차
    "000270.KS",  # 기아
    "068270.KS",  # 셀트리온
    "051910.KS",  # LG화학
    "055550.KS",  # 신한지주
    "035420.KS",  # NAVER
    "012330.KS",  # 현대모비스
    "028260.KS",  # 삼성물산
    "105560.KS",  # KB금융
    "066570.KS",  # LG전자
    "032830.KS",  # 삼성생명
    "096770.KS",  # SK이노베이션
    "003550.KS",  # LG
    "017670.KS",  # SK텔레콤
    "015760.KS",  # 한국전력
    "034730.KS",  # SK
    "000810.KS",  # 삼성화재
    "086790.KS",  # 하나금융지주
    "009150.KS",  # 삼성전기
    "030200.KS",  # KT
    "018260.KS",  # 삼성SDS
    "010130.KS",  # 고려아연
    "006400.KS",  # 삼성SDI
    "011200.KS",  # HMM
    "035720.KS",  # 카카오
    "003490.KS",  # 대한항공
    "090430.KS",  # 아모레퍼시픽
    "010950.KS",  # S-Oil
    "024110.KS",  # 기업은행
    "316140.KS",  # 우리금융지주
    "259960.KS",  # 크래프톤
    "047810.KS",  # 한국항공우주
    "009540.KS",  # HD한국조선해양
    "034020.KS",  # 두산에너빌리티
    "000720.KS",  # 현대건설
    "012450.KS",  # 한화에어로스페이스
    "329180.KS",  # HD현대중공업
    "267250.KS",  # HD현대
    "078930.KS",  # GS
    "000100.KS",  # 유한양행
    "128940.KS",  # 한미약품
    "352820.KS",  # 하이브
    "021240.KS",  # 코웨이
    "139480.KS",  # 이마트
    "097950.KS",  # CJ제일제당
    "004990.KS",  # 롯데지주
    "272210.KS",  # 한화시스템
    "336260.KS",  # 두산밥캣
    "028050.KS",  # 삼성엔지니어링
    "003230.KS",  # 삼양식품
    "010140.KS",  # 삼성중공업
    "011790.KS",  # SKC
    "377300.KS",  # 카카오페이
    "251270.KS",  # 넷마블
    "004370.KS",  # 농심
    "161390.KS",  # 한국타이어앤테크놀로지
    "042660.KS",  # 한화오션
]


# ── DART 공통 유틸 ──────────────────────────────────────────────

def _get_corp_map() -> dict[str, str]:
    """stock_code → corp_code 매핑. /tmp 캐시 → 메모리 캐시 순으로 활용."""
    global _CORP_MAP
    if _CORP_MAP is not None:
        return _CORP_MAP

    # /tmp 캐시 확인 (Vercel warm instance 재활용)
    try:
        with open(_CORP_MAP_CACHE) as f:
            _CORP_MAP = json.load(f)
            return _CORP_MAP
    except Exception:
        pass

    # DART에서 직접 다운로드
    try:
        r = requests.get(
            f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_KEY}",
            timeout=30,
        )
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(zf.read("CORPCODE.xml"))
        mapping = {
            item.findtext("stock_code", "").strip(): item.findtext("corp_code", "").strip()
            for item in root.findall("list")
            if item.findtext("stock_code", "").strip()
        }
        _CORP_MAP = mapping
        try:
            with open(_CORP_MAP_CACHE, "w") as f:
                json.dump(_CORP_MAP, f)
        except Exception:
            pass
    except Exception:
        _CORP_MAP = {}

    return _CORP_MAP


def _fetch_dart_roe(stock_code: str) -> str:
    """DART 공식 재무제표에서 ROE(자기자본이익률) 조회."""
    if not DART_KEY:
        return "N/A"
    corp_map = _get_corp_map()
    corp_code = corp_map.get(stock_code)
    if not corp_code:
        return "N/A"

    for year in ("2024", "2023"):
        try:
            r = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglIndx.json",
                params={
                    "crtfc_key": DART_KEY,
                    "corp_code": corp_code,
                    "bsns_year": year,
                    "reprt_code": "11011",   # 사업보고서
                    "idx_cl_code": "M210000",  # 수익성 지표
                },
                timeout=10,
            )
            data = r.json()
            if data.get("status") != "000":
                continue
            for item in data.get("list", []):
                nm = item.get("idx_nm", "")
                if "자기자본이익률" in nm or "ROE" in nm.upper():
                    val = item.get("idx_val", "").replace(",", "")
                    return str(round(float(val), 1))
        except Exception:
            continue
    return "N/A"


# ── yfinance 수집 ────────────────────────────────────────────────

def _fetch_one(ticker_sym: str) -> dict | None:
    try:
        info = yf.Ticker(ticker_sym).info
        code = ticker_sym.replace(".KS", "").replace(".KQ", "")
        name = info.get("longName") or info.get("shortName") or code

        cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        market_cap = info.get("marketCap") or 0
        if not cur_price or not market_cap:
            return None

        change_pct = info.get("regularMarketChangePercent") or 0
        per_val = info.get("trailingPE") or 0
        roe_raw = info.get("returnOnEquity") or 0
        w52_high = info.get("fiftyTwoWeekHigh") or 0
        w52_low = info.get("fiftyTwoWeekLow") or 0

        mktcap_억 = market_cap // 100_000_000
        pct_from_high = round((cur_price - w52_high) / w52_high * 100, 1) if w52_high > 0 else 0

        return {
            "code": code,
            "name": name,
            "price": f"{int(cur_price):,}",
            "change_rate": f"{change_pct:+.2f}%" if change_pct else "",
            "market_cap": f"{mktcap_억:,}",
            "market_cap_raw": market_cap,
            "per": str(round(per_val, 1)) if per_val > 0 else "N/A",
            "roe": str(round(roe_raw * 100, 1)) if roe_raw else "N/A",
            "current_price_raw": int(cur_price),
            "week52_high": f"{int(w52_high):,}" if w52_high else "N/A",
            "week52_low": f"{int(w52_low):,}" if w52_low else "N/A",
            "week52_pct_from_high": f"{pct_from_high:+.1f}%" if w52_high else "N/A",
        }
    except Exception:
        return None


def fetch_kospi_stocks(top_n: int = 80) -> list[dict]:
    """KOSPI 주요 종목 — yfinance (AI 분석용 초기 데이터)."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_fetch_one, KOSPI_TICKERS))

    stocks = [r for r in results if r is not None]
    if not stocks:
        raise RuntimeError("종목 데이터를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")

    stocks.sort(key=lambda x: x["market_cap_raw"], reverse=True)
    for i, s in enumerate(stocks[:top_n], start=1):
        s["rank"] = i
    return stocks[:top_n]


def fetch_stock_detail(code: str) -> dict:
    """AI 선택 종목 상세 — 52주 고저가(yfinance) + ROE(DART 공식)."""
    detail = {
        "current_price_raw": 0,
        "week52_high": "N/A",
        "week52_low": "N/A",
        "week52_pct_from_high": "N/A",
    }
    try:
        info = yf.Ticker(f"{code}.KS").info
        cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        w52_high = info.get("fiftyTwoWeekHigh") or 0
        w52_low = info.get("fiftyTwoWeekLow") or 0

        if cur_price and w52_high:
            pct = round((cur_price - w52_high) / w52_high * 100, 1)
            detail.update({
                "current_price_raw": int(cur_price),
                "week52_high": f"{int(w52_high):,}",
                "week52_low": f"{int(w52_low):,}",
                "week52_pct_from_high": f"{pct:+.1f}%",
            })
    except Exception:
        pass

    # ROE를 DART 공식 데이터로 교체
    dart_roe = _fetch_dart_roe(code)
    if dart_roe != "N/A":
        detail["roe"] = dart_roe

    return detail


def format_for_prompt(stocks: list[dict]) -> str:
    """AI 프롬프트 테이블 문자열."""
    lines = ["순위 | 종목명 | 종목코드 | 현재가 | 등락률 | 시가총액(억) | PER | ROE(%)"]
    lines.append("-" * 80)
    for s in stocks:
        lines.append(
            f"{s['rank']} | {s['name']} | {s['code']} | {s['price']} | "
            f"{s['change_rate']} | {s['market_cap']} | {s['per']} | {s['roe']}"
        )
    return "\n".join(lines)
