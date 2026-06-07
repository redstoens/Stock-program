#!/usr/bin/env python3
"""GitHub Actions 일일 분석 스크립트 — DART + 뉴스 + 3년 트렌드 + 2차 AI 분석."""
import html
import io
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import quote

import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

import requests
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

from scraper import fetch_kospi_stocks, fetch_stock_detail, format_for_prompt
from scraper_us import fetch_sp500_stocks, format_for_prompt_us
from analyzer import analyze_stocks
from analyzer_us import analyze_stocks_us
from report import build_report
from history import save_report, load_previous_report, compare_with_previous

DART_KEY = os.getenv("DART_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"


# ── DART 유틸 ──────────────────────────────────────────────────

def _get_dart_corp_map() -> dict[str, str]:
    """DART 전체 기업코드 다운로드 → stock_code: corp_code 매핑."""
    try:
        print("  DART 기업코드 다운로드 중...")
        r = requests.get(
            f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_KEY}",
            timeout=60,
        )
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(zf.read("CORPCODE.xml"))
        mapping = {
            item.findtext("stock_code", "").strip(): item.findtext("corp_code", "").strip()
            for item in root.findall("list")
            if item.findtext("stock_code", "").strip()
        }
        print(f"  기업코드 {len(mapping)}개 로드 완료")
        return mapping
    except Exception as e:
        print(f"  DART 기업코드 실패: {e}")
        return {}


def _fetch_dart_all(stock_code: str, corp_map: dict) -> tuple[dict, dict]:
    """DART에서 현재 지표 + 3년 트렌드를 한번에 조회 (API 호출 최소화).

    Returns:
        indicators: {roe, operating_margin, debt_ratio} — 최신 연도 기준
        trend:      {year: {roe, om}} — 2022·2023·2024
    """
    corp_code = corp_map.get(stock_code)
    if not corp_code:
        return {}, {}

    indicators = {}
    trend = {}

    # ── 수익성 지표 (M210000) — 3년 전부 조회 ──────────────────
    for year in ("2024", "2023", "2022"):
        try:
            r = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglIndx.json",
                params={
                    "crtfc_key": DART_KEY,
                    "corp_code": corp_code,
                    "bsns_year": year,
                    "reprt_code": "11011",
                    "idx_cl_code": "M210000",
                },
                timeout=15,
            )
            data = r.json()
            if data.get("status") != "000":
                continue
            roe, om = None, None
            for item in data.get("list", []):
                nm = item.get("idx_nm", "")
                val = item.get("idx_val", "").replace(",", "")
                try:
                    if "자기자본이익률" in nm:
                        roe = round(float(val), 1)
                    elif "매출액영업이익률" in nm:
                        om = round(float(val), 1)
                except Exception:
                    pass
            # 최신 연도 값을 indicators에 저장 (처음 성공한 값)
            if roe is not None and "roe" not in indicators:
                indicators["roe"] = str(roe)
            if om is not None and "operating_margin" not in indicators:
                indicators["operating_margin"] = str(om)
            # 3년 트렌드에 저장
            if roe is not None or om is not None:
                trend[year] = {}
                if roe is not None:
                    trend[year]["roe"] = roe
                if om is not None:
                    trend[year]["om"] = om
        except Exception:
            continue

    # ── 안정성 지표 (M220000) — 최신 연도만 ────────────────────
    for year in ("2024", "2023"):
        if "debt_ratio" in indicators:
            break
        try:
            r = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglIndx.json",
                params={
                    "crtfc_key": DART_KEY,
                    "corp_code": corp_code,
                    "bsns_year": year,
                    "reprt_code": "11011",
                    "idx_cl_code": "M220000",
                },
                timeout=15,
            )
            data = r.json()
            if data.get("status") != "000":
                continue
            for item in data.get("list", []):
                nm = item.get("idx_nm", "")
                val = item.get("idx_val", "").replace(",", "")
                if "부채비율" in nm:
                    try:
                        indicators["debt_ratio"] = str(round(float(val), 1))
                        break
                    except Exception:
                        pass
        except Exception:
            continue

    return indicators, trend


def _format_trend_str(trend: dict) -> str:
    """트렌드 딕셔너리 → AI 프롬프트용 문자열."""
    if not trend:
        return "데이터 없음"
    parts = []
    for year in ("2022", "2023", "2024"):
        if year in trend:
            d = trend[year]
            parts.append(
                f"{year}년 ROE {d.get('roe','N/A')}% / 영업이익률 {d.get('om','N/A')}%"
            )
    return " → ".join(parts) if parts else "데이터 없음"


# ── 투자 검증 추적 ────────────────────────────────────────────

def _parse_price_raw(price_str: str) -> int:
    """'400,000원' 또는 '290,000원 (현재가 대비 -11.8%)' → 400000."""
    if not price_str:
        return 0
    m = re.search(r"([\d,]+)원", price_str)
    return int(m.group(1).replace(",", "")) if m else 0


def _hold_max_days(hold_period: str) -> int:
    """'1~2년' → 730, '1-3개월' → 90, '3~6개월' → 180."""
    if not hold_period:
        return 365
    nums = [int(n) for n in re.findall(r"\d+", hold_period)]
    if not nums:
        return 365
    max_n = max(nums)
    return max_n * 365 if "년" in hold_period else max_n * 30


def _fetch_current_price(code: str) -> int:
    """yfinance로 현재가 조회 (KS → KQ 순서로 시도)."""
    for suffix in (".KS", ".KQ"):
        try:
            info = yf.Ticker(f"{code}{suffix}").info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            if price:
                return int(price)
        except Exception:
            pass
    return 0


def _update_tracking(analyzed: list[dict], track_path: str) -> None:
    """추천 종목을 누적 추적하고 현재가 기반 수익률·상태를 업데이트."""
    try:
        with open(track_path, encoding="utf-8") as f:
            track = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        track = {"records": []}

    today = date.today().isoformat()

    # 오늘 분석 결과 신규 추가 (같은 날 중복 방지)
    existing_today = {r["code"] for r in track["records"] if r["rec_date"] == today}
    added = 0
    for stock in analyzed:
        code = stock.get("code", "")
        if not code or code in existing_today:
            continue
        entry_price = stock.get("current_price_raw", 0)
        if not entry_price:
            continue
        hold_period = stock.get("hold_period", "")
        track["records"].append({
            "rec_date":          today,
            "code":              code,
            "name":              stock.get("name", ""),
            "entry_price":       entry_price,
            "target_price_raw":  _parse_price_raw(stock.get("future_target", "")),
            "stop_loss_raw":     _parse_price_raw(stock.get("stop_loss", "")),
            "target_str":        stock.get("future_target", ""),
            "stop_loss_str":     stock.get("stop_loss", ""),
            "investment_horizon": stock.get("investment_horizon", ""),
            "hold_period":       hold_period,
            "max_days":          _hold_max_days(hold_period),
            "status":            "진행중",
            "current_price":     entry_price,
            "current_return_pct": 0.0,
            "last_updated":      today,
            "exit_date":         None,
            "exit_price":        None,
            "exit_return_pct":   None,
        })
        added += 1

    # 진행중 종목 현재가 조회
    active_codes = list({r["code"] for r in track["records"] if r["status"] == "진행중"})
    price_map: dict[str, int] = {}
    if active_codes:
        print(f"  추적 종목 현재가 조회 중 ({len(active_codes)}개)...")
        for code in active_codes:
            price = _fetch_current_price(code)
            if price:
                price_map[code] = price

    # 각 레코드 상태 업데이트
    for rec in track["records"]:
        if rec["status"] != "진행중":
            continue
        code = rec["code"]
        current_price = price_map.get(code, rec["current_price"])
        if not current_price:
            continue
        entry_price = rec["entry_price"]
        return_pct  = round((current_price - entry_price) / entry_price * 100, 2) if entry_price else 0.0
        days_held   = (date.today() - date.fromisoformat(rec["rec_date"])).days
        target_raw  = rec.get("target_price_raw", 0)
        stop_raw    = rec.get("stop_loss_raw", 0)

        if target_raw > 0 and current_price >= target_raw:
            new_status = "목표달성"
        elif stop_raw > 0 and current_price <= stop_raw:
            new_status = "손절"
        elif days_held > rec.get("max_days", 365):
            new_status = "만료"
        else:
            new_status = "진행중"

        rec["current_price"]       = current_price
        rec["current_return_pct"]  = return_pct
        rec["last_updated"]        = today
        if new_status != "진행중":
            rec["status"]          = new_status
            rec["exit_date"]       = today
            rec["exit_price"]      = current_price
            rec["exit_return_pct"] = return_pct

    with open(track_path, "w", encoding="utf-8") as f:
        json.dump(track, f, ensure_ascii=False, indent=2)

    total  = len(track["records"])
    active = sum(1 for r in track["records"] if r["status"] == "진행중")
    print(f"  투자 검증 업데이트: 총 {total}개 (신규 {added}개, 진행중 {active}개)")


# ── 뉴스 ───────────────────────────────────────────────────────

def _fetch_news(company_name: str) -> list[str]:
    """Google News RSS에서 최신 뉴스 헤드라인 3개 반환."""
    try:
        query = quote(f"{company_name} 주식")
        url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item")[:3]:
            el = item.find("title")
            if el is not None and el.text:
                titles.append(html.unescape(el.text))
        return titles
    except Exception:
        return []


# ── 2차 AI 분석 ────────────────────────────────────────────────

def _refine_with_context(analyzed: list[dict],
                         news_map: dict,
                         trend_map: dict) -> list[dict]:
    """뉴스 + 3년 트렌드를 반영한 2차 Gemini 분석."""
    if not GEMINI_KEY:
        return analyzed

    context_blocks = []
    for stock in analyzed:
        code = stock.get("code", "")
        name = stock.get("name", "")
        news = news_map.get(code, [])
        trend = trend_map.get(code, {})
        news_str = " / ".join(news) if news else "뉴스 없음"
        trend_str = _format_trend_str(trend)
        context_blocks.append(
            f"[{name} ({code})]\n"
            f"최신 뉴스: {news_str}\n"
            f"3년 재무 트렌드: {trend_str}"
        )

    prompt = f"""아래는 AI가 선정한 10개 한국 주식 종목과 각 종목의 실제 최신 뉴스 및 3년 재무 트렌드입니다.

=== 종목별 실제 데이터 ===
{chr(10).join(context_blocks)}

=== 기존 분석 JSON ===
{json.dumps({"stocks": analyzed}, ensure_ascii=False)}

위 실제 데이터를 반영해 각 종목의 분석을 보강하세요:
1. news_summary → 실제 뉴스 헤드라인을 반영해 업데이트 (1-2문장)
2. reason → 3년 재무 트렌드(ROE·영업이익률 개선/악화 여부) 반영해 보강
3. trend_summary → "YYYY→YYYY→YYYY 영업이익률/ROE 흐름" 한 줄 요약 (신규 필드)
4. earnings_trend → 실제 3년 데이터 기반으로 재평가

future_target, stop_loss, investment_horizon 등 나머지 필드는 그대로 유지하세요.

JSON만 반환하세요 (다른 텍스트 없이):
{{"stocks": [보강된 10개 종목]}}"""

    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        refined = json.loads(response.text)
        stocks = refined.get("stocks", [])
        if stocks:
            print(f"  2차 AI 분석 완료: {len(stocks)}개 종목 보강")
            return stocks
    except Exception as e:
        print(f"  2차 AI 분석 실패 (기존 결과 유지): {e}")
    return analyzed


# ── 메인 분석 함수 ─────────────────────────────────────────────

def run_korean() -> dict:
    print("\n🇰🇷 한국 주식 분석 시작...")
    stocks_raw = fetch_kospi_stocks(top_n=80)
    stock_table = format_for_prompt(stocks_raw)

    prev_report = load_previous_report()
    analyzed = analyze_stocks(stock_table, "")

    for stock in analyzed:
        detail = fetch_stock_detail(stock.get("code", ""))
        stock.update(detail)

    corp_map = {}
    trend_map = {}
    if DART_KEY:
        corp_map = _get_dart_corp_map()

        # DART 공식 지표 + 3년 트렌드 (한번에 조회)
        print("  DART 지표 + 3년 트렌드 수집 중...")
        for stock in analyzed:
            indicators, trend = _fetch_dart_all(stock.get("code", ""), corp_map)
            if indicators:
                stock.update(indicators)
            trend_map[stock.get("code", "")] = trend
            print(f"    {stock['name']}: ROE={indicators.get('roe','N/A')}% "
                  f"영업이익률={indicators.get('operating_margin','N/A')}% "
                  f"부채비율={indicators.get('debt_ratio','N/A')}% "
                  f"트렌드={list(trend.keys())}")
    else:
        print("  DART_API_KEY 없음 — yfinance 데이터 사용")

    # 뉴스 수집
    print("  뉴스 수집 중...")
    news_map = {}
    for stock in analyzed:
        code = stock.get("code", "")
        name = stock.get("name", "")
        news_map[code] = _fetch_news(name)
        print(f"    {name}: 뉴스 {len(news_map[code])}건")

    # 2차 AI 분석 (뉴스 + 트렌드 반영)
    print("  2차 AI 분석 (뉴스·트렌드 반영) 중...")
    analyzed = _refine_with_context(analyzed, news_map, trend_map)

    overlaps = compare_with_previous(analyzed, prev_report)
    save_report(analyzed)

    # 투자 추적 업데이트
    _update_tracking(analyzed, "api/data/track_kr.json")

    return {
        "success": True,
        "stocks": analyzed,
        "report": build_report(analyzed),
        "date": date.today().isoformat(),
        "prev_date": prev_report.get("date") if prev_report else None,
        "overlaps": overlaps,
    }


def run_us() -> dict:
    print("\n🇺🇸 미국 주식 분석 시작...")
    stocks_raw = fetch_sp500_stocks(top_n=50)
    stock_table = format_for_prompt_us(stocks_raw)
    analyzed = analyze_stocks_us(stock_table, "")

    stock_map = {s["code"]: s for s in stocks_raw}
    for stock in analyzed:
        code = stock.get("code", "")
        if code in stock_map:
            raw = stock_map[code]
            stock.setdefault("current_price_raw", raw.get("current_price_raw", 0))
            stock.setdefault("week52_high", raw.get("week52_high", ""))
            stock.setdefault("week52_low", raw.get("week52_low", ""))
            stock.setdefault("week52_pct_from_high", raw.get("week52_pct_from_high", ""))

    return {
        "success": True,
        "stocks": analyzed,
        "report": build_report(analyzed),
        "date": date.today().isoformat(),
        "prev_date": None,
        "overlaps": [],
    }


if __name__ == "__main__":
    os.makedirs("api/data", exist_ok=True)

    kr = run_korean()
    with open("api/data/latest_kr.json", "w", encoding="utf-8") as f:
        json.dump(kr, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 한국: {len(kr['stocks'])}개 → api/data/latest_kr.json")

    us = run_us()
    with open("api/data/latest_us.json", "w", encoding="utf-8") as f:
        json.dump(us, f, ensure_ascii=False, indent=2)
    print(f"✅ 미국: {len(us['stocks'])}개 → api/data/latest_us.json")
