#!/usr/bin/env python3
"""GitHub Actions 일일 분석 스크립트 — DART ROE + yfinance."""
import io
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

import requests
from dotenv import load_dotenv

load_dotenv()

from scraper import fetch_kospi_stocks, fetch_stock_detail, format_for_prompt
from scraper_us import fetch_sp500_stocks, format_for_prompt_us
from analyzer import analyze_stocks
from analyzer_us import analyze_stocks_us
from report import build_report
from history import save_report, load_previous_report, compare_with_previous

DART_KEY = os.getenv("DART_API_KEY", "")


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


def _fetch_dart_roe(stock_code: str, corp_map: dict) -> str | None:
    """DART 공식 사업보고서에서 ROE(자기자본이익률) 조회."""
    corp_code = corp_map.get(stock_code)
    if not corp_code:
        return None
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
                timeout=15,
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
    return None


def run_korean() -> dict:
    print("\n🇰🇷 한국 주식 분석 시작...")
    stocks_raw = fetch_kospi_stocks(top_n=80)
    stock_table = format_for_prompt(stocks_raw)

    prev_report = load_previous_report()
    analyzed = analyze_stocks(stock_table, "")

    for stock in analyzed:
        detail = fetch_stock_detail(stock.get("code", ""))
        stock.update(detail)

    # DART 공식 ROE로 교체
    if DART_KEY:
        corp_map = _get_dart_corp_map()
        for stock in analyzed:
            roe = _fetch_dart_roe(stock.get("code", ""), corp_map)
            if roe:
                stock["roe"] = roe
                print(f"  {stock['name']}: ROE {roe}% (DART)")
    else:
        print("  DART_API_KEY 없음 — yfinance ROE 사용")

    overlaps = compare_with_previous(analyzed, prev_report)
    save_report(analyzed)

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
