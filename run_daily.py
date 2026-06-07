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
                print(f"      DART {year} M210000 status={data.get('status')} msg={data.get('message','')}")
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


# ── 종합 매수 점수 ───────────────────────────────────────────

def _calculate_buy_score(stock: dict, market: str = "kr") -> dict:
    """재무·기술·수급·트렌드를 종합해 0~100점 매수 점수 계산.

    KR: 재무30 + 기술30 + 수급20 + 실적·리스크20
    US: 밸류에이션30 + 기술30 + 수급20 + 실적·리스크20
    """
    score = 0

    # ── 1. 재무 건전성 / 밸류에이션 (30점) ──────────────────────
    if market == "kr":
        # ROE (15점)
        try:
            roe = float(stock.get("roe") or 0)
            pts = 15 if roe >= 15 else 11 if roe >= 10 else 7 if roe >= 5 else 3 if roe > 0 else 0
        except Exception:
            pts = 0
        score += pts

        # 영업이익률 (10점)
        try:
            om = float(stock.get("operating_margin") or 0)
            pts = 10 if om >= 15 else 8 if om >= 10 else 5 if om >= 5 else 2 if om > 0 else 0
        except Exception:
            pts = 0
        score += pts

        # 부채비율 (5점) — 낮을수록 좋음
        try:
            dr = float(stock.get("debt_ratio") or 0)
            pts = (5 if dr <= 50 else 3 if dr <= 100 else 1 if dr <= 200 else 0) if dr > 0 else 3
        except Exception:
            pts = 3
        score += pts
    else:
        # US: PER (20점) + ROE가 있으면 보너스 (10점)
        per_str = str(stock.get("per") or "")
        try:
            per = float(per_str.replace(",", ""))
            pts = 20 if 0 < per <= 15 else 16 if per <= 25 else 10 if per <= 35 else 5 if per <= 50 else 2
        except Exception:
            pts = 8  # N/A → 중립
        score += pts

        try:
            roe = float(stock.get("roe") or 0)
            pts = 10 if roe >= 15 else 7 if roe >= 10 else 4 if roe >= 5 else 2 if roe > 0 else 0
        except Exception:
            pts = 3
        score += pts

    # ── 2. 기술적 지표 (30점) ────────────────────────────────────
    # RSI (15점)
    try:
        rsi = float(stock.get("rsi") or -1)
        if rsi < 0:       pts = 7   # 데이터 없음
        elif rsi <= 30:   pts = 9   # 과매도 — 반등 기대, 다소 위험
        elif rsi <= 45:   pts = 15  # 매수 적정 구간
        elif rsi <= 60:   pts = 10  # 중립
        elif rsi <= 70:   pts = 4   # 과매수 근접
        else:             pts = 0   # 과매수
    except Exception:
        pts = 7
    score += pts

    # MA 신호 (10점)
    ma_sig = stock.get("ma_signal", "")
    if "정배열" in ma_sig:     pts = 10
    elif "단기상승" in ma_sig: pts = 7
    elif "혼조" in ma_sig:     pts = 5
    elif "단기하락" in ma_sig: pts = 3
    elif "역배열" in ma_sig:   pts = 0
    else:                      pts = 5
    score += pts

    # MACD (5점)
    macd_sig = stock.get("macd_signal", "")
    pts = 5 if "골든" in macd_sig else 0 if "데드" in macd_sig else 3
    score += pts

    # ── 3. 수급 동향 (20점) ──────────────────────────────────────
    ft = stock.get("foreign_trend", "")
    # 외국인 (10점)
    f_pts = 10 if ("외국인" in ft and "순매수" in ft) else 0 if ("외국인" in ft and "순매도" in ft) else 5
    # 기관 (10점)
    i_pts = 10 if ("기관" in ft and "순매수" in ft) else 0 if ("기관" in ft and "순매도" in ft) else 5
    score += f_pts + i_pts

    # ── 4. 실적 트렌드 + 리스크 (20점) ──────────────────────────
    et = stock.get("earnings_trend", "")
    pts = 12 if "개선" in et else 6 if "보합" in et else 0 if "악화" in et else 5
    score += pts

    rl = stock.get("risk_level", "")
    pts = 8 if rl == "하" else 4 if rl == "중" else 0 if rl == "상" else 4
    score += pts

    score = max(0, min(100, score))

    if score >= 75:   label = "강력매수"
    elif score >= 60: label = "매수고려"
    elif score >= 40: label = "중립"
    elif score >= 25: label = "관망"
    else:             label = "매수비추"

    return {"buy_score": score, "buy_score_label": label}


# ── 투자 검증 추적 ────────────────────────────────────────────

def _parse_price_raw(price_str: str) -> float:
    """가격 문자열에서 숫자 추출. '400,000원', '$250.50', '250달러' 등 지원."""
    if not price_str:
        return 0.0
    # 한국 원화: "400,000원"
    m = re.search(r"([\d,]+)원", price_str)
    if m:
        return float(m.group(1).replace(",", ""))
    # 달러: "$250.50" 또는 "250.50달러"
    m = re.search(r"\$?([\d,]+\.?\d*)", price_str)
    if m:
        return float(m.group(1).replace(",", ""))
    return 0.0


def _hold_max_days(hold_period: str) -> int:
    """'1~2년' → 730, '1-3개월' → 90, '3~6개월' → 180."""
    if not hold_period:
        return 365
    nums = [int(n) for n in re.findall(r"\d+", hold_period)]
    if not nums:
        return 365
    max_n = max(nums)
    return max_n * 365 if "년" in hold_period else max_n * 30


def _fetch_benchmark_index(market: str = "kr") -> float:
    """벤치마크 지수 현재값 조회 (KOSPI 또는 S&P 500)."""
    ticker = "^KS11" if market == "kr" else "^GSPC"
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 0.0


def _fetch_current_price(code: str, market: str = "kr") -> float:
    """yfinance로 현재가 조회."""
    if market == "us":
        try:
            info = yf.Ticker(code).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            return round(float(price), 2) if price else 0.0
        except Exception:
            return 0.0
    for suffix in (".KS", ".KQ"):
        try:
            info = yf.Ticker(f"{code}{suffix}").info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            if price:
                return float(int(price))
        except Exception:
            pass
    return 0.0


def _fetch_dividend_history(code: str, market: str = "kr") -> dict:
    """yfinance로 최근 3년(2022·2023·2024) 배당 히스토리 + CAGR 계산."""
    try:
        if market == "us":
            divs = yf.Ticker(code).dividends
        else:
            divs = None
            for suffix in (".KS", ".KQ"):
                d = yf.Ticker(f"{code}{suffix}").dividends
                if d is not None and not d.empty:
                    divs = d
                    break
        if divs is None or divs.empty:
            return {}

        # Timestamp.year은 tz 무관하게 동작 → tz 변환 불필요
        annual: dict[str, float] = {}
        for year in (2022, 2023, 2024):
            total = float(sum(v for ts, v in divs.items() if ts.year == year))
            if total > 0:
                annual[str(year)] = total

        if not annual:
            return {}

        years = sorted(annual.keys())

        # CAGR 계산 (2개년 이상일 때)
        growth_str = ""
        if len(years) >= 2:
            n = int(years[-1]) - int(years[0])
            if n > 0 and annual[years[0]] > 0:
                cagr = (annual[years[-1]] / annual[years[0]]) ** (1 / n) - 1
                sign = "+" if cagr >= 0 else ""
                growth_str = f" (CAGR {sign}{cagr * 100:.1f}%)"

        if market == "us":
            parts = [f"{y}: ${annual[y]:.2f}" for y in years]
        else:
            parts = [f"{y}: {int(annual[y]):,}원" for y in years]

        return {"dividend_history": " → ".join(parts) + growth_str}
    except Exception as e:
        print(f"    배당 히스토리 실패 ({code}): {e}")
        return {}


def _fetch_technical_indicators(code: str, market: str = "kr") -> dict:
    """yfinance로 RSI(14)·이동평균(20/60일)·MACD(12,26,9) 계산."""
    try:
        if market == "us":
            hist = yf.Ticker(code).history(period="4mo")
            closes = hist["Close"] if not hist.empty and len(hist) >= 20 else None
        else:
            closes = None
            for suffix in (".KS", ".KQ"):
                hist = yf.Ticker(f"{code}{suffix}").history(period="4mo")
                if not hist.empty and len(hist) >= 20:
                    closes = hist["Close"]
                    break
        if closes is None:
            return {}

        current = float(closes.iloc[-1])

        # RSI(14)
        delta = closes.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = round(float((100 - 100 / (1 + gain / loss)).iloc[-1]), 1)
        rsi_sig = "과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립"

        # 이동평균
        ma20 = float(closes.rolling(20).mean().iloc[-1])
        ma60 = float(closes.rolling(60).mean().iloc[-1]) if len(closes) >= 60 else None

        if ma60:
            ma_sig = "정배열" if current > ma20 > ma60 else \
                     "역배열" if current < ma20 < ma60 else "혼조"
        else:
            ma_sig = "단기상승" if current > ma20 else "단기하락"

        # MACD(12, 26, 9)
        ema12  = closes.ewm(span=12, adjust=False).mean()
        ema26  = closes.ewm(span=26, adjust=False).mean()
        macd_l = ema12 - ema26
        sig_l  = macd_l.ewm(span=9, adjust=False).mean()
        hist_v = float((macd_l - sig_l).iloc[-1])
        macd_sig = "골든크로스" if hist_v > 0 else "데드크로스"

        # 거래량 이상 감지 (오늘 vs 20일 평균)
        vol_result = {}
        try:
            volumes = hist["Volume"].dropna()
            if len(volumes) >= 2:
                today_vol = int(volumes.iloc[-1])
                avg_len   = min(20, len(volumes) - 1)
                avg_vol   = float(volumes.iloc[-avg_len - 1:-1].mean())
                if avg_vol > 0:
                    v_ratio = round(today_vol / avg_vol, 1)
                    if v_ratio >= 3.0:   v_sig = "급증"
                    elif v_ratio >= 2.0: v_sig = "증가"
                    elif v_ratio >= 1.2: v_sig = "보통"
                    elif v_ratio >= 0.5: v_sig = "감소"
                    else:                v_sig = "급감"
                    vol_result = {
                        "volume_ratio":  str(v_ratio),
                        "volume_signal": v_sig,
                    }
        except Exception:
            pass

        fmt = (lambda p: f"${p:.2f}") if market == "us" else (lambda p: f"{int(p):,}")
        return {
            "rsi":         str(rsi),
            "rsi_signal":  rsi_sig,
            "ma20":        fmt(ma20),
            "ma60":        fmt(ma60) if ma60 else "N/A",
            "ma_signal":   ma_sig,
            "macd_hist":   str(round(hist_v, 1)),
            "macd_signal": macd_sig,
            "tech_summary": f"RSI {rsi}({rsi_sig}) / MA {ma_sig} / MACD {macd_sig}",
            **vol_result,
        }
    except Exception as e:
        print(f"    기술적 지표 실패 ({code}): {e}")
        return {}


def _fetch_investor_trend(code: str) -> str:
    """pykrx로 외국인·기관 순매수 추이 조회 (최근 1개월, 국내주 전용)."""
    try:
        from pykrx import stock as pk
        from datetime import datetime, timedelta

        today = datetime.today()
        end_dt   = today.strftime("%Y%m%d")
        start_dt = (today - timedelta(days=30)).strftime("%Y%m%d")

        df = pk.get_market_trading_value_by_investor(start_dt, end_dt, code, detail=False)
        if df is None or df.empty:
            return ""

        # 컬럼명은 pykrx 버전에 따라 "외국인" 또는 "외국인합계"
        f_col = next((c for c in df.columns if "외국인" in c), None)
        i_col = next((c for c in df.columns if "기관" in c), None)

        def _fmt(net: int) -> str:
            a = abs(net)
            direction = "순매수" if net > 0 else "순매도"
            if a >= 100_000_000_000:
                return f"{direction} {a / 100_000_000_000:.1f}천억원"
            if a >= 100_000_000:
                return f"{direction} {a // 100_000_000}억원"
            return f"{direction} {a // 10_000}만원"

        parts = []
        for col, label in [(f_col, "외국인"), (i_col, "기관")]:
            if col is None:
                continue
            net = int(df[col].sum())
            parts.append(f"{label} {_fmt(net)}")

        return " / ".join(parts) + " (최근 1개월)" if parts else ""
    except Exception as e:
        print(f"    투자자 동향 실패 ({code}): {e}")
        return ""


def _update_tracking(analyzed: list[dict], track_path: str, market: str = "kr") -> None:
    """추천 종목을 누적 추적하고 현재가 기반 수익률·상태를 업데이트."""
    try:
        with open(track_path, encoding="utf-8") as f:
            track = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        track = {"records": [], "benchmark": {}}

    if "benchmark" not in track:
        track["benchmark"] = {}

    today = date.today().isoformat()

    # KOSPI 현재값 저장 (진입 기준값 + current 모두 갱신)
    kospi_now = _fetch_benchmark_index(market)
    if kospi_now:
        track["benchmark"]["current"] = kospi_now
        track["benchmark"]["updated"] = today
        if today not in track["benchmark"]:
            track["benchmark"][today] = kospi_now  # 오늘 추천 배치의 진입 기준

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
            price = _fetch_current_price(code, market)
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


# ── 연속 추천 횟수 ───────────────────────────────────────────────

def _count_consecutive_recs(code: str, track_path: str) -> int:
    """해당 종목이 며칠 연속으로 추천됐는지 계산 (오늘 포함)."""
    try:
        with open(track_path, encoding="utf-8") as f:
            track = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1

    rec_dates = sorted(
        set(r["rec_date"] for r in track.get("records", []) if r.get("code") == code),
        reverse=True,
    )
    if not rec_dates:
        return 1

    streak = 1
    for i in range(1, len(rec_dates)):
        d1 = date.fromisoformat(rec_dates[i - 1])
        d2 = date.fromisoformat(rec_dates[i])
        # 3일 이내 간격이면 연속 거래일로 간주 (주말·공휴일 처리)
        if (d1 - d2).days <= 3:
            streak += 1
        else:
            break
    return streak


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
        tech_str = stock.get("tech_summary", "")
        context_blocks.append(
            f"[{name} ({code})]\n"
            f"최신 뉴스: {news_str}\n"
            f"3년 재무 트렌드: {trend_str}"
            + (f"\n기술적 지표: {tech_str}" if tech_str else "")
        )

    prompt = f"""아래는 AI가 선정한 10개 한국 주식 종목과 각 종목의 실제 최신 뉴스, 3년 재무 트렌드, 기술적 지표입니다.

=== 종목별 실제 데이터 ===
{chr(10).join(context_blocks)}

=== 기존 분석 JSON ===
{json.dumps({"stocks": analyzed}, ensure_ascii=False)}

위 실제 데이터를 반영해 각 종목의 분석을 보강하세요:
1. news_summary → 실제 뉴스 헤드라인을 반영해 업데이트 (1-2문장)
2. reason → 3년 재무 트렌드 + 기술적 지표(RSI·MA·MACD 신호) 반영해 보강

trend_summary, earnings_trend, future_target, stop_loss, investment_horizon, roe, operating_margin, debt_ratio, rsi, ma_signal, macd_signal 등 나머지 필드는 모두 기존 값을 그대로 유지하세요.

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
            # Gemini가 덮어쓰면 안 되는 필드 원본값으로 복원
            _PRESERVE = {
                "buy_score", "buy_score_label", "consecutive_days",
                "earnings_trend", "trend_summary",
                "roe", "operating_margin", "debt_ratio",
                "rsi", "rsi_signal", "ma20", "ma60", "ma_signal",
                "macd_hist", "macd_signal", "tech_summary",
                "volume_ratio", "volume_signal",
                "dividend_history",
                "future_target", "stop_loss", "investment_horizon",
                "current_price_raw", "week52_high", "week52_low", "week52_pct_from_high",
            }
            orig_by_code = {s.get("code"): s for s in analyzed}
            for s in stocks:
                orig = orig_by_code.get(s.get("code"), {})
                for key in _PRESERVE:
                    if key in orig:
                        s[key] = orig[key]
            print(f"  2차 AI 분석 완료: {len(stocks)}개 종목 보강")
            return stocks
    except Exception as e:
        print(f"  2차 AI 분석 실패 (기존 결과 유지): {e}")
    return analyzed


# ── 메인 분석 함수 ─────────────────────────────────────────────

def run_korean() -> dict:
    print("\n🇰🇷 한국 주식 분석 시작...")
    stocks_raw = fetch_kospi_stocks(top_n=100)
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
            # trend_summary를 Gemini에 의존하지 않고 직접 세팅
            if trend:
                stock["trend_summary"] = _format_trend_str(trend)
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

    # 기술적 지표 수집 (RSI · 이동평균 · MACD)
    print("  기술적 지표 수집 중...")
    for stock in analyzed:
        tech = _fetch_technical_indicators(stock.get("code", ""))
        if tech:
            stock.update(tech)
            print(f"    {stock['name']}: {tech.get('tech_summary', '')}")

    # 배당 히스토리 수집 (3년)
    print("  배당 히스토리 수집 중...")
    for stock in analyzed:
        div = _fetch_dividend_history(stock.get("code", ""))
        if div:
            stock.update(div)
            print(f"    {stock['name']}: {div.get('dividend_history', '')}")

    # 2차 AI 분석 (뉴스 + 트렌드 반영)
    print("  2차 AI 분석 (뉴스·트렌드 반영) 중...")
    analyzed = _refine_with_context(analyzed, news_map, trend_map)

    # 외국인·기관 수급 실데이터 덮어쓰기 (pykrx)
    print("  외국인·기관 수급 수집 중...")
    for stock in analyzed:
        trend = _fetch_investor_trend(stock.get("code", ""))
        if trend:
            stock["foreign_trend"] = trend
            print(f"    {stock['name']}: {trend}")

    # 종합 매수 점수 계산
    print("  종합 매수 점수 계산 중...")
    for stock in analyzed:
        stock.update(_calculate_buy_score(stock, market="kr"))
        print(f"    {stock['name']}: {stock['buy_score']}점 ({stock['buy_score_label']})")

    overlaps = compare_with_previous(analyzed, prev_report)
    save_report(analyzed)

    # 투자 추적 업데이트
    _update_tracking(analyzed, "api/data/track_kr.json")

    # 연속 추천 횟수 (트래킹 기록 후 계산해야 오늘 포함)
    for stock in analyzed:
        days = _count_consecutive_recs(stock.get("code", ""), "api/data/track_kr.json")
        if days >= 2:
            stock["consecutive_days"] = days
            print(f"    {stock['name']}: {days}일 연속 추천")

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
    stocks_raw = fetch_sp500_stocks(top_n=97)
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

    # 기술적 지표 수집 (RSI · 이동평균 · MACD)
    print("  기술적 지표 수집 중...")
    for stock in analyzed:
        tech = _fetch_technical_indicators(stock.get("code", ""), market="us")
        if tech:
            stock.update(tech)
            print(f"    {stock['name']}: {tech.get('tech_summary', '')}")

    # 배당 히스토리 수집 (3년)
    print("  배당 히스토리 수집 중...")
    for stock in analyzed:
        div = _fetch_dividend_history(stock.get("code", ""), market="us")
        if div:
            stock.update(div)
            print(f"    {stock['name']}: {div.get('dividend_history', '')}")

    # 종합 매수 점수 계산
    print("  종합 매수 점수 계산 중...")
    for stock in analyzed:
        stock.update(_calculate_buy_score(stock, market="us"))
        print(f"    {stock['name']}: {stock['buy_score']}점 ({stock['buy_score_label']})")

    # 미국주 투자 추적 업데이트
    _update_tracking(analyzed, "api/data/track_us.json", market="us")

    # 연속 추천 횟수
    for stock in analyzed:
        days = _count_consecutive_recs(stock.get("code", ""), "api/data/track_us.json")
        if days >= 2:
            stock["consecutive_days"] = days
            print(f"    {stock['name']}: {days}일 연속 추천")

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
