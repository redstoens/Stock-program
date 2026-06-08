import os
import json
import re
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"


def _yf_search(q: str) -> str | None:
    """yfinance Search로 ticker 탐색"""
    try:
        quotes = yf.Search(q, max_results=6, news_count=0).quotes
        if quotes:
            sym = quotes[0].get("symbol", "")
            return sym if sym else None
    except Exception:
        pass
    return None


def _gemini_ticker(query: str, client: genai.Client) -> str | None:
    """Gemini에게 ticker 심볼 질의 (yfinance fallback)"""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=(
                f'주식 "{query}"의 yfinance ticker 심볼을 반환하세요. '
                '한국 주식은 "종목코드.KS" 또는 "종목코드.KQ" 형식 (예: 005930.KS), '
                '미국 주식은 알파벳 티커 (예: AAPL). '
                '정확히 모르면 null로 반환. '
                '{"ticker": null}'
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        sym = json.loads(resp.text).get("ticker")
        return sym if sym and str(sym).lower() != "null" else None
    except Exception:
        return None


def _resolve_ticker(query: str, client: genai.Client) -> str | None:
    q = query.strip()

    # 한국 6자리 숫자 코드
    if re.match(r"^\d{5,6}$", q):
        for suffix in (".KS", ".KQ"):
            try:
                t = yf.Ticker(q + suffix)
                if t.fast_info.last_price:
                    return q + suffix
            except Exception:
                continue

    # yfinance Search + Gemini 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_yf = ex.submit(_yf_search, q)
        f_gm = ex.submit(_gemini_ticker, q, client)
        yf_sym = None
        gm_sym = None
        try:
            yf_sym = f_yf.result(timeout=6)
        except Exception:
            pass
        try:
            gm_sym = f_gm.result(timeout=8)
        except Exception:
            pass

    return yf_sym or gm_sym


def _fetch_yf_data(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    fi = ticker.fast_info

    data: dict = {
        "symbol":      symbol,
        "price":       round(float(fi.last_price or 0), 2),
        "prev_close":  round(float(fi.previous_close or 0), 2),
        "week52_high": round(float(fi.year_high or 0), 2),
        "week52_low":  round(float(fi.year_low or 0), 2),
        "market_cap":  fi.market_cap,
        "volume":      fi.three_month_average_volume,
    }

    try:
        hist = ticker.history(period="3mo")
        if not hist.empty:
            closes = hist["Close"]
            data["ma20"] = round(float(closes.tail(20).mean()), 2)
            data["ma60"] = round(float(closes.tail(60).mean()), 2) if len(closes) >= 60 else None
            delta = closes.diff().dropna()
            gain  = delta.clip(lower=0).tail(14).mean()
            loss  = (-delta.clip(upper=0)).tail(14).mean()
            data["rsi"] = round(float(100 - (100 / (1 + gain / loss))), 1) if loss > 0 else 50.0
            pct = (data["price"] - data["prev_close"]) / data["prev_close"] * 100 if data["prev_close"] else 0
            data["change_pct"] = round(pct, 2)
    except Exception:
        pass

    return data


def analyze_single_stock(query: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 없습니다")

    client = genai.Client(api_key=api_key)

    # ticker 탐색 (yfinance + Gemini fallback 병렬)
    symbol  = _resolve_ticker(query, client)
    yf_data = {}
    if symbol:
        try:
            yf_data = _fetch_yf_data(symbol)
        except Exception:
            pass

    lines = []
    if yf_data.get("price"):
        lines.append(f"현재가: {yf_data['price']}")
    if yf_data.get("change_pct") is not None:
        lines.append(f"전일비: {yf_data['change_pct']:+.2f}%")
    if yf_data.get("week52_high"):
        lines.append(f"52주 고가: {yf_data['week52_high']}  52주 저가: {yf_data['week52_low']}")
    if yf_data.get("rsi"):
        lines.append(f"RSI(14): {yf_data['rsi']}")
    if yf_data.get("ma20"):
        ma60 = f"  MA60: {yf_data['ma60']}" if yf_data.get("ma60") else ""
        lines.append(f"MA20: {yf_data['ma20']}{ma60}")
    if yf_data.get("market_cap"):
        lines.append(f"시가총액: {yf_data['market_cap']:,.0f}")

    data_section = "\n".join(lines) if lines else "(실시간 데이터 없음 — Gemini 학습 지식 기반 분석)"

    prompt = f"""사용자가 다음 주식의 실시간 분석을 요청했습니다: "{query}"

실시간 데이터:
{data_section}

투자자 관점에서 종합 분석하고 아래 JSON 형식으로만 반환하세요.
실시간 데이터가 없으면 학습 지식을 바탕으로 최선의 추정치를 제공하세요.
{{
  "name": "종목명 (한국어 또는 영문 공식명)",
  "code": "종목코드 또는 티커심볼",
  "market": "KR 또는 US",
  "sector": "섹터명",
  "current_price": "현재가 표시 (예: 85,400원 또는 $195.30, 모르면 '정보없음')",
  "change_pct": "전일 대비 등락률 (예: +1.2% 또는 -0.8%, 모르면 '')",
  "summary": "현재 상황 및 투자 포인트 요약 2~3문장",
  "valuation": "저평가 / 적정가 / 고평가 중 하나 + 한 줄 근거",
  "target_price": "12개월 목표주가",
  "upside_pct": "현재가 대비 상승여력 (예: +23.0%)",
  "stop_loss": "손절 기준가 및 비율",
  "investment_horizon": "단기 또는 중장기",
  "sentiment": "긍정 / 부정 / 중립 중 하나",
  "strength": ["강점1", "강점2", "강점3"],
  "risk": ["리스크1", "리스크2"],
  "technical": "기술적 분석 한 줄 (실시간 데이터 없으면 최근 동향 기술)",
  "action": "매수 또는 관망 또는 매도",
  "action_reason": "추천 근거 한 줄 (20자 이내)",
  "data_note": "실시간데이터 있으면 '' , 없으면 '학습 지식 기반 분석'"
}}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    result = json.loads(response.text)
    result["yf_data"] = yf_data
    return result
