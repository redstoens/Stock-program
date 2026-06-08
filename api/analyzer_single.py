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
    try:
        quotes = yf.Search(q, max_results=6, news_count=0).quotes
        if quotes:
            sym = quotes[0].get("symbol", "")
            return sym if sym else None
    except Exception:
        pass
    return None


def _gemini_ticker(query: str, client: genai.Client) -> str | None:
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=(
                f'주식 "{query}"의 yfinance ticker 심볼을 반환하세요. '
                '한국 주식은 "종목코드.KS" 또는 "종목코드.KQ" (예: 005930.KS), '
                '미국 주식은 알파벳 티커 (예: AAPL). '
                '확실하지 않으면 null. {"ticker": null}'
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


def _validate_and_fetch(symbol: str) -> dict | None:
    """ticker로 yfinance 데이터 조회 — 유효한 가격이 없으면 None 반환"""
    try:
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        price = float(fi.last_price or 0)
        if price == 0:
            return None

        data: dict = {
            "symbol":      symbol,
            "price":       round(price, 2),
            "prev_close":  round(float(fi.previous_close or 0), 2),
            "week52_high": round(float(fi.year_high or 0), 2),
            "week52_low":  round(float(fi.year_low or 0), 2),
            "market_cap":  fi.market_cap,
        }

        hist = ticker.history(period="3mo")
        if not hist.empty:
            closes = hist["Close"]
            data["ma20"] = round(float(closes.tail(20).mean()), 2)
            data["ma60"] = round(float(closes.tail(60).mean()), 2) if len(closes) >= 60 else None
            delta = closes.diff().dropna()
            gain  = delta.clip(lower=0).tail(14).mean()
            loss  = (-delta.clip(upper=0)).tail(14).mean()
            data["rsi"] = round(float(100 - (100 / (1 + gain / loss))), 1) if loss > 0 else 50.0
            if data["prev_close"]:
                data["change_pct"] = round((price - data["prev_close"]) / data["prev_close"] * 100, 2)

        return data
    except Exception:
        return None


def _resolve_ticker(query: str, client: genai.Client) -> str | None:
    q = query.strip()

    # 한국 6자리 숫자 코드 직접 시도
    if re.match(r"^\d{5,6}$", q):
        for suffix in (".KS", ".KQ"):
            d = _validate_and_fetch(q + suffix)
            if d:
                return q + suffix

    # yfinance Search + Gemini 병렬 탐색
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_yf = ex.submit(_yf_search, q)
        f_gm = ex.submit(_gemini_ticker, q, client)
        yf_sym, gm_sym = None, None
        try:
            yf_sym = f_yf.result(timeout=6)
        except Exception:
            pass
        try:
            gm_sym = f_gm.result(timeout=8)
        except Exception:
            pass

    # yfinance 결과 우선 검증
    for sym in filter(None, [yf_sym, gm_sym]):
        d = _validate_and_fetch(sym)
        if d:
            return sym

    return None


def _fmt_price(price: float, is_kr: bool) -> str:
    if is_kr:
        return f"{price:,.0f}원"
    return f"${price:,.2f}"


def analyze_single_stock(query: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 없습니다")

    client = genai.Client(api_key=api_key)

    symbol  = _resolve_ticker(query, client)
    yf_data = _validate_and_fetch(symbol) if symbol else None
    has_rt  = yf_data is not None  # 실시간 데이터 보유 여부

    # ── 가격 필드를 yfinance 값으로 직접 세팅 (Gemini가 덮어쓰지 못하도록) ──
    is_kr = bool(symbol and (symbol.endswith(".KS") or symbol.endswith(".KQ")))
    if has_rt:
        price_str  = _fmt_price(yf_data["price"], is_kr)
        chg_str    = f"{yf_data.get('change_pct', 0):+.2f}%" if yf_data.get("change_pct") is not None else ""
        w52h_str   = _fmt_price(yf_data["week52_high"], is_kr) if yf_data.get("week52_high") else ""
        w52l_str   = _fmt_price(yf_data["week52_low"],  is_kr) if yf_data.get("week52_low")  else ""

        data_lines = [
            f"현재가(실시간): {price_str}",
            f"전일비: {chg_str}" if chg_str else "",
            f"52주 고가: {w52h_str}  52주 저가: {w52l_str}" if w52h_str else "",
            f"RSI(14): {yf_data['rsi']}" if yf_data.get("rsi") else "",
            (f"MA20: {yf_data['ma20']}" + (f"  MA60: {yf_data['ma60']}" if yf_data.get("ma60") else ""))
                if yf_data.get("ma20") else "",
            f"시가총액: {yf_data['market_cap']:,.0f}" if yf_data.get("market_cap") else "",
        ]
        data_section = "\n".join(l for l in data_lines if l)
        price_instruction = (
            f"▶ current_price 필드에는 반드시 '{price_str}' 을 그대로 사용하세요. "
            f"target_price와 stop_loss는 현재가 {price_str} 기준으로 계산하세요."
        )
    else:
        data_section = "(실시간 시세 조회 불가)"
        price_instruction = (
            "▶ 실시간 가격 데이터가 없습니다. "
            "current_price 필드에는 '시세 조회 불가'로 표시하고, "
            "target_price·stop_loss는 학습 지식 기반 추정임을 괄호로 명시하세요."
        )

    prompt = f"""사용자가 다음 주식의 분석을 요청했습니다: "{query}"

실시간 데이터:
{data_section}

{price_instruction}

투자자 관점에서 종합 분석하고 아래 JSON 형식으로만 반환하세요:
{{
  "name": "종목명 (한국어 또는 영문 공식명)",
  "code": "종목코드 또는 티커",
  "market": "KR 또는 US",
  "sector": "섹터명",
  "current_price": "위 지시에 따른 현재가",
  "change_pct": "전일비 (실시간 없으면 빈 문자열)",
  "summary": "현재 상황 및 투자 포인트 2~3문장",
  "valuation": "저평가 / 적정가 / 고평가 + 한 줄 근거",
  "target_price": "12개월 목표주가",
  "upside_pct": "현재가 대비 상승여력",
  "stop_loss": "손절 기준가 및 비율",
  "investment_horizon": "단기 또는 중장기",
  "sentiment": "긍정 / 부정 / 중립",
  "strength": ["강점1", "강점2", "강점3"],
  "risk": ["리스크1", "리스크2"],
  "technical": "기술적 분석 한 줄",
  "action": "매수 또는 관망 또는 매도",
  "action_reason": "추천 근거 20자 이내",
  "data_note": "{'' if has_rt else '실시간 시세 없음 — 분석은 AI 학습 지식 기반'}"
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

    # 가격 필드는 yfinance 값으로 강제 덮어쓰기
    if has_rt:
        result["current_price"] = price_str
        result["change_pct"]    = chg_str
        result["_realtime"]     = True
    else:
        result["_realtime"] = False

    result["yf_data"] = yf_data or {}
    return result
