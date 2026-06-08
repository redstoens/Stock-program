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
_KR_SUFFIXES = (".KS", ".KQ")


def _clean_ticker(sym: str) -> str:
    if not sym:
        return ""
    upper = sym.strip().upper()
    for kr in _KR_SUFFIXES:
        if upper.endswith(kr):
            return upper
    return upper.split(".")[0]


def _validate_and_fetch(symbol: str) -> dict | None:
    if not symbol:
        return None
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

        # 기본 재무지표
        try:
            info = ticker.info
            per = info.get("trailingPE") or info.get("forwardPE")
            data["per"]              = round(float(per), 1) if per else None
            pbr = info.get("priceToBook")
            data["pbr"]              = round(float(pbr), 2) if pbr else None
            roe = info.get("returnOnEquity")
            data["roe"]              = round(float(roe) * 100, 1) if roe else None
            op_margin = info.get("operatingMargins")
            data["operating_margin"] = round(float(op_margin) * 100, 1) if op_margin else None
            div = info.get("dividendYield")
            data["dividend_yield"]   = round(float(div) * 100, 2) if div else None
        except Exception:
            pass

        # 다음 실적발표일 — 미래 날짜만 사용
        try:
            import datetime
            today = datetime.date.today()

            def _parse_date(val):
                if val is None:
                    return None
                if hasattr(val, "date"):        # datetime/Timestamp
                    return val.date()
                if hasattr(val, "year"):        # date
                    return val
                ts = float(val)                 # Unix timestamp
                return datetime.date.fromtimestamp(ts)

            # ticker.calendar 시도
            cal = ticker.calendar
            candidates = []
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    for v in (ed if isinstance(ed, (list, tuple)) else [ed]):
                        d = _parse_date(v)
                        if d and d >= today:
                            candidates.append(d)
            # ticker.earnings_dates 시도 (DataFrame)
            try:
                ed_df = ticker.earnings_dates
                if ed_df is not None and not ed_df.empty:
                    for idx in ed_df.index:
                        d = _parse_date(idx)
                        if d and d >= today:
                            candidates.append(d)
            except Exception:
                pass

            if candidates:
                next_ed = min(candidates)
                data["next_earnings"] = next_ed.strftime("%Y년 %m월 %d일")
        except Exception:
            pass

        return data
    except Exception:
        return None


def _gemini_ticker(query: str, client: genai.Client) -> str | None:
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=(
                f'주식 "{query}"의 yfinance ticker를 반환하세요.\n'
                '규칙:\n'
                '- 한국 주식: "종목코드.KS" 또는 "종목코드.KQ" (예: 005930.KS)\n'
                '- 미국 주식: NYSE/NASDAQ 티커 알파벳만 (예: AAPL) — .US .O 등 접미사 금지\n'
                '- 확실하지 않으면 null\n'
                '{"ticker": null}'
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        sym = json.loads(resp.text).get("ticker")
        return _clean_ticker(str(sym)) if sym and str(sym).lower() != "null" else None
    except Exception:
        return None


def _resolve_ticker(query: str, client: genai.Client) -> str | None:
    q = query.strip()

    if re.match(r"^\d{5,6}$", q):
        for suffix in _KR_SUFFIXES:
            d = _validate_and_fetch(q + suffix)
            if d:
                return q + suffix

    if re.match(r"^[A-Z]{1,5}$", q):
        d = _validate_and_fetch(q)
        if d:
            return q

    has_korean = bool(re.search(r"[가-힣]", q))
    if has_korean:
        sym = _gemini_ticker(q, client)
        if sym:
            d = _validate_and_fetch(sym)
            if d:
                return sym
        return None

    def _yf_search_us(q: str) -> str | None:
        try:
            quotes = yf.Search(q, max_results=8, news_count=0).quotes
            for r in quotes:
                sym = r.get("symbol", "")
                exchange = r.get("exchange", "")
                if sym and exchange in ("NMS", "NYQ", "NGM", "PCX", "ASE"):
                    return sym
            if quotes:
                return _clean_ticker(quotes[0].get("symbol", ""))
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_yf = ex.submit(_yf_search_us, q)
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

    for sym in filter(None, [yf_sym, gm_sym]):
        d = _validate_and_fetch(sym)
        if d:
            return sym

    return None


def _fmt_price(price: float, is_kr: bool) -> str:
    return f"{price:,.0f}원" if is_kr else f"${price:,.2f}"


def analyze_single_stock(query: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 없습니다")

    client = genai.Client(api_key=api_key)
    symbol  = _resolve_ticker(query, client)
    yf_data = _validate_and_fetch(symbol) if symbol else None
    has_rt  = yf_data is not None
    is_kr   = bool(symbol and symbol.upper().endswith(_KR_SUFFIXES))

    if has_rt:
        price_str = _fmt_price(yf_data["price"], is_kr)
        chg_str   = (f"{yf_data['change_pct']:+.2f}%"
                     if yf_data.get("change_pct") is not None else "")

        fund_lines = []
        if yf_data.get("per"):          fund_lines.append(f"PER: {yf_data['per']}x")
        if yf_data.get("pbr"):          fund_lines.append(f"PBR: {yf_data['pbr']}x")
        if yf_data.get("roe"):          fund_lines.append(f"ROE: {yf_data['roe']}%")
        if yf_data.get("operating_margin"): fund_lines.append(f"영업이익률: {yf_data['operating_margin']}%")
        if yf_data.get("dividend_yield"):   fund_lines.append(f"배당수익률: {yf_data['dividend_yield']}%")

        next_earnings_rt = yf_data.get("next_earnings")  # 실시간 실적발표일

        data_lines = [
            f"ticker: {symbol}",
            f"현재가(실시간): {price_str}",
            f"전일비: {chg_str}" if chg_str else "",
            f"52주 고가: {_fmt_price(yf_data['week52_high'], is_kr)}  "
            f"52주 저가: {_fmt_price(yf_data['week52_low'], is_kr)}"
                if yf_data.get("week52_high") else "",
            f"RSI(14): {yf_data['rsi']}" if yf_data.get("rsi") else "",
            (f"MA20: {yf_data['ma20']}" + (f"  MA60: {yf_data['ma60']}" if yf_data.get("ma60") else ""))
                if yf_data.get("ma20") else "",
            f"시가총액: {yf_data['market_cap']:,.0f}" if yf_data.get("market_cap") else "",
            "재무지표(yfinance): " + ", ".join(fund_lines) if fund_lines else "",
            f"다음 실적발표일(yfinance 실시간): {next_earnings_rt} ← 이 값을 next_earnings 필드에 그대로 사용"
                if next_earnings_rt else "",
        ]
        data_section = "\n".join(l for l in data_lines if l)
        price_instruction = (
            f"▶ current_price 는 반드시 '{price_str}' 을 그대로 사용하세요. "
            f"target_price·stop_loss 는 현재가 {price_str} 기준으로 계산하세요. "
            f"yfinance 재무지표가 있으면 그 값을 우선 사용하고, 없으면 학습 지식으로 추정하세요."
            + (f" 다음 실적발표일은 반드시 '{next_earnings_rt}' 을 사용하세요." if next_earnings_rt else
               " 다음 실적발표일은 학습 지식 기반 추정이므로 불확실하면 '확인 필요'로 표시하세요.")
        )
        data_note = ""
    else:
        data_section = f"ticker 조회 실패 (검색어: {query})"
        price_instruction = (
            "▶ 실시간 가격 없음. current_price 는 '시세 조회 불가' 로 표시하고, "
            "수치는 학습 지식 기반 추정임을 명시하세요."
        )
        data_note = "실시간 시세 없음 — AI 학습 지식 기반 분석"

    prompt = f"""사용자가 다음 주식의 상세 분석을 요청했습니다: "{query}"

실시간 데이터:
{data_section}

{price_instruction}

투자자 관점에서 아래 JSON 형식으로 최대한 상세하게 분석하세요:
{{
  "name": "종목 공식명",
  "code": "종목코드 또는 티커",
  "market": "KR 또는 US",
  "sector": "섹터명",
  "current_price": "위 지시에 따른 현재가",
  "change_pct": "전일비",

  "summary": "현재 상황 및 핵심 투자 포인트 3~4문장 (구체적 수치 포함)",

  "per":              "PER 수치 (예: 18.5x, 없으면 'N/A')",
  "pbr":              "PBR 수치 (예: 1.8x)",
  "roe":              "ROE (예: 15.2%)",
  "operating_margin": "영업이익률 (예: 12.3%)",
  "dividend_yield":   "배당수익률 (예: 2.1%, 없으면 '무배당')",
  "earnings_trend":   "최근 실적 트렌드: 개선 / 보합 / 악화 + 한 줄 근거",
  "next_earnings":    "다음 실적발표 예상 시기 (예: 2025년 7월)",
  "supply_demand":    "외국인·기관 수급 동향 한 줄 (예: '외국인 3주 연속 순매수, 기관 중립')",

  "valuation":         "저평가 / 적정가 / 고평가 + 근거",
  "target_price":      "12개월 목표주가",
  "upside_pct":        "상승여력 (예: +23.0%)",
  "stop_loss":         "손절 기준가 및 비율",
  "investment_horizon":"단기 또는 중장기",
  "action":            "매수 또는 관망 또는 매도",
  "action_reason":     "추천 근거 20자 이내",
  "sentiment":         "긍정 / 부정 / 중립",

  "strength": ["강점1", "강점2", "강점3"],
  "risk":     ["리스크1", "리스크2", "리스크3"],

  "catalysts": ["단기 모멘텀/촉매1", "촉매2", "촉매3"],

  "scenario_bull": "낙관 시나리오: 조건 + 예상 주가 (예: AI 수요 급증 시 $280 +43%)",
  "scenario_base": "중립 시나리오: 현 추세 유지 시 예상 주가",
  "scenario_bear": "비관 시나리오: 리스크 현실화 시 예상 주가",

  "technical": "기술적 분석 2줄 (MA 위치·RSI·지지저항 포함)",

  "entry_price":    "추천 진입가 — 기술적 지지선·눌림목 기준 구체적 가격 (예: 83,500원, $188.00). 즉시 진입이 유리하면 현재가 근처 값",
  "entry_timing":   "진입 타이밍: '즉시 진입' / '눌림목 대기' / '조건부 진입' 중 하나 + 한 줄 이유",
  "entry_strategy": "분할 매수 전략 2~3단계 (예: '1차 83,500원 50% → 2차 80,000원 30% → 3차 76,000원 20%'). 관망·매도면 '해당 없음'",

  "data_note": "{data_note}"
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

    if has_rt:
        result["current_price"] = price_str
        result["change_pct"]    = chg_str
        result["_realtime"]     = True
        # yfinance 실데이터로 덮어쓰기
        for field, key in [("per","per"),("pbr","pbr"),("roe","roe"),
                           ("operating_margin","operating_margin"),
                           ("dividend_yield","dividend_yield")]:
            if yf_data.get(key) is not None:
                unit = "%" if key in ("roe","operating_margin","dividend_yield") else "x"
                result[field] = f"{yf_data[key]}{unit}"
        # 다음 실적발표일 — yfinance 실시간 값으로 강제 덮어쓰기
        if yf_data.get("next_earnings"):
            result["next_earnings"] = yf_data["next_earnings"]
    else:
        result["_realtime"] = False

    result["yf_data"] = yf_data or {}
    return result
