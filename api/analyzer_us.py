import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE_US = """당신은 미국 주식 시장 전문 가치투자 분석가입니다.
S&P 500 대형주의 펀더멘털(PER, ROE, 시가총액 등)을 분석해 현재 주가가 기업 내재가치 대비 저평가된 종목을 선정합니다.

아래는 오늘 기준 S&P 500 주요 종목 데이터입니다.

{stock_table}

위 데이터를 분석해 아래 두 카테고리로 나누어 총 10개 종목을 선정하세요.

▶ 단기 투자 5개 (investment_horizon = "단기")
- 1~3개월 내 수익 실현 가능한 종목
- 선정 기준: 모멘텀(실적 서프라이즈·수급·뉴스), 기술적 돌파 임박, 단기 촉매 존재

▶ 중장기 투자 5개 (investment_horizon = "중장기")
- 6개월 이상 보유에 적합한 종목
- 선정 기준: 낮은 PER, 높은 ROE, 안정적 사업 모델, 장기 성장 스토리
{memo_section}

각 종목에 대해 당신이 알고 있는 최신 지식을 바탕으로 다음 항목도 함께 분석하세요:
- 기관·헤지펀드 수급: 최근 매수·매도 동향
- 배당 수익률: 최근 연간 배당수익률 (%)
- 실적 트렌드: 최근 3분기 EPS/매출 방향 (개선/보합/악화)
- 리스크 등급: 하(안전)·중(보통)·상(위험) 중 하나와 주요 리스크 요인

반드시 단기 5개 + 중장기 5개, 합계 정확히 10개를 반환하세요.
동일 종목이 두 카테고리에 중복되면 안 됩니다.

다음 JSON 형식으로 정확히 반환하세요 (다른 텍스트 없이 JSON만):
{{
  "stocks": [
    {{
      "name": "종목명 (영문)",
      "code": "티커심볼",
      "per": "PER 수치",
      "roe": "ROE 수치",
      "stock_type": "배당주·가치주·성장주·급등주·경기민감주 중 1개",
      "reason": "선정 근거 (단기: 모멘텀·촉매 / 중장기: PER·ROE·업종 비교 등 구체적 수치 포함, 2-3문장)",
      "news_summary": "최근 기업 동향 요약 (1-2문장)",
      "valuation": "기업가치 한 줄 평가",
      "future_target": "12개월 예상 목표주가 달러 (예: '$195.00')",
      "upside_pct": "현재가 대비 목표주가 상승여력 (예: '+18.5%')",
      "stop_loss": "손절 기준가 달러 (예: '$155.00 (현재가 대비 -8%)')",
      "investment_horizon": "단기 또는 중장기 중 하나만",
      "hold_period": "예상 보유 기간 (예: '1~2개월', '6개월~1년')",
      "horizon_reason": "단기 또는 중장기로 분류한 핵심 이유 한 줄",
      "foreign_trend": "기관·헤지펀드 수급 요약",
      "dividend_yield": "배당수익률 (예: '1.8%' 또는 '무배당')",
      "earnings_trend": "실적 트렌드 — '개선'·'보합'·'악화' 중 하나 + 한 줄 이유",
      "risk_level": "하·중·상 중 하나",
      "risk_reason": "주요 리스크 요인 한 줄"
    }}
  ]
}}

주의: 본 분석은 정보 제공 목적이며 투자 추천이 아닙니다."""


def analyze_stocks_us(stock_table: str, memo: str = "") -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    client = genai.Client(api_key=api_key)
    memo_section = f"5. 추가 요청: {memo}" if memo else ""
    prompt = PROMPT_TEMPLATE_US.format(stock_table=stock_table, memo_section=memo_section)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        data = json.loads(response.text)
        stocks = data.get("stocks", [])
        if not stocks:
            raise RuntimeError("AI 분석 결과에서 종목을 찾을 수 없습니다.")
        return stocks[:10]
    except RuntimeError:
        raise
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "credits" in err_str.lower() or "prepayment" in err_str.lower():
            raise RuntimeError("AI 분석 서비스를 일시적으로 사용할 수 없습니다. Gemini API 크레딧이 소진되었습니다. Google AI Studio(aistudio.google.com)에서 크레딧을 충전해주세요.")
        raise RuntimeError(f"AI 분석 중 오류가 발생했습니다: {err_str}")
