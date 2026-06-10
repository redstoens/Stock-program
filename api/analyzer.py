import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """당신은 대한민국 주식 시장 전문 가치투자 분석가입니다.
기업의 펀더멘털(PER, ROE, 시가총액 등)을 분석해 현재 주가가 기업 내재가치 대비 저평가된 종목을 선정합니다.

아래는 오늘 기준 KOSPI 시가총액 상위 종목 데이터입니다.
(컬럼 설명: RSI = 14일 RSI(30↓과매도·70↑과매수), MA배열 = 이동평균 배열(정배열↑/역배열↓/혼조), 52주고가대비 = 현재가가 52주 최고가 대비 몇% 위치)

{stock_table}

위 데이터를 분석해 아래 두 카테고리로 나누어 총 10개 종목을 선정하세요.

▶ 단기 투자 5개 (investment_horizon = "단기")
- 1~3개월 내 수익 실현 가능한 종목
- 선정 기준: 모멘텀(뉴스·실적 서프라이즈·수급 급변), 기술적 돌파 임박, 단기 촉매 존재
- PER·ROE보다 단기 주가 동력에 집중
- 기술적 우대: RSI 40~65 구간 + 정배열 종목 우선 고려

▶ 중장기 투자 5개 (investment_horizon = "중장기")
- 6개월 이상 보유에 적합한 종목
- 선정 기준 (우선순위 순):
  1. PER이 낮을수록 유리 (동종 업계 평균 이하, 단 음수·N/A는 실적 악화 신호로 감점)
  2. ROE가 높을수록 유리 (10% 이상이면 우수, 15% 이상이면 매우 우수)
  3. 시가총액이 충분히 크고 사업 안정성이 검증된 대형주
  4. 일시적 악재로 저평가된 종목 우대 (구조적 문제 종목 제외)
  5. 기술적 우대: RSI 30~50 구간(눌림목) + 52주 고가 대비 -20% 이상 하락 종목 우선 고려
{memo_section}

각 종목에 대해 당신이 알고 있는 최신 지식을 바탕으로 다음 항목도 함께 분석하세요:
- 외국인·기관 수급: 최근 외국인/기관의 순매수 또는 순매도 방향과 강도
- 배당 수익률: 최근 연간 배당수익률 (%)
- 실적 트렌드: 최근 3분기 영업이익 방향 (개선/보합/악화)
- 리스크 등급: 하(안전)·중(보통)·상(위험) 중 하나와 주요 리스크 요인

반드시 단기 5개 + 중장기 5개, 합계 정확히 10개를 반환하세요.
동일 종목이 두 카테고리에 중복되면 안 됩니다.

다음 JSON 형식으로 정확히 반환하세요 (다른 텍스트 없이 JSON만):
{{
  "stocks": [
    {{
      "name": "종목명",
      "code": "종목코드",
      "per": "PER 수치",
      "roe": "ROE 수치",
      "stock_type": "배당주·가치주·성장주·급등주·경기민감주 중 1개",
      "sector": "업종/섹터명 (예: 반도체, 바이오/제약, 금융, 자동차, 조선/방산, IT서비스, 소비재, 에너지/화학, 건설, 통신 등 간결하게)",
      "reason": "선정 근거 (단기: 모멘텀·촉매 / 중장기: PER·ROE·업종 비교 등 구체적 수치 포함, 2-3문장)",
      "news_summary": "최근 기업 동향 또는 선정 원인 요약 (1-2문장)",
      "valuation": "기업가치 한 줄 평가",
      "future_target": "12개월 예상 목표주가 — 업종 평균 PER·실적 성장률 기반으로 계산. '000,000원' 형식 (숫자만, 콤마 포함)",
      "upside_pct": "현재가 대비 목표주가 상승여력 — '+X.X%' 형식 (예: '+28.5%')",
      "stop_loss": "손절 기준가 — 핵심 지지선 기준. '000,000원 (현재가 대비 -X%)' 형식",
      "investment_horizon": "단기 또는 중장기 중 하나만",
      "hold_period": "예상 보유 기간 (예: '1~2개월', '6개월~1년', '1~2년')",
      "horizon_reason": "단기 또는 중장기로 분류한 핵심 이유 한 줄",
      "foreign_trend": "외국인·기관 수급 요약 (예: '외국인 3주 연속 순매수, 기관 중립')",
      "dividend_yield": "배당수익률 (예: '3.2%' 또는 '무배당')",
      "earnings_trend": "실적 트렌드 — '개선'·'보합'·'악화' 중 하나 + 한 줄 이유",
      "risk_level": "하·중·상 중 하나",
      "risk_reason": "주요 리스크 요인 한 줄"
    }}
  ]
}}

주의: 본 분석은 정보 제공 목적이며 투자 추천이 아닙니다."""


def analyze_stocks(stock_table: str, memo: str = "") -> list[dict]:
    """KOSPI 종목 데이터를 Gemini API로 분석해 저평가 가치주 5개를 반환한다."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = genai.Client(api_key=api_key)
    memo_section = f"5. 추가 요청: {memo}" if memo else ""
    prompt = PROMPT_TEMPLATE.format(stock_table=stock_table, memo_section=memo_section)

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
