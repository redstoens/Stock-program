from datetime import date


DISCLAIMER = (
    "\n---\n"
    "> **주의**: 본 리포트는 정보 제공 목적이며 투자 추천이 아닙니다. "
    "최종 투자 판단은 본인의 책임입니다."
)

RISK_EMOJI = {"하": "🟢", "중": "🟡", "상": "🔴"}
TREND_EMOJI = {"개선": "📈", "보합": "➡️", "악화": "📉"}


def build_report(stocks: list[dict], memo: str = "", generated_at: date | None = None) -> str:
    if generated_at is None:
        generated_at = date.today()

    lines = [
        "# 저평가 가치주 분석 리포트",
        "",
        f"- **생성일**: {generated_at.strftime('%Y-%m-%d')}",
    ]
    if memo:
        lines.append(f"- **분석 메모**: {memo}")
    lines += ["", "---", ""]

    for i, s in enumerate(stocks, start=1):
        risk = s.get("risk_level", "")
        trend = s.get("earnings_trend", "")
        trend_key = trend.split()[0] if trend else ""

        lines += [
            f"## 【종목 {i}】 {s.get('name', '')} ({s.get('code', '')})",
            "",
            f"| 항목 | 내용 |",
            f"|------|------|",
            f"| 유형 | {s.get('stock_type', '-')} |",
            f"| PER / ROE | {s.get('per', '-')} / {s.get('roe', '-')}% |",
            f"| 배당수익률 | {s.get('dividend_yield', '-')} |",
            f"| 실적 트렌드 | {TREND_EMOJI.get(trend_key, '')} {trend} |",
            f"| 외국인·기관 수급 | {s.get('foreign_trend', '-')} |",
            f"| 리스크 등급 | {RISK_EMOJI.get(risk, '')} {risk} — {s.get('risk_reason', '')} |",
            f"| 52주 최고가 | {s.get('week52_high', 'N/A')} ({s.get('week52_pct_from_high', 'N/A')}) |",
            f"| 52주 최저가 | {s.get('week52_low', 'N/A')} |",
            f"| 🎯 12개월 목표주가 | {s.get('future_target', '-')} (상승여력 {s.get('upside_pct', '-')}) |",
            f"| 🛑 손절 기준가 | {s.get('stop_loss', '-')} |",
            "",
            f"**선정 이유**: {s.get('reason', '')}",
            "",
            f"**관련 뉴스**: {s.get('news_summary', '')}",
            "",
            f"**기업가치 평가**: {s.get('valuation', '')}",
            "",
            "---",
            "",
        ]

    lines.append(DISCLAIMER)
    return "\n".join(lines)
