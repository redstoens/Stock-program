import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

SP500_TICKERS = [
    # ── 빅테크 / AI ──────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B",
    "TSLA", "AVGO", "ORCL", "ADBE", "NOW", "INTU", "PANW", "CSCO",
    "INTC", "AMD", "TXN", "QCOM", "IBM", "ANET",
    # ── 금융 ─────────────────────────────────────────────────────
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SPGI",
    "AXP", "C", "COF", "SCHW", "USB", "PNC",
    # ── 헬스케어 / 제약 ──────────────────────────────────────────
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "ABT", "TMO", "DHR",
    "ISRG", "BSX", "MDT", "REGN", "GILD", "SYK", "PFE", "AMGN",
    "ELV", "ZTS",
    # ── 소비재 / 유통 ─────────────────────────────────────────────
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "PM",
    "TGT", "SBUX", "NKE", "CMG", "LOW",
    # ── 에너지 ───────────────────────────────────────────────────
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    # ── 산업재 / 방산 ─────────────────────────────────────────────
    "GE", "RTX", "HON", "CAT", "LMT", "BA", "UPS", "FDX", "DE",
    "ACN", "ETN", "EMR",
    # ── 통신 / 미디어 ─────────────────────────────────────────────
    "NFLX", "CRM", "LIN",
    # ── 유틸리티 / 리츠 ──────────────────────────────────────────
    "NEE", "DUK", "SO", "AMT", "PLD", "EQIX",
    # ── 플랫폼 / 기타 ─────────────────────────────────────────────
    "PYPL", "UBER", "ABNB",
]


def _fetch_one(ticker_sym: str) -> dict | None:
    try:
        info = yf.Ticker(ticker_sym).info
        current = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        week52_high = info.get("fiftyTwoWeekHigh") or 0
        week52_low = info.get("fiftyTwoWeekLow") or 0

        pct_from_high = ""
        if week52_high and current:
            pct = round((current - week52_high) / week52_high * 100, 1)
            pct_from_high = f"{pct}%"

        per_val = info.get("trailingPE") or info.get("forwardPE")
        roe_val = info.get("returnOnEquity")
        pbr_val = info.get("priceToBook")

        return {
            "name": info.get("shortName") or info.get("longName") or ticker_sym,
            "code": ticker_sym,
            "per": round(per_val, 1) if per_val else "N/A",
            "roe": round(roe_val * 100, 1) if roe_val else "N/A",
            "pbr": str(round(pbr_val, 2)) if pbr_val and pbr_val > 0 else "N/A",
            "market_cap": info.get("marketCap") or 0,
            "sector": info.get("sector") or "",
            "current_price_raw": round(current, 2),
            "week52_high": f"${week52_high:,.2f}" if week52_high else "",
            "week52_low": f"${week52_low:,.2f}" if week52_low else "",
            "week52_pct_from_high": pct_from_high,
        }
    except Exception:
        return None


def fetch_sp500_stocks(top_n: int = 97) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in SP500_TICKERS[:top_n]}
        for future in as_completed(futures):
            result = future.result()
            if result and result["market_cap"] > 0 and result["current_price_raw"] > 0:
                results.append(result)
    results.sort(key=lambda x: x["market_cap"], reverse=True)
    return results


def format_for_prompt_us(stocks: list[dict]) -> str:
    lines = ["종목명 | 티커 | PER | ROE(%) | 시가총액(십억달러) | 섹터 | 현재가"]
    for s in stocks:
        cap_b = round(s["market_cap"] / 1e9, 1)
        lines.append(
            f"{s['name']} | {s['code']} | {s['per']} | {s['roe']}"
            f" | ${cap_b}B | {s['sector']} | ${s['current_price_raw']}"
        )
    return "\n".join(lines)
