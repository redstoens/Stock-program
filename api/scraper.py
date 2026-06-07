from datetime import date, timedelta
from pykrx import stock as krx


def _find_trading_days(n: int = 2) -> list[str]:
    """최근 n개 영업일(YYYYMMDD) 반환, 최신일이 index 0."""
    today = date.today()
    found = []
    for offset in range(1, 15):
        d = (today - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            if krx.get_market_ticker_list(d, market="KOSPI"):
                found.append(d)
                if len(found) >= n:
                    break
        except Exception:
            continue
    return found


def fetch_kospi_stocks(top_n: int = 80) -> list[dict]:
    """KOSPI 시가총액 상위 종목 — KRX 공식 데이터(pykrx)."""
    trading_days = _find_trading_days(2)
    if not trading_days:
        raise RuntimeError("최근 영업일을 찾을 수 없습니다.")

    trade_date = trading_days[0]
    prev_date = trading_days[1] if len(trading_days) > 1 else None

    # 시가총액 (정렬 기준)
    cap_df = krx.get_market_cap(trade_date, market="KOSPI")
    if cap_df.empty:
        raise RuntimeError("시가총액 데이터를 가져올 수 없습니다.")
    cap_df = cap_df.sort_values("시가총액", ascending=False)

    # 종가 (오늘 / 전일 — 등락률 계산용)
    ohlcv_today = krx.get_market_ohlcv_by_ticker(trade_date, market="KOSPI")
    ohlcv_prev = krx.get_market_ohlcv_by_ticker(prev_date, market="KOSPI") if prev_date else None

    # PER · EPS · BPS (ROE = EPS/BPS × 100)
    fund_df = krx.get_market_fundamental(trade_date, market="KOSPI")

    stocks = []
    for i, ticker in enumerate(cap_df.index[:top_n], start=1):
        name = krx.get_market_ticker_name(ticker)

        # 현재가 · 등락률
        cur_price, change_rate = 0, ""
        if ticker in ohlcv_today.index:
            cur_price = int(ohlcv_today.loc[ticker, "종가"])
            if ohlcv_prev is not None and ticker in ohlcv_prev.index:
                prev_close = int(ohlcv_prev.loc[ticker, "종가"])
                if prev_close > 0:
                    chg = (cur_price - prev_close) / prev_close * 100
                    change_rate = f"{chg:+.2f}%"

        # 시가총액 (억원)
        mktcap_억 = int(cap_df.loc[ticker, "시가총액"]) // 100_000_000

        # PER · ROE
        per, roe = "N/A", "N/A"
        if ticker in fund_df.index:
            row_f = fund_df.loc[ticker]
            per_val = float(row_f.get("PER", 0) or 0)
            eps = float(row_f.get("EPS", 0) or 0)
            bps = float(row_f.get("BPS", 0) or 0)
            if per_val > 0:
                per = str(round(per_val, 1))
            if bps > 0:
                roe = str(round(eps / bps * 100, 1))

        stocks.append({
            "rank": i,
            "name": name,
            "code": ticker,
            "price": f"{cur_price:,}" if cur_price else "N/A",
            "change_rate": change_rate,
            "market_cap": f"{mktcap_억:,}",
            "per": per,
            "roe": roe,
            "current_price_raw": cur_price,
        })

    return stocks


def fetch_stock_detail(code: str) -> dict:
    """개별 종목 52주 고저가 — pykrx KRX 데이터."""
    detail = {
        "current_price_raw": 0,
        "week52_high": "N/A",
        "week52_low": "N/A",
        "week52_pct_from_high": "N/A",
    }
    try:
        today_str = date.today().strftime("%Y%m%d")
        year_ago = (date.today() - timedelta(days=365)).strftime("%Y%m%d")

        ohlcv = krx.get_market_ohlcv_by_date(year_ago, today_str, code)
        if ohlcv.empty:
            return detail

        w52_high = int(ohlcv["고가"].max())
        w52_low = int(ohlcv["저가"].min())
        cur_price = int(ohlcv["종가"].iloc[-1])
        pct = round((cur_price - w52_high) / w52_high * 100, 1) if w52_high > 0 else 0

        detail.update({
            "current_price_raw": cur_price,
            "week52_high": f"{w52_high:,}",
            "week52_low": f"{w52_low:,}",
            "week52_pct_from_high": f"{pct:+.1f}%",
        })
    except Exception:
        pass
    return detail


def format_for_prompt(stocks: list[dict]) -> str:
    """AI 프롬프트 테이블 문자열."""
    lines = ["순위 | 종목명 | 종목코드 | 현재가 | 등락률 | 시가총액(억) | PER | ROE(%)"]
    lines.append("-" * 80)
    for s in stocks:
        lines.append(
            f"{s['rank']} | {s['name']} | {s['code']} | {s['price']} | "
            f"{s['change_rate']} | {s['market_cap']} | {s['per']} | {s['roe']}"
        )
    return "\n".join(lines)
