from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# KOSPI + KOSDAQ 주요 종목 (.KS = KOSPI, .KQ = KOSDAQ)
KOSPI_TICKERS = [
    # ── KOSPI 시가총액 상위 ──────────────────────────────────
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "207940.KS",  # 삼성바이오로직스
    "005380.KS",  # 현대차
    "000270.KS",  # 기아
    "068270.KS",  # 셀트리온
    "051910.KS",  # LG화학
    "055550.KS",  # 신한지주
    "035420.KS",  # NAVER
    "012330.KS",  # 현대모비스
    "028260.KS",  # 삼성물산
    "105560.KS",  # KB금융
    "066570.KS",  # LG전자
    "032830.KS",  # 삼성생명
    "096770.KS",  # SK이노베이션
    "003550.KS",  # LG
    "017670.KS",  # SK텔레콤
    "015760.KS",  # 한국전력
    "034730.KS",  # SK
    "000810.KS",  # 삼성화재
    "086790.KS",  # 하나금융지주
    "009150.KS",  # 삼성전기
    "030200.KS",  # KT
    "018260.KS",  # 삼성SDS
    "010130.KS",  # 고려아연
    "006400.KS",  # 삼성SDI
    "011200.KS",  # HMM
    "035720.KS",  # 카카오
    "003490.KS",  # 대한항공
    "090430.KS",  # 아모레퍼시픽
    "010950.KS",  # S-Oil
    "024110.KS",  # 기업은행
    "316140.KS",  # 우리금융지주
    "259960.KS",  # 크래프톤
    "047810.KS",  # 한국항공우주
    "009540.KS",  # HD한국조선해양
    "034020.KS",  # 두산에너빌리티
    "000720.KS",  # 현대건설
    "012450.KS",  # 한화에어로스페이스
    "329180.KS",  # HD현대중공업
    "267250.KS",  # HD현대
    "078930.KS",  # GS
    "000100.KS",  # 유한양행
    "128940.KS",  # 한미약품
    "352820.KS",  # 하이브
    "021240.KS",  # 코웨이
    "139480.KS",  # 이마트
    "097950.KS",  # CJ제일제당
    "004990.KS",  # 롯데지주
    "272210.KS",  # 한화시스템
    "336260.KS",  # 두산밥캣
    "028050.KS",  # 삼성엔지니어링
    "003230.KS",  # 삼양식품
    "010140.KS",  # 삼성중공업
    "011790.KS",  # SKC
    "377300.KS",  # 카카오페이
    "251270.KS",  # 넷마블
    "004370.KS",  # 농심
    "161390.KS",  # 한국타이어앤테크놀로지
    "042660.KS",  # 한화오션
    # ── KOSPI 추가 종목 ──────────────────────────────────────
    "005490.KS",  # POSCO홀딩스
    "004020.KS",  # 현대제철
    "036570.KS",  # 엔씨소프트
    "009830.KS",  # 한화솔루션
    "011780.KS",  # 금호석유화학
    "023530.KS",  # 롯데쇼핑
    "001040.KS",  # CJ
    "000150.KS",  # 두산
    "007310.KS",  # 오뚜기
    "047050.KS",  # 포스코인터내셔널
    "005940.KS",  # NH투자증권
    "006800.KS",  # 미래에셋증권
    "000120.KS",  # CJ대한통운
    "011070.KS",  # LG이노텍
    "016360.KS",  # 삼성증권
    "071050.KS",  # 한국금융지주
    "000880.KS",  # 한화
    "069960.KS",  # 현대백화점
    "006360.KS",  # GS건설
    "051600.KS",  # 한전KPS
    "030000.KS",  # 제일기획
    "011170.KS",  # 롯데케미칼
    "010060.KS",  # OCI홀딩스
    "003600.KS",  # SK케미칼
    "020560.KS",  # 아시아나항공
    # ── KOSDAQ 주요 종목 ─────────────────────────────────────
    "247540.KQ",  # 에코프로비엠
    "086520.KQ",  # 에코프로
    "196170.KQ",  # 알테오젠
    "357780.KQ",  # 솔브레인
    "066970.KQ",  # L&F
    "022100.KQ",  # 포스코DX
    "039030.KQ",  # 이오테크닉스
    "145020.KQ",  # 휴젤
    "214150.KQ",  # 클래시스
    "293490.KQ",  # 카카오게임즈
    "263750.KQ",  # 펄어비스
    "328130.KQ",  # 루닛
    "095340.KQ",  # ISC
    "064760.KQ",  # 티씨케이
    "131370.KQ",  # 엠아이텍
]


def _fetch_one(ticker_sym: str) -> dict | None:
    try:
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info
        code = ticker_sym.replace(".KS", "").replace(".KQ", "")
        name = info.get("longName") or info.get("shortName") or code

        cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        market_cap = info.get("marketCap") or 0
        if not cur_price or not market_cap:
            return None

        change_pct = info.get("regularMarketChangePercent") or 0
        per_val = info.get("trailingPE") or 0
        roe_raw = info.get("returnOnEquity") or 0
        w52_high = info.get("fiftyTwoWeekHigh") or 0
        w52_low = info.get("fiftyTwoWeekLow") or 0

        mktcap_억 = market_cap // 100_000_000
        pct_from_high = round((cur_price - w52_high) / w52_high * 100, 1) if w52_high > 0 else 0

        # RSI(14) — 1개월 히스토리
        rsi_str = "N/A"
        try:
            hist = ticker.history(period="1mo")
            if len(hist) >= 15:
                closes = hist["Close"]
                delta = closes.diff().dropna()
                gain  = delta.clip(lower=0).tail(14).mean()
                loss  = (-delta.clip(upper=0)).tail(14).mean()
                if loss > 0:
                    rsi_str = str(round(float(100 - (100 / (1 + gain / loss))), 1))
        except Exception:
            pass

        # MA 배열 신호 — info의 fiftyDayAverage / twoHundredDayAverage 활용 (추가 호출 없음)
        ma50  = info.get("fiftyDayAverage") or 0
        ma200 = info.get("twoHundredDayAverage") or 0
        if ma50 and ma200:
            if cur_price > ma50 > ma200:   ma_sig = "정배열"
            elif cur_price < ma50 < ma200: ma_sig = "역배열"
            else:                          ma_sig = "혼조"
        elif ma50:
            ma_sig = "MA위" if cur_price > ma50 else "MA아래"
        else:
            ma_sig = "N/A"

        return {
            "code": code,
            "name": name,
            "price": f"{int(cur_price):,}",
            "change_rate": f"{change_pct:+.2f}%" if change_pct else "",
            "market_cap": f"{mktcap_억:,}",
            "market_cap_raw": market_cap,
            "per": str(round(per_val, 1)) if per_val > 0 else "N/A",
            "roe": str(round(roe_raw * 100, 1)) if roe_raw else "N/A",
            "current_price_raw": int(cur_price),
            "week52_high": f"{int(w52_high):,}" if w52_high else "N/A",
            "week52_low": f"{int(w52_low):,}" if w52_low else "N/A",
            "week52_pct_from_high": f"{pct_from_high:+.1f}%" if w52_high else "N/A",
            "rsi": rsi_str,
            "ma_signal": ma_sig,
        }
    except Exception:
        return None


def fetch_kospi_stocks(top_n: int = 80) -> list[dict]:
    """KOSPI + KOSDAQ 주요 종목 데이터 — yfinance."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_fetch_one, KOSPI_TICKERS))

    stocks = [r for r in results if r is not None]
    if not stocks:
        raise RuntimeError("종목 데이터를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")

    stocks.sort(key=lambda x: x["market_cap_raw"], reverse=True)
    for i, s in enumerate(stocks[:top_n], start=1):
        s["rank"] = i
    return stocks[:top_n]


def fetch_stock_detail(code: str) -> dict:
    """AI 선택 종목 상세 — 52주 고저가 + 현재가."""
    detail = {
        "current_price_raw": 0,
        "week52_high": "N/A",
        "week52_low": "N/A",
        "week52_pct_from_high": "N/A",
    }
    # .KS 먼저 시도, 실패 시 .KQ
    for suffix in (".KS", ".KQ"):
        try:
            info = yf.Ticker(f"{code}{suffix}").info
            cur_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            w52_high = info.get("fiftyTwoWeekHigh") or 0
            w52_low = info.get("fiftyTwoWeekLow") or 0
            if cur_price and w52_high:
                pct = round((cur_price - w52_high) / w52_high * 100, 1)
                pbr_raw = info.get("priceToBook")
                detail.update({
                    "current_price_raw": int(cur_price),
                    "week52_high": f"{int(w52_high):,}",
                    "week52_low": f"{int(w52_low):,}",
                    "week52_pct_from_high": f"{pct:+.1f}%",
                    "pbr": str(round(pbr_raw, 2)) if pbr_raw and pbr_raw > 0 else "N/A",
                })
                break
        except Exception:
            continue
    return detail


def format_for_prompt(stocks: list[dict]) -> str:
    """AI 프롬프트 테이블 문자열."""
    has_dart = any(s.get("trend_label") for s in stocks)
    header = (
        "순위 | 종목명 | 종목코드 | 현재가 | 등락률 | 시가총액(억) | PER | ROE(%) "
        "| RSI | MA배열 | 52주고가대비"
        + (" | 영업이익트렌드(YoY) | 부채비율(%)" if has_dart else "")
    )
    lines = [header, "-" * (120 if has_dart else 100)]
    for s in stocks:
        row = (
            f"{s['rank']} | {s['name']} | {s['code']} | {s['price']} | "
            f"{s['change_rate']} | {s['market_cap']} | {s['per']} | {s['roe']} | "
            f"{s.get('rsi', 'N/A')} | {s.get('ma_signal', 'N/A')} | {s.get('week52_pct_from_high', 'N/A')}"
        )
        if has_dart:
            trend = s.get("trend_label", "N/A")
            debt  = f"{s['debt_ratio']}%" if s.get("debt_ratio") is not None else "N/A"
            row  += f" | {trend} | {debt}"
        lines.append(row)
    return "\n".join(lines)
