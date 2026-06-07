import requests
from bs4 import BeautifulSoup


# KOSPI 시가총액 순위 (PER, ROE 포함)
NAVER_MARKET_SUM_URL = "https://finance.naver.com/sise/sise_market_sum.naver?sosok=0"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_kospi_stocks(top_n: int = 50) -> list[dict]:
    """KOSPI 시가총액 상위 종목을 PER·ROE 포함해 스크래핑한다."""
    try:
        response = requests.get(NAVER_MARKET_SUM_URL, headers=HEADERS, timeout=10)
        response.encoding = "euc-kr"
    except requests.RequestException as e:
        raise RuntimeError(f"네이버금융 접속 실패: {e}")

    soup = BeautifulSoup(response.text, "lxml")

    table = soup.find("table", class_="type_2")
    if table is None:
        raise RuntimeError("시가총액 순위 테이블을 찾을 수 없습니다. 네이버금융 페이지 구조가 변경되었을 수 있습니다.")

    stocks = []
    rows = table.find_all("tr")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 9:
            continue

        rank_text = cols[0].get_text(strip=True)
        if not rank_text.isdigit():
            continue

        rank = int(rank_text)
        if rank > top_n:
            break

        name_tag = cols[1].find("a")
        if name_tag is None:
            continue

        name = name_tag.get_text(strip=True)
        href = name_tag.get("href", "")
        code = href.split("code=")[-1].strip() if "code=" in href else ""

        price      = cols[2].get_text(strip=True)
        change_rate = cols[4].get_text(strip=True)
        market_cap = cols[6].get_text(strip=True)
        per        = cols[7].get_text(strip=True)
        roe        = cols[8].get_text(strip=True)

        stocks.append({
            "rank": rank,
            "name": name,
            "code": code,
            "price": price,
            "change_rate": change_rate,
            "market_cap": market_cap,
            "per": per,
            "roe": roe,
        })

    if not stocks:
        raise RuntimeError("스크래핑된 종목이 없습니다. 페이지 구조를 확인하세요.")

    return stocks


def fetch_stock_detail(code: str) -> dict:
    """개별 종목 페이지에서 52주 고저가·현재가를 스크래핑한다."""
    detail = {
        "current_price_raw": 0,
        "week52_high": "N/A",
        "week52_low": "N/A",
        "week52_pct_from_high": "N/A",
    }
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = "euc-kr"
        soup = BeautifulSoup(r.text, "lxml")

        # 현재가
        price_tag = soup.select_one("p.no_today span.blind")
        if price_tag:
            raw = price_tag.get_text(strip=True).replace(",", "")
            if raw.isdigit():
                detail["current_price_raw"] = int(raw)

        # 52주 고저가
        for tr in soup.select("table tr"):
            tds = tr.find_all("td")
            for i, td in enumerate(tds):
                txt = td.get_text(strip=True)
                if "52주최고" in txt and i + 1 < len(tds):
                    detail["week52_high"] = tds[i + 1].get_text(strip=True)
                elif "52주최저" in txt and i + 1 < len(tds):
                    detail["week52_low"] = tds[i + 1].get_text(strip=True)

        # 현재가 대비 52주 고점 대비 비율
        try:
            cur = detail["current_price_raw"]
            high = int(detail["week52_high"].replace(",", ""))
            pct = round((cur - high) / high * 100, 1)
            detail["week52_pct_from_high"] = f"{pct:+.1f}%"
        except Exception:
            pass

    except Exception:
        pass
    return detail


def format_for_prompt(stocks: list[dict]) -> str:
    """AI 프롬프트에 삽입할 표 형태 문자열을 반환한다."""
    lines = ["순위 | 종목명 | 종목코드 | 현재가 | 등락률 | 시가총액(억) | PER | ROE(%)"]
    lines.append("-" * 80)
    for s in stocks:
        lines.append(
            f"{s['rank']} | {s['name']} | {s['code']} | {s['price']} | "
            f"{s['change_rate']} | {s['market_cap']} | {s['per']} | {s['roe']}"
        )
    return "\n".join(lines)
