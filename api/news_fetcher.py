import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import requests

RSS_URLS = [
    "https://news.google.com/rss/search?q=코스피+코스닥+주식+증시&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=미국주식+나스닥+S%26P500&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=환율+금리+원자재+주식&hl=ko&gl=KR&ceid=KR:ko",
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _parse_rss(url: str) -> list[dict]:
    res = requests.get(url, headers=_HEADERS, timeout=7)
    res.encoding = "utf-8"
    root = ET.fromstring(res.content)
    items = []
    for el in root.findall(".//item"):
        title = (el.findtext("title") or "").strip()
        if not title:
            continue
        link_el = el.find("link")
        link = ""
        if link_el is not None:
            link = (link_el.text or "").strip()
            if not link and link_el.tail:
                link = link_el.tail.strip()
        source_el = el.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        pub_str = (el.findtext("pubDate") or "").strip()
        # pubDate를 timestamp로 변환 (정렬용)
        try:
            pub_ts = parsedate_to_datetime(pub_str).timestamp()
        except Exception:
            pub_ts = 0
        items.append({"title": title, "url": link, "source": source,
                       "published": pub_str, "_ts": pub_ts})
    return items


def fetch_stock_news(limit: int = 20) -> list[dict]:
    raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_parse_rss, url) for url in RSS_URLS]
        for f in futures:
            try:
                raw.extend(f.result())
            except Exception:
                pass

    # 중복 제목 제거 (앞 25자 기준)
    seen, deduped = set(), []
    for item in sorted(raw, key=lambda x: x["_ts"], reverse=True):
        key = item["title"][:25]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # _ts 필드 제거 후 반환
    result = []
    for item in deduped[:limit]:
        result.append({k: v for k, v in item.items() if k != "_ts"})
    return result
