import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"

RSS_URLS = [
    "https://news.google.com/rss/search?q=코스피+코스닥+주식+증시&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=미국주식+나스닥+S%26P500&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=환율+금리+원자재+주식&hl=ko&gl=KR&ceid=KR:ko",
]

SECTORS = "반도체, 금융, 바이오/제약, 자동차, 조선/방산, IT서비스, 소비재, 에너지/화학, 건설, 통신, 플랫폼/게임, 거시경제, 기타"

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
        try:
            pub_ts = parsedate_to_datetime(pub_str).timestamp()
        except Exception:
            pub_ts = 0
        items.append({"title": title, "url": link, "source": source,
                      "published": pub_str, "_ts": pub_ts})
    return items


def _normalize(title: str) -> str:
    """중복 비교용 정규화: 태그·출처 제거 후 소문자화."""
    t = re.sub(r'\[.*?\]|\(.*?\)', '', title)   # [속보], (종합) 등 제거
    t = re.sub(r'[=|·…""\'\-]+.*$', '', t)      # 구분자 이후 출처 제거
    t = re.sub(r'\s+', ' ', t).strip().lower()
    return t


def _is_similar(norm_a: str, norm_b: str, threshold: float = 0.55) -> bool:
    """단어 집합 겹침 비율로 유사 기사 판별."""
    wa = set(norm_a.split())
    wb = set(norm_b.split())
    if len(wa) < 3 or len(wb) < 3:
        return norm_a[:20] == norm_b[:20]
    overlap = len(wa & wb) / min(len(wa), len(wb))
    return overlap >= threshold


def _fetch_raw(limit: int = 30) -> list[dict]:
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_parse_rss, url) for url in RSS_URLS]
        for f in futures:
            try:
                raw.extend(f.result())
            except Exception:
                pass

    deduped: list[dict] = []
    norms: list[str] = []

    for item in sorted(raw, key=lambda x: x["_ts"], reverse=True):
        norm = _normalize(item["title"])
        # 이미 추가된 기사와 유사하면 건너뜀
        if any(_is_similar(norm, n) for n in norms):
            continue
        norms.append(norm)
        deduped.append({k: v for k, v in item.items() if k != "_ts"})

    return deduped[:limit]


def _analyze_sentiment(news_list: list[dict]) -> tuple[list[dict], list[dict]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        for item in news_list:
            item.update({"sector": "기타", "sentiment": "중립", "reason": ""})
        return news_list, []

    lines = [f"{i+1}. {it['title']} ({it['source']})" for i, it in enumerate(news_list)]
    prompt = f"""다음은 오늘의 주식·금융 관련 뉴스 목록입니다.

{chr(10).join(lines)}

다음 두 가지를 분석하세요.

[1] 각 뉴스 기사별 분석
- sector: 섹터 분류 (반드시 아래 중 하나): {SECTORS}
- sentiment: 긍정 / 부정 / 중립 중 하나
- reason: 투자자 관점 15자 이내

[2] 오늘 실질적으로 영향받는 섹터 분석 (뉴스에서 실제 이슈가 있는 섹터만 포함, 최대 8개)
- sector: 섹터명 (위 분류 기준)
- direction: 상승압력 / 하락압력 / 중립 중 하나
- summary: 해당 섹터에 영향을 주는 핵심 이슈 한 줄 (30자 이내)
- stocks: 직접 영향받는 대표 종목명 최대 4개 (한국 또는 미국 상장사, 종목명만)

다음 JSON 형식으로만 반환하세요:
{{
  "items": [{{"idx": 1, "sector": "반도체", "sentiment": "긍정", "reason": "AI 수요 증가 수혜"}}, ...],
  "sector_impacts": [
    {{"sector": "반도체", "direction": "상승압력", "summary": "미국 AI 투자 확대로 HBM 수요 급증", "stocks": ["SK하이닉스", "삼성전자"]}},
    ...
  ]
}}"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.15,
        ),
    )
    data = json.loads(response.text)
    result_map = {it["idx"]: it for it in data.get("items", [])}

    for i, item in enumerate(news_list, 1):
        info = result_map.get(i, {})
        item["sector"]    = info.get("sector", "기타")
        item["sentiment"] = info.get("sentiment", "중립")
        item["reason"]    = info.get("reason", "")

    sector_impacts = data.get("sector_impacts", [])
    return news_list, sector_impacts


def _build_sector_summary(news_list: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for item in news_list:
        sec  = item.get("sector", "기타")
        sent = item.get("sentiment", "중립")
        if sec not in summary:
            summary[sec] = {"pos": 0, "neg": 0, "neu": 0, "items": []}
        if sent == "긍정":
            summary[sec]["pos"] += 1
        elif sent == "부정":
            summary[sec]["neg"] += 1
        else:
            summary[sec]["neu"] += 1
        summary[sec]["items"].append(item["title"])
    return summary


def fetch_news_with_sentiment(limit: int = 20) -> dict:
    news = _fetch_raw(limit)
    if not news:
        return {"news": [], "sector_summary": {}, "sector_impacts": []}
    news, sector_impacts = _analyze_sentiment(news)
    sector_summary = _build_sector_summary(news)
    return {"news": news, "sector_summary": sector_summary, "sector_impacts": sector_impacts}
