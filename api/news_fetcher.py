import json
import os
import xml.etree.ElementTree as ET

import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
RSS_URLS = [
    "https://news.google.com/rss/search?q=코스피+코스닥+주식+증시&hl=ko&gl=KR&ceid=KR:ko",
    "https://news.google.com/rss/search?q=미국주식+나스닥+S%26P500&hl=ko&gl=KR&ceid=KR:ko",
]


def _parse_rss(url: str, limit: int = 20) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    res = requests.get(url, headers=headers, timeout=7)
    res.encoding = "utf-8"
    root = ET.fromstring(res.content)
    items = []
    for el in root.findall(".//item")[:limit]:
        title = (el.findtext("title") or "").strip()
        if not title:
            continue
        # Google RSS: <link> is sometimes a tail node
        link_el = el.find("link")
        link = ""
        if link_el is not None:
            link = (link_el.text or "").strip()
            if not link and link_el.tail:
                link = link_el.tail.strip()
        source_el = el.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        pub = (el.findtext("pubDate") or "").strip()
        items.append({"title": title, "url": link, "source": source, "published": pub})
    return items


def fetch_stock_news() -> list[dict]:
    raw: list[dict] = []
    for url in RSS_URLS:
        try:
            raw.extend(_parse_rss(url, limit=20))
        except Exception:
            pass

    # 중복 제목 제거
    seen, deduped = set(), []
    for item in raw:
        key = item["title"][:30]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    if not deduped:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return deduped[:10]

    lines = [f"{i+1}. {it['title']} ({it['source']})" for i, it in enumerate(deduped[:35])]
    prompt = f"""아래는 최근 한국·미국 주식 시장 관련 뉴스 제목 목록입니다.

{chr(10).join(lines)}

위 뉴스 중 오늘 주식 시장(코스피·코스닥·나스닥·S&P500·환율·금리·원자재 등)에 가장 영향력이 큰 뉴스 10개를 선별하고, 각 뉴스의 시장 영향을 투자자 관점에서 한 줄(15자 이내)로 코멘트하세요.

다음 JSON 형식으로만 반환하세요 (다른 텍스트 없이):
{{"selected": [{{"idx": 원본번호, "comment": "한 줄 코멘트"}}, ...]}}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        data = json.loads(response.text)
        selected = data.get("selected", [])
        result = []
        for sel in selected[:10]:
            raw_idx = int(sel.get("idx", 0)) - 1
            if 0 <= raw_idx < len(deduped):
                item = dict(deduped[raw_idx])
                item["comment"] = sel.get("comment", "")
                result.append(item)
        return result if result else deduped[:10]
    except Exception:
        return deduped[:10]
