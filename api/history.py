import json
import os
from datetime import date, datetime

_BASE = "/tmp" if os.path.isdir("/tmp") else os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(_BASE, "reports")


def _ensure_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def save_report(stocks: list[dict]) -> None:
    """오늘 날짜로 리포트를 JSON 파일에 저장한다."""
    _ensure_dir()
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    path = os.path.join(REPORTS_DIR, filename)
    payload = {
        "date": date.today().isoformat(),
        "stocks": [
            {
                "code": s.get("code", ""),
                "name": s.get("name", ""),
                "target_price": s.get("target_price", ""),
                "stop_loss": s.get("stop_loss", ""),
                "current_price_raw": s.get("current_price_raw", 0),
            }
            for s in stocks
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_previous_report() -> dict | None:
    """오늘 이전 날짜 리포트 중 가장 최근 것을 반환한다. 없으면 None."""
    _ensure_dir()
    today = date.today().isoformat()
    files = sorted(
        [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")],
        reverse=True,
    )
    for filename in files:
        path = os.path.join(REPORTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if report.get("date", "") < today:
            return report
    return None


def compare_with_previous(current_stocks: list[dict], prev_report: dict | None) -> list[dict]:
    """이전 리포트와 현재 선정 종목을 비교해 겹치는 종목의 등락률을 계산한다."""
    if not prev_report:
        return []

    prev_map = {s["code"]: s for s in prev_report.get("stocks", [])}
    prev_date = prev_report.get("date", "")
    results = []

    for stock in current_stocks:
        code = stock.get("code", "")
        if code not in prev_map:
            continue

        prev = prev_map[code]
        prev_price = prev.get("current_price_raw", 0)
        curr_price = stock.get("current_price_raw", 0)

        change_pct = "N/A"
        if prev_price and curr_price:
            pct = round((curr_price - prev_price) / prev_price * 100, 2)
            sign = "+" if pct >= 0 else ""
            change_pct = f"{sign}{pct}%"

        results.append({
            "code": code,
            "name": stock.get("name", ""),
            "prev_date": prev_date,
            "change_pct": change_pct,
        })

    return results
