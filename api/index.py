import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import json
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template
import yfinance as yf
from scraper import fetch_kospi_stocks, fetch_stock_detail, format_for_prompt
from scraper_us import fetch_sp500_stocks, format_for_prompt_us
from analyzer import analyze_stocks
from analyzer_us import analyze_stocks_us
from report import build_report
from history import save_report, load_previous_report, compare_with_previous
from news_fetcher import fetch_stock_news

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cached-kr")
def cached_kr():
    path = os.path.join(_HERE, "data", "latest_kr.json")
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"success": False, "error": "no_data"}), 404


@app.route("/api/verify-kr")
def verify_kr():
    path = os.path.join(_HERE, "data", "track_kr.json")
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"records": [], "benchmark": {}}), 200


@app.route("/api/verify-us")
def verify_us():
    path = os.path.join(_HERE, "data", "track_us.json")
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"records": [], "benchmark": {}}), 200


@app.route("/api/cached-us")
def cached_us():
    path = os.path.join(_HERE, "data", "latest_us.json")
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"success": False, "error": "no_data"}), 404


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        memo = data.get("memo", "")

        # 1. KOSPI 시가총액 상위 스크래핑
        stocks_raw = fetch_kospi_stocks(top_n=80)
        stock_table = format_for_prompt(stocks_raw)

        # 2. 이전 리포트 로드
        prev_report = load_previous_report()

        # 3. Gemini AI 분석
        analyzed = analyze_stocks(stock_table, memo)

        # 4. 개별 종목 상세 (52주 고저가) 스크래핑 및 병합
        for stock in analyzed:
            detail = fetch_stock_detail(stock.get("code", ""))
            stock.update(detail)

        # 5. 과거 리포트와 비교
        overlaps = compare_with_previous(analyzed, prev_report)

        # 6. 현재 리포트 저장
        save_report(analyzed)

        # 7. 마크다운 리포트 생성
        report_md = build_report(analyzed, memo)

        return jsonify({
            "success": True,
            "report": report_md,
            "stocks": analyzed,
            "prev_date": prev_report.get("date") if prev_report else None,
            "overlaps": overlaps,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/current-prices", methods=["POST"])
def current_prices():
    data   = request.get_json(force=True)
    codes  = data.get("codes", [])
    market = data.get("market", "kr")

    def _fetch(code):
        try:
            if market == "us":
                price = yf.Ticker(code).fast_info.last_price or 0
            else:
                price = 0
                for suffix in (".KS", ".KQ"):
                    try:
                        p = yf.Ticker(f"{code}{suffix}").fast_info.last_price or 0
                        if p:
                            price = p
                            break
                    except Exception:
                        continue
            return code, round(float(price), 2) if price else 0
        except Exception:
            return code, 0

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_fetch, c) for c in codes[:30]]
        results = dict(f.result() for f in futures)

    return jsonify({"prices": results})


@app.route("/api/news")
def news():
    try:
        items = fetch_stock_news()
        return jsonify({"success": True, "news": items})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/market-index")
def market_index():
    def _fetch(symbol):
        try:
            info = yf.Ticker(symbol).fast_info
            price = info.last_price or info.regular_market_price or 0
            prev  = info.previous_close or price
            change     = round(price - prev, 2)
            change_pct = round((change / prev * 100), 2) if prev else 0
            return {"price": round(price, 2), "change": change, "change_pct": change_pct}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_kr  = ex.submit(_fetch, "^KS11")
        f_kq  = ex.submit(_fetch, "^KQ11")
        f_sp  = ex.submit(_fetch, "^GSPC")
        f_nq  = ex.submit(_fetch, "^IXIC")
        kr, kq, sp, nq = f_kr.result(), f_kq.result(), f_sp.result(), f_nq.result()

    return jsonify({"kr": kr, "kq": kq, "sp": sp, "nq": nq})


@app.route("/api/analyze-us", methods=["POST"])
def analyze_us():
    try:
        data = request.get_json(force=True)
        memo = data.get("memo", "")

        stocks_raw = fetch_sp500_stocks(top_n=50)
        stock_table = format_for_prompt_us(stocks_raw)
        analyzed = analyze_stocks_us(stock_table, memo)

        stock_map = {s["code"]: s for s in stocks_raw}
        for stock in analyzed:
            code = stock.get("code", "")
            if code in stock_map:
                raw = stock_map[code]
                stock.setdefault("current_price_raw", raw.get("current_price_raw", 0))
                stock.setdefault("week52_high", raw.get("week52_high", ""))
                stock.setdefault("week52_low", raw.get("week52_low", ""))
                stock.setdefault("week52_pct_from_high", raw.get("week52_pct_from_high", ""))

        report_md = build_report(analyzed, memo)

        return jsonify({
            "success": True,
            "report": report_md,
            "stocks": analyzed,
            "prev_date": None,
            "overlaps": [],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
