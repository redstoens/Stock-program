import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from flask import Flask, request, jsonify, render_template
from scraper import fetch_kospi_stocks, fetch_stock_detail, format_for_prompt
from scraper_us import fetch_sp500_stocks, format_for_prompt_us
from analyzer import analyze_stocks
from analyzer_us import analyze_stocks_us
from report import build_report
from history import save_report, load_previous_report, compare_with_previous

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))


@app.route("/")
def index():
    return render_template("index.html")


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
