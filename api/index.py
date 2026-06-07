import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from scraper import fetch_kospi_stocks, fetch_stock_detail, format_for_prompt
from analyzer import analyze_stocks
from report import build_report
from history import save_report, load_previous_report, compare_with_previous

app = Flask(__name__, static_folder="../public")


@app.route("/")
def index():
    return send_from_directory("../public", "index.html")


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
