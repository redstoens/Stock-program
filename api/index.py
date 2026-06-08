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
from news_fetcher import fetch_news_with_sentiment
from analyzer_single import analyze_single_stock

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


@app.route("/api/fear-greed")
def fear_greed():
    def _fetch_vix():
        return yf.Ticker("^VIX").fast_info.last_price or 20.0

    def _fetch_sp500():
        hist = yf.Ticker("^GSPC").history(period="6mo")
        if hist.empty:
            return None
        closes = hist["Close"]
        current = float(closes.iloc[-1])
        ma125   = float(closes.tail(125).mean()) if len(closes) >= 125 else float(closes.mean())
        delta   = closes.diff().dropna()
        gain    = delta.clip(lower=0).tail(14).mean()
        loss    = (-delta.clip(upper=0)).tail(14).mean()
        rsi     = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50.0
        return {"current": current, "ma125": ma125, "rsi": float(rsi)}

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_vix = ex.submit(_fetch_vix)
            f_sp  = ex.submit(_fetch_sp500)
            vix     = f_vix.result()
            sp_data = f_sp.result()

        vix_score = max(0.0, min(100.0, (40 - vix) / 30 * 100))

        if sp_data:
            pct            = (sp_data["current"] - sp_data["ma125"]) / sp_data["ma125"] * 100
            momentum_score = max(0.0, min(100.0, 50 + pct * 5))
            rsi_score      = max(0.0, min(100.0, float(sp_data["rsi"])))
            rsi_val        = round(sp_data["rsi"], 1)
        else:
            momentum_score = rsi_score = 50.0
            rsi_val = 50.0

        score = round(vix_score * 0.5 + momentum_score * 0.3 + rsi_score * 0.2)

        if   score <= 24: label = "극단적 공포"
        elif score <= 44: label = "공포"
        elif score <= 55: label = "중립"
        elif score <= 74: label = "탐욕"
        else:             label = "극단적 탐욕"

        return jsonify({
            "score": score, "label": label,
            "vix": round(vix, 1), "rsi": rsi_val,
            "components": {
                "vix":      round(vix_score),
                "momentum": round(momentum_score),
                "rsi":      round(rsi_score),
            }
        })
    except Exception as e:
        return jsonify({"score": 50, "label": "중립", "error": str(e)})


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


@app.route("/api/track-manual-add", methods=["POST"])
def track_manual_add():
    import re as _re, datetime as _dt
    data   = request.get_json(force=True)
    market = (data.get("market") or "kr").lower()
    today  = _dt.date.today().isoformat()

    symbol = data.get("symbol", "")
    code   = _re.sub(r'\.(KS|KQ)$', '', symbol, flags=_re.IGNORECASE) if market == "kr" else symbol

    def _num(v):
        if v is None: return 0.0
        try: return float(v)
        except Exception: return 0.0

    entry_price      = _num(data.get("entry_price_num"))
    target_price_raw = _num(data.get("target_price_num"))
    stop_loss_raw    = _num(data.get("stop_loss_num"))
    current_price    = _num(data.get("current_price_num")) or entry_price
    investment_horizon = data.get("investment_horizon") or "단기"
    is_long   = "중장기" in investment_horizon
    hold_period = "3~12개월" if is_long else "1~3개월"
    max_days    = 365 if is_long else 90
    ret_pct = round((current_price - entry_price) / entry_price * 100, 2) if entry_price else 0.0

    record = {
        "rec_date":          today,
        "code":              code,
        "name":              data.get("name") or code,
        "entry_price":       entry_price,
        "target_price_raw":  target_price_raw,
        "stop_loss_raw":     stop_loss_raw,
        "target_str":        data.get("target_price_str") or "",
        "stop_loss_str":     data.get("stop_loss_str")    or "",
        "investment_horizon": investment_horizon,
        "hold_period":       hold_period,
        "max_days":          max_days,
        "status":            "진행중",
        "current_price":     current_price,
        "current_return_pct": ret_pct,
        "last_updated":      today,
        "exit_date":         None,
        "exit_price":        None,
        "exit_return_pct":   None,
        "source":            "manual",
    }

    fname = "track_us.json" if market == "us" else "track_kr.json"
    path  = os.path.join(_HERE, "data", fname)
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {"records": []}

    existing["records"].append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "record": record})


@app.route("/api/stock-search", methods=["POST"])
def stock_search():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"success": False, "error": "검색어를 입력하세요"}), 400
    try:
        result = analyze_single_stock(query)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stock-chart", methods=["POST"])
def stock_chart():
    data   = request.get_json(force=True)
    symbol = (data.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"success": False, "error": "symbol이 없습니다"}), 400
    try:
        import pandas as pd
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="6mo")
        if hist.empty:
            return jsonify({"success": False, "error": "차트 데이터 없음"}), 404

        closes  = hist["Close"]
        dates   = [str(d.date()) for d in hist.index]

        def _ma(n):
            ma = closes.rolling(n).mean()
            return [round(float(v), 2) if not pd.isna(v) else None for v in ma]

        delta = closes.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_s = 100 - (100 / (1 + gain / loss))

        return jsonify({
            "success": True,
            "dates":   dates,
            "closes":  [round(float(v), 2) for v in closes],
            "ma20":    _ma(20),
            "ma60":    _ma(60),
            "rsi":     [round(float(v), 1) if not pd.isna(v) else None for v in rsi_s],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/news")
def news():
    try:
        result = fetch_news_with_sentiment()
        return jsonify({"success": True, **result})
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
