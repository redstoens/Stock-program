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
from dart_fetcher import fetch_dart_metrics

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))


def _safe_float(v) -> float | None:
    if v is None or str(v).strip() in ("N/A", "", "-"):
        return None
    try:
        return float(str(v).replace(",", "").replace("%", "").replace("+", ""))
    except Exception:
        return None


def _pre_screen(stocks: list[dict], top_n: int = 25) -> list[dict]:
    """펀더멘털·기술적 지표를 점수화해 Gemini에 넘길 상위 종목만 추린다."""
    def _score(s: dict) -> int:
        score = 0

        # PER — 저평가 우대, 고평가 패널티
        per = _safe_float(s.get("per"))
        if per is not None:
            if 0 < per <= 10:  score += 20
            elif per <= 15:    score += 14
            elif per <= 20:    score += 8
            elif per <= 30:    score += 3
            elif per > 50:     score -= 8

        # ROE — 수익성 우대
        roe = _safe_float(s.get("roe"))
        if roe is not None:
            if roe >= 20:   score += 20
            elif roe >= 15: score += 14
            elif roe >= 10: score += 8
            elif roe >= 5:  score += 3
            elif roe < 0:   score -= 15

        # RSI — 과매수 패널티, 눌림목 우대
        rsi = _safe_float(s.get("rsi"))
        if rsi is not None:
            if rsi > 75:            score -= 20
            elif rsi > 70:          score -= 8
            elif 40 <= rsi <= 60:   score += 10
            elif 30 <= rsi < 40:    score += 15  # 눌림목
            elif 20 <= rsi < 30:    score += 12  # 과매도 반등
            elif rsi < 20:          score += 5

        # MA 배열 신호
        ma = s.get("ma_signal", "N/A")
        if ma == "정배열":   score += 12
        elif ma == "MA위":   score += 6
        elif ma == "혼조":   score += 2
        elif ma == "역배열": score -= 8

        # 52주 고가 대비 위치 — 적정 눌림 우대
        pct = _safe_float(s.get("week52_pct_from_high"))
        if pct is not None:
            if -25 <= pct <= -5:    score += 12  # 적정 눌림목
            elif -40 <= pct < -25:  score += 8   # 가치 기회
            elif pct > -5:          score += 5   # 고가 근처 모멘텀
            elif pct < -50:         score -= 5   # 과도한 낙폭

        return score

    return sorted(stocks, key=_score, reverse=True)[:top_n]


def _enrich_technicals(stocks: list[dict], market: str = "kr") -> None:
    """RSI·MA·MACD·거래량 신호를 yfinance 3개월 데이터로 계산해 stocks에 in-place 추가."""
    def _compute(stock: dict) -> None:
        code = stock.get("code", "")
        if not code:
            return
        symbols = [f"{code}.KS", f"{code}.KQ"] if market == "kr" else [code]
        hist = None
        for sym in symbols:
            try:
                h = yf.Ticker(sym).history(period="3mo")
                if not h.empty:
                    hist = h
                    break
            except Exception:
                continue
        if hist is None or len(hist) < 5:
            return

        closes  = hist["Close"]
        volumes = hist["Volume"]
        price   = float(closes.iloc[-1])
        n       = len(closes)

        # MA20 / MA60
        ma20 = float(closes.tail(20).mean())
        ma60 = float(closes.tail(60).mean()) if n >= 60 else None
        stock["ma20"] = round(ma20) if market == "kr" else round(ma20, 2)
        stock["ma60"] = ((round(ma60) if market == "kr" else round(ma60, 2)) if ma60 else None)

        # MA 배열 신호
        if ma60:
            if price > ma20 > ma60:   stock["ma_signal"] = "정배열"
            elif price < ma20 < ma60: stock["ma_signal"] = "역배열"
            else:                     stock["ma_signal"] = "혼조"
        else:
            stock["ma_signal"] = "단기상승" if price > ma20 else "단기하락"

        # RSI(14)
        delta = closes.diff().dropna()
        gain  = delta.clip(lower=0).tail(14).mean()
        loss  = (-delta.clip(upper=0)).tail(14).mean()
        rsi   = round(float(100 - (100 / (1 + gain / loss))), 1) if loss > 0 else 50.0
        stock["rsi"] = rsi
        stock["rsi_signal"] = "과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립"

        # MACD (12/26/9) — 최근 크로스 또는 현재 추세
        if n >= 26:
            ema12     = closes.ewm(span=12, adjust=False).mean()
            ema26     = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            sig_line  = macd_line.ewm(span=9, adjust=False).mean()
            prev_diff = float(macd_line.iloc[-2]) - float(sig_line.iloc[-2])
            curr_diff = float(macd_line.iloc[-1]) - float(sig_line.iloc[-1])
            if prev_diff < 0 <= curr_diff:
                stock["macd_signal"] = "골든크로스"
            elif prev_diff > 0 > curr_diff:
                stock["macd_signal"] = "데드크로스"
            elif curr_diff > 0:
                stock["macd_signal"] = "상승추세"
            else:
                stock["macd_signal"] = "하락추세"

        # 거래량 비율 (최근 20일 평균 대비)
        if n >= 21:
            avg_vol   = float(volumes.iloc[-21:-1].mean())
            today_vol = float(volumes.iloc[-1])
            if avg_vol > 0:
                ratio = round(today_vol / avg_vol, 1)
                stock["volume_ratio"] = ratio
                if ratio >= 3.0:   stock["volume_signal"] = "급증"
                elif ratio >= 1.5: stock["volume_signal"] = "증가"
                elif ratio <= 0.3: stock["volume_signal"] = "급감"
                elif ratio <= 0.7: stock["volume_signal"] = "감소"
                else:              stock["volume_signal"] = "보통"

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_compute, stocks))


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

        # 1-1. 정량 필터링 — 80개 → 상위 25개로 압축 후 Gemini에 전달
        screened = _pre_screen(stocks_raw, top_n=25)

        # 1-2. DART 재무지표 추가 (영업이익 트렌드·부채비율) — 최대 12초
        from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE
        try:
            with _TPE(max_workers=1) as _ex:
                dart_data = _ex.submit(fetch_dart_metrics, [s["code"] for s in screened]).result(timeout=12)
        except (_TE, Exception):
            dart_data = {}
        for s in screened:
            d = dart_data.get(s["code"], {})
            if d:
                s["trend_label"]   = d.get("trend_label")
                s["debt_ratio"]    = d.get("debt_ratio")
                s["op_margin_dart"]= d.get("op_margin")
                s["op_growth_pct"] = d.get("op_growth_pct")

        stock_table = format_for_prompt(screened)

        # 2. 이전 리포트 로드
        prev_report = load_previous_report()

        # 3. Gemini AI 분석
        analyzed = analyze_stocks(stock_table, memo)

        # 4. 개별 종목 상세 (52주 고저가) 스크래핑 및 병합
        for stock in analyzed:
            detail = fetch_stock_detail(stock.get("code", ""))
            stock.update(detail)

        # 4-1. 기술적 지표 (RSI·MA·MACD·거래량) 계산
        _enrich_technicals(analyzed, market="kr")

        # 4-2. DART 재무지표를 선정 종목에 병합 (카드 표시용)
        screened_map = {s["code"]: s for s in screened}
        for stock in analyzed:
            src = screened_map.get(stock.get("code", ""), {})
            for key in ("trend_label", "debt_ratio", "op_margin_dart", "op_growth_pct"):
                if src.get(key) is not None:
                    stock[key] = src[key]

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

        # 정량 필터링 — 50개 → 상위 25개로 압축
        screened_us = _pre_screen(stocks_raw, top_n=25)
        stock_table = format_for_prompt_us(screened_us)
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

        # 기술적 지표 (RSI·MA·MACD·거래량) 계산
        _enrich_technicals(analyzed, market="us")

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
