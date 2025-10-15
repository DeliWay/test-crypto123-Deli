# app.py — Flask-версия под твои templates/static и асинхронный backend
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from functools import wraps
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from flask import (
    Flask, abort, jsonify, make_response, redirect, render_template,
    render_template_string, request, url_for
)

# ================== проектные модули ==================
try:
    from backend.exchange_data import get_candles, get_ticker
except Exception as e:
    raise RuntimeError(f"Не найден backend.exchange_data: {e}")

try:
    # из твоего analysis/signals.py
    from analysis.signals import (
        generate_trading_signals,
        generate_adaptive_stop_loss_take_profit,
    )
except Exception as e:
    raise RuntimeError(f"Не найден analysis.signals: {e}")


# ================== настройки ==================
APP_TITLE        = os.getenv("APP_TITLE", "CryptoLab — Крипто-аналитика")
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "bybit")
DEFAULT_SYMBOL   = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
DEFAULT_TF       = os.getenv("DEFAULT_TF", "15")   # "15", "15m", "1h"
DEFAULT_LIMIT    = int(os.getenv("DEFAULT_LIMIT", "200"))
API_TIMEOUT      = float(os.getenv("API_TIMEOUT", "4.0"))   # сек
CACHE_TTL        = float(os.getenv("APP_CACHE_TTL", "2.0")) # быстрый локальный кэш, сек.
TEMPLATE_DIRS    = [d for d in ("templates", "tamplates") if os.path.isdir(d)]
if not TEMPLATE_DIRS:
    TEMPLATE_DIRS = ["templates"]  # будет работать даже если директории нет — просто страница-заглушка

# ================== Flask app ==================
app = Flask(
    __name__,
    static_folder="static",
    template_folder=TEMPLATE_DIRS[0]  # Flask берёт одну корневую; ниже сделаем небольшой хак для поиска в двух
)



# — небольшой хак: если шаблон не найден в основной папке, попробуем оставшиеся
_original_render = render_template
def smart_render(template_name: str, **ctx):
    searched = []
    for base in TEMPLATE_DIRS:
        path = os.path.join(base, template_name)
        if os.path.isfile(path):
            app.jinja_loader.searchpath = [base]
            return _original_render(template_name, **ctx)
        searched.append(path)
    # не нашли — мягкая заглушка
    return render_template_string(
        "<h2>Шаблон не найден</h2><p>{{name}}</p><pre>{{paths|tojson(indent=2)}}</pre>",
        name=template_name, paths=searched, **ctx
    )
render_template = smart_render  # подменяем


# ================== общий event loop (в отдельном потоке) ==================
class _AsyncRunner:
    """Единый событийный цикл в фоне — дешёвые await-вызовы в Flask (WSGI)."""
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thr = threading.Thread(target=self._loop.run_forever, name="loop-thread", daemon=True)
        self._thr.start()

    def run(self, coro, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def call_soon(self, fn, *a, **kw):
        self._loop.call_soon_threadsafe(fn, *a, **kw)

_runner = _AsyncRunner()

def arun(coro, timeout: Optional[float] = API_TIMEOUT):
    """Синхронный helper: выполнить корутину в фоне и вернуть результат/ошибку."""
    return _runner.run(coro, timeout=timeout)


# ================== утилиты ==================
def normalize_tf(tf: str) -> str:
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return tf[:-1] or "1"
    if tf.endswith("h"):
        try:
            return str(int(tf[:-1]) * 60)
        except Exception:
            return tf
    return tf

def df_to_records(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for _, r in df.iterrows():
        row = {}
        for c, v in r.items():
            if isinstance(v, pd.Timestamp):
                row[c] = v.isoformat()
            else:
                try:
                    row[c] = float(v) if isinstance(v, (int, float)) else v
                except Exception:
                    row[c] = v
        out.append(row)
    return out

def as_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, pd.DataFrame):
        return df_to_records(obj)
    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    return obj

# лёгкий TTL-кэш на запросы (в памяти процесса)
_cache: Dict[str, Tuple[float, Any]] = {}
def cache_get(key: str) -> Any:
    if not CACHE_TTL:
        return None
    item = _cache.get(key)
    if not item:
        return None
    expires, val = item
    if expires < time.time():
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl: Optional[float] = None) -> None:
    ttl = CACHE_TTL if ttl is None else ttl
    if ttl and ttl > 0:
        _cache[key] = (time.time() + ttl, val)

def cache_key(*parts: Any) -> str:
    return "|".join(map(lambda x: str(x)[:256], parts))


# ================== ошибки / ответы ==================
@app.errorhandler(404)
def _404(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "not_found", "message": "Ресурс не найден"}), 404
    return render_template("errors/404.html", title="404 — не найдено") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "errors/404.html")
    ) else (render_template_string("<h2>404</h2><p>Страница не найдена.</p>"), 404)

@app.errorhandler(500)
def _500(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "server_error", "message": str(e)}), 500
    return render_template("errors/500.html", title="500 — ошибка") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "errors/500.html")
    ) else (render_template_string("<h2>500</h2><p>Внутренняя ошибка сервера.</p>"), 500)

def json_ok(payload: Dict[str, Any], status: int = 200):
    resp = jsonify({"ok": True, **payload})
    resp.headers["Cache-Control"] = "no-store"
    return resp, status

def json_err(message: str, status: int = 400, code: str = "bad_request"):
    resp = jsonify({"ok": False, "error": code, "message": message})
    resp.headers["Cache-Control"] = "no-store"
    return resp, status


# ================== страницы ==================
@app.route("/")
def index():
    # твой index.html с {% include 'partials/_header.html' %}
    return render_template(
        "index.html",
        title=APP_TITLE,
        default_exchange=DEFAULT_EXCHANGE,
        default_symbol=DEFAULT_SYMBOL,
        default_tf=DEFAULT_TF,
        default_limit=DEFAULT_LIMIT,
    )

@app.route("/analytics")
def analytics():
    # SSR-страница с сигналами (если есть templates/signals.html)
    exchange = request.args.get("exchange", DEFAULT_EXCHANGE)
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL)
    tf_raw   = request.args.get("tf", DEFAULT_TF)
    limit    = int(request.args.get("limit", DEFAULT_LIMIT))
    tf       = normalize_tf(tf_raw)

    t0 = time.perf_counter()
    try:
        # свечи
        df = arun(get_candles(exchange, symbol, tf, limit))
        if df is None or len(df) == 0:
            return render_template_string("<h3>Пустые свечи</h3>"), 404

        df = df.rename(columns=str.lower)
        if "timestamp" not in df.columns:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index.tz_convert("UTC")
            except Exception:
                pass

        # сигналы
        payload = arun(generate_trading_signals(df, symbol=symbol))
        payload.update({
            "exchange": exchange, "symbol": symbol, "tf": tf,
            "rows": int(len(df)), "elapsed_ms": round((time.perf_counter()-t0)*1000.0, 1),
        })

        # тикер — не критично
        try:
            t = arun(get_ticker(exchange, symbol), timeout=1.0)
            if t:
                payload["ticker"] = {
                    "price": float(getattr(t, "price", None) or (t.get("price") if isinstance(t, dict) else 0.0)),
                    "change_percent_24h": float(getattr(t, "change_percent_24h", 0.0) or (t.get("change_percent_24h") if isinstance(t, dict) else 0.0)),
                    "high_24h": float(getattr(t, "high_24h", 0.0) or (t.get("high_24h") if isinstance(t, dict) else 0.0)),
                    "low_24h": float(getattr(t, "low_24h", 0.0) or (t.get("low_24h") if isinstance(t, dict) else 0.0)),
                }
        except Exception:
            pass

        # если есть шаблон — отрисуем; если нет, вернём JSON
        if os.path.isfile(os.path.join(TEMPLATE_DIRS[0], "signals.html")):
            return render_template(
                "signals.html",
                title=f"{APP_TITLE} · {symbol} {tf}",
                payload=payload,
                overall=payload.get("overall_signal"),
                confidence=payload.get("confidence"),
                score=payload.get("confidence_score"),
                filters=payload.get("filters", {}),
                signals=payload.get("signals", []),
                ticker=payload.get("ticker", {}),
                params={"exchange":exchange,"symbol":symbol,"tf":tf,"limit":limit},
            )
        return jsonify(payload)
    except Exception as e:
        return render_template_string("<h3>Ошибка аналитики</h3><pre>{{e}}</pre>", e=e), 500


# — простые заглушки под ссылки из хедера (чтобы не 404)
@app.route("/patterns")
def patterns():
    return render_template("stubs/patterns.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/patterns.html")
    ) else render_template_string("<h2>Паттерны</h2><p>Страница в разработке.</p>")

@app.route("/dashboard")
def dashboard():
    return render_template("stubs/dashboard.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/dashboard.html")
    ) else render_template_string("<h2>Дашборд</h2><p>В разработке.</p>")

@app.route("/docs")
def docs():
    return render_template("stubs/docs.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/docs.html")
    ) else render_template_string("<h2>Документация</h2><p>Скоро.</p>")

@app.route("/community")
def community():
    return render_template("stubs/community.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/community.html")
    ) else render_template_string("<h2>Комьюнити</h2><p>Скоро.</p>")

@app.route("/billing")
def billing():
    return render_template("stubs/billing.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/billing.html")
    ) else render_template_string("<h2>Подписка</h2><p>Скоро.</p>")

@app.route("/legal/terms")
def legal_terms():
    return render_template("stubs/terms.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/terms.html")
    ) else render_template_string("<h2>Terms</h2>")

@app.route("/legal/privacy")
def legal_privacy():
    return render_template("stubs/privacy.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/privacy.html")
    ) else render_template_string("<h2>Privacy</h2>")

@app.route("/changelog")
def changelog():
    return render_template("stubs/changelog.html") if os.path.isfile(
        os.path.join(TEMPLATE_DIRS[0], "stubs/changelog.html")
    ) else render_template_string("<h2>Changelog</h2>")

# соц/логин — заглушка, просто чтобы ссылка не падала
@app.route("/auth/google")
def auth_google():
    return render_template_string("<h2>Вход через Google</h2><p>Интеграция в процессе.</p>")


# ================== API ==================
@app.route("/health")
def health():
    return "ok", 200

@app.route("/api/candles")
def api_candles():
    exchange = request.args.get("exchange", DEFAULT_EXCHANGE)
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL)
    tf       = normalize_tf(request.args.get("tf", DEFAULT_TF))
    limit    = int(request.args.get("limit", DEFAULT_LIMIT))

    ck = cache_key("candles", exchange, symbol, tf, limit)
    cached = cache_get(ck)
    if cached is not None:
        return json_ok({"exchange":exchange,"symbol":symbol,"tf":tf,"rows":cached["rows"],"data":cached["data"]})

    try:
        df = arun(get_candles(exchange, symbol, tf, limit))
        if df is None or len(df) == 0:
            return json_err("Пустые данные свечей", 404, code="no_data")

        df = df.rename(columns=str.lower)
        if "timestamp" not in df.columns:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index.tz_convert("UTC")
            except Exception:
                pass

        data = df_to_records(df.tail(limit))
        payload = {"exchange":exchange,"symbol":symbol,"tf":tf,"rows":len(df),"data":data}
        cache_set(ck, payload)
        return json_ok(payload)
    except Exception as e:
        return json_err(f"/api/candles error: {e}", 500, code="server_error")

@app.route("/api/ticker")
def api_ticker():
    exchange = request.args.get("exchange", DEFAULT_EXCHANGE)
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL)

    ck = cache_key("ticker", exchange, symbol)
    cached = cache_get(ck)
    if cached is not None:
        return json_ok(cached)

    try:
        t = arun(get_ticker(exchange, symbol))
        if not t:
            return json_err("Тикер не найден", 404, code="no_ticker")
        payload = {
            "symbol": symbol,
            "price": float(getattr(t, "price", None) or (t.get("price") if isinstance(t, dict) else 0.0)),
            "volume": float(getattr(t, "volume", 0.0) or (t.get("volume") if isinstance(t, dict) else 0.0)),
            "change_percent_24h": float(getattr(t, "change_percent_24h", 0.0) or (t.get("change_percent_24h") if isinstance(t, dict) else 0.0)),
            "high_24h": float(getattr(t, "high_24h", 0.0) or (t.get("high_24h") if isinstance(t, dict) else 0.0)),
            "low_24h": float(getattr(t, "low_24h", 0.0) or (t.get("low_24h") if isinstance(t, dict) else 0.0)),
            "timestamp": getattr(t, "timestamp", None) or (t.get("timestamp") if isinstance(t, dict) else None),
        }
        cache_set(ck, payload)
        return json_ok(payload)
    except Exception as e:
        return json_err(f"/api/ticker error: {e}", 500, code="server_error")

@app.route("/api/signals")
def api_signals():
    exchange = request.args.get("exchange", DEFAULT_EXCHANGE)
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL)
    tf       = normalize_tf(request.args.get("tf", DEFAULT_TF))
    limit    = int(request.args.get("limit", DEFAULT_LIMIT))

    # кэшируем по параметрам (внутри analysis/signals есть свой кэш, но этот — фронтовый)
    ck = cache_key("signals", exchange, symbol, tf, limit)
    cached = cache_get(ck)
    if cached is not None:
        return json_ok(cached)

    t0 = time.perf_counter()
    try:
        df = arun(get_candles(exchange, symbol, tf, limit))
        if df is None or len(df) == 0:
            return json_err("Пустые свечи", 404, code="no_data")

        df = df.rename(columns=str.lower)
        if "timestamp" not in df.columns:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index.tz_convert("UTC")
            except Exception:
                pass

        res = arun(generate_trading_signals(df, symbol=symbol))
        payload = {
            **res,
            "exchange": exchange,
            "symbol": symbol,
            "tf": tf,
            "rows": int(len(df)),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

        # ленивый тикер
        try:
            t = arun(get_ticker(exchange, symbol), timeout=1.0)
            if t:
                payload["ticker"] = {
                    "price": float(getattr(t, "price", None) or (t.get("price") if isinstance(t, dict) else 0.0)),
                    "change_percent_24h": float(getattr(t, "change_percent_24h", 0.0) or (t.get("change_percent_24h") if isinstance(t, dict) else 0.0)),
                    "high_24h": float(getattr(t, "high_24h", 0.0) or (t.get("high_24h") if isinstance(t, dict) else 0.0)),
                    "low_24h": float(getattr(t, "low_24h", 0.0) or (t.get("low_24h") if isinstance(t, dict) else 0.0)),
                }
        except Exception:
            pass

        cache_set(ck, payload)
        return json_ok(payload)
    except Exception as e:
        return json_err(f"/api/signals error: {e}", 500, code="server_error")

@app.route("/api/sltp")
def api_sltp():
    exchange = request.args.get("exchange", DEFAULT_EXCHANGE)
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL)
    tf       = normalize_tf(request.args.get("tf", DEFAULT_TF))
    limit    = int(request.args.get("limit", DEFAULT_LIMIT))
    entry    = request.args.get("entry_price", type=float)
    side     = request.args.get("side", type=str)

    if entry is None or side not in ("buy", "sell"):
        return json_err("entry_price и side=buy|sell обязательны", 400)

    try:
        df = arun(get_candles(exchange, symbol, tf, limit))
        if df is None or len(df) == 0:
            return json_err("Пустые свечи", 404, code="no_data")

        df = df.rename(columns=str.lower)
        if "timestamp" not in df.columns:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index.tz_convert("UTC")
            except Exception:
                pass

        res = arun(generate_adaptive_stop_loss_take_profit(
            market_data=df,
            entry_price=float(entry),
            position_type="long" if side == "buy" else "short",
            symbol=symbol,
        ))
        return json_ok({"exchange":exchange, "symbol":symbol, "tf":tf, **res})
    except Exception as e:
        return json_err(f"/api/sltp error: {e}", 500, code="server_error")
#дрочит ико
@app.route('/favicon.ico')
def favicon():
    from flask import send_from_directory
    p = os.path.join(app.root_path, 'static')
    # положи иконку в static/favicon.ico (или вернётся 204)
    try:
        return send_from_directory(p, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except Exception:
        return ("", 204)
# ================== запуск ==================
# flask --app app run --host 0.0.0.0 --port 8000 --debug
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
