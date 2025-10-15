# scripts/check_signals.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict

import pandas as pd

# project imports
try:
    from analysis.signals import generate_trading_signals
except Exception as e:
    print("Не найден analysis.signals.generate_trading_signals:", e, file=sys.stderr)
    sys.exit(2)

try:
    from backend.exchange_data import get_candles, get_ticker
except Exception as e:
    print("Не найден backend.exchange_data.{get_candles,get_ticker}:", e, file=sys.stderr)
    sys.exit(2)


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


async def fetch_df(exchange: str, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    print(f"[fetch] {exchange} {symbol} {tf} limit={limit}")
    df = await get_candles(exchange, symbol, tf, limit)
    if df is None or len(df) == 0:
        raise RuntimeError("Пустые свечи")
    # нормализуем колонки
    df = df.rename(columns=str.lower)
    # убедимся, что есть timestamp в UTC (если его нет — создадим из индекса)
    if "timestamp" not in df.columns:
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index.tz_convert("UTC")
            else:
                df["timestamp"] = pd.to_datetime(df.get("time") or df.get("date"), utc=True, errors="coerce")
        except Exception:
            pass
    print(f"rows: {len(df)} last_close: {_fmt(df['close'].iloc[-1])}")
    return df


def print_result(res: Dict[str, Any], *, max_signals: int = 10) -> None:
    sigs = res.get("signals", []) or []
    overall = res.get("overall_signal")
    conf = res.get("confidence")
    score = res.get("confidence_score")
    print("[signals] calculating…")
    print(f"overall: {overall} conf: {conf} score: {_fmt(score)}")
    print(f"signals_total: {len(sigs)}")

    # печатаем топ-N по strength*weight
    sigs_sorted = sorted(
        sigs,
        key=lambda s: float(s.get("strength", 0)) * float(s.get("weight", 1.0)),
        reverse=True,
    )[:max_signals]

    for s in sigs_sorted:
        ind = s.get("indicator", "?")
        st = int(s.get("strength", 0))
        t = s.get("type")
        c = s.get("confidence")
        msg = s.get("message", "")
        print(f"- {ind}: {t} ({c}) str={st} msg={msg}")

    # полезные фильтры/диагностика
    filters = res.get("filters", {}) or {}
    ticker = res.get("ticker") or {}
    print("[filters] →", json.dumps(filters, ensure_ascii=False))
    if ticker:
        print("[ticker] →", json.dumps(ticker, ensure_ascii=False))


async def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка генерации сигналов (analysis/signals.py)")
    parser.add_argument("--exchange", "-x", default=os.getenv("EXCHANGE", "bybit"), help="Биржа (bybit/binance/…)")
    parser.add_argument("--symbol", "-s", default=os.getenv("SYMBOL", "DOGEUSDT"), help="Тикер")
    parser.add_argument("--tf", "-t", default=os.getenv("TF", "15"), help="Таймфрейм (например: 15 | 15m | 60)")
    parser.add_argument("--limit", "-l", type=int, default=int(os.getenv("LIMIT", "200")), help="Количество свечей")
    parser.add_argument("--json", action="store_true", help="Вывести JSON-результат")
    parser.add_argument("--runs", type=int, default=2, help="Сколько прогонов (для оценки кэша/скорости)")
    args = parser.parse_args()

    # заглушки Redis-ENV для наглядности (если используется в exchange_data)
    for key in ("EXDATA_REDIS", "SIGNALS_STRICT", "SIGNALS_MTF"):
        if key in os.environ:
            print(f"{key}={os.environ[key]}")

    # приводим tf к формату, который понимает exchange_data
    tf = str(args.tf).lower()
    if tf.endswith("m"):
        tf = tf[:-1]  # "15m" -> "15"
    elif tf.endswith("h"):
        tf = str(int(tf[:-1]) * 60)  # "1h" -> "60"

    # загрузка данных
    df = await fetch_df(args.exchange, args.symbol, tf, args.limit)

    # пробуем также дернуть тикер (не обязательно)
    ticker_data = {}
    try:
        t = await get_ticker(args.exchange, args.symbol)
        if t:
            ticker_data = {
                "price": float(getattr(t, "price", None) or t.get("price")),
                "change_percent_24h": float(getattr(t, "change_percent_24h", None) or t.get("change_percent_24h", 0.0)),
                "high_24h": float(getattr(t, "high_24h", None) or t.get("high_24h", 0.0)),
                "low_24h": float(getattr(t, "low_24h", None) or t.get("low_24h", 0.0)),
            }
    except Exception as e:
        print("ticker fetch error:", e, file=sys.stderr)

    # несколько прогонов — видим ускорение от кэша signals.py
    last_res = None
    for i in range(max(1, args.runs)):
        t0 = time.perf_counter()
        res = await generate_trading_signals(df, symbol=args.symbol)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # подмешаем тикер (для удобства вывода)
        if ticker_data:
            res["ticker"] = ticker_data

        print(f"\n[run {i+1}/{args.runs}] elapsed={dt_ms:.1f}ms")
        print_result(res)

        last_res = res

    # произвольный JSON-дамп последнего прогона (по запросу)
    if args.json and last_res is not None:
        print("\n=== JSON ===")
        print(json.dumps(last_res, ensure_ascii=False, default=str, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
