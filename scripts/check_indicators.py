# scripts/check_indicators.py
import asyncio
import os
import pandas as pd

# Без Redis, чтобы не шумело в логе
os.environ.setdefault("EXDATA_REDIS", "0")

from backend.exchange_data import get_candles
from analysis.indicators.indicators import TechnicalIndicators

SYMBOL = "BTCUSDT"
EXCHANGE = "bybit"
TIMEFRAME = "15m"
LIMIT = 200

async def main():
    print(f"[fetch] {EXCHANGE} {SYMBOL} {TIMEFRAME} limit={LIMIT}")
    df: pd.DataFrame = await get_candles(EXCHANGE, SYMBOL, TIMEFRAME, LIMIT)
    print("rows:", len(df), "last_close:", df["close"].iloc[-1])

    ti = TechnicalIndicators(df)

    sma20 = ti.sma(20)
    ema20 = ti.ema(20)
    rsi14 = ti.rsi(14)
    macd_line, macd_sig, macd_hist = ti.macd()
    bb_up, bb_mid, bb_lo = ti.bollinger_bands()
    atr14 = ti.atr(14)
    k, d = ti.stochastic()

    print("[indicators] tail snapshot")
    print("SMA20:", float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None)
    print("EMA20:", float(ema20.iloc[-1]) if pd.notna(ema20.iloc[-1]) else None)
    print("RSI14:", float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else None)
    print("BB:", float(bb_up[-1]), float(bb_mid[-1]), float(bb_lo[-1]))
    print("ATR14:", float(atr14[-1]))
    print("STOCH:", float(k[-1]), float(d[-1]))

if __name__ == "__main__":
    asyncio.run(main())
