from typing import List, Dict, Any
from .indicators import ema, macd, rsi, atr

def detect_signals(closes: List[float], strategy: str = 'classic') -> Dict[str, Any]:
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    macd_line, sig_line, hist = macd(closes)
    r = rsi(closes)
    out = []

    # EMA cross
    for i in range(1, len(closes)):
        if e9[i-1] <= e21[i-1] and e9[i] > e21[i]:
            out.append({"type":"EMA_CROSS_UP","idx": i})
        if e9[i-1] >= e21[i-1] and e9[i] < e21[i]:
            out.append({"type":"EMA_CROSS_DOWN","idx": i})
    # MACD cross
    for i in range(1, len(closes)):
        if macd_line[i-1] <= sig_line[i-1] and macd_line[i] > sig_line[i]:
            out.append({"type":"MACD_CROSS_UP","idx": i})
        if macd_line[i-1] >= sig_line[i-1] and macd_line[i] < sig_line[i]:
            out.append({"type":"MACD_CROSS_DOWN","idx": i})
    # RSI zones
    for i, rv in enumerate(r):
        if rv >= 70: out.append({"type":"RSI_OVERBOUGHT","idx": i})
        if rv <= 30: out.append({"type":"RSI_OVERSOLD","idx": i})

    return {
        "ema9": e9,
        "ema21": e21,
        "macd": macd_line,
        "macd_signal": sig_line,
        "macd_hist": hist,
        "rsi": r,
        "alerts": out
    }


def sma(values: List[float], period: int) -> List[float]:
    if period <= 0 or not values: return []
    out=[]; s=0.0
    for i,v in enumerate(values):
        s += v
        if i>=period: s -= values[i-period]
        out.append(s/period if i>=period-1 else v)
    return out

def stddev(values: List[float], period: int) -> List[float]:
    out=[]; from math import sqrt
    for i in range(len(values)):
        if i<period-1: out.append(0.0); continue
        win = values[i-period+1:i+1]
        m = sum(win)/period
        out.append((sum((x-m)*(x-m) for x in win)/period)**0.5)
    return out

def bollinger(values: List[float], period: int = 20, k: float = 2.0):
    m = sma(values, period)
    s = stddev(values, period)
    upper = [mi + k*si for mi,si in zip(m,s)]
    lower = [mi - k*si for mi,si in zip(m,s)]
    return m, upper, lower

def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int=14, d_period: int=3):
    k_vals=[]
    for i in range(len(closes)):
        lo = min(lows[max(0,i-k_period+1):i+1])
        hi = max(highs[max(0,i-k_period+1):i+1])
        val = 100.0 * (closes[i]-lo) / (hi-lo) if hi!=lo else 0.0
        k_vals.append(val)
    d_vals = sma(k_vals, d_period)
    return k_vals, d_vals
