from typing import List, Dict, Any
import math

def ema(values: List[float], period: int) -> List[float]:
    if period <= 0 or not values:
        return []
    k = 2/(period+1)
    out: List[float] = []
    ema_prev = None
    for v in values:
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v * k + ema_prev * (1-k)
        out.append(ema_prev)
    return out

def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [ (f - s) if (f is not None and s is not None) else None for f,s in zip(ema_fast, ema_slow) ]
    # replace None by earliest valid
    macd_line = [x if x is not None else (macd_line[i-1] if i>0 else 0.0) for i,x in enumerate(macd_line)]
    signal_line = ema([x for x in macd_line], signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < 2:
        return []
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))
    def _ema(seq, p):
        return ema(seq, p)
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    out = []
    for g,l in zip(avg_gain, avg_loss):
        rs = (g / l) if l != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs)) if math.isfinite(rs) else 100.0
        out.append(rsi)
    return out

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    if not highs or not lows or not closes:
        return []
    trs = []
    prev_close = closes[0]
    for h,l,c in zip(highs, lows, closes):
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    return ema(trs, period)
