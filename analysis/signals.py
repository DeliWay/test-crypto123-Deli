# analysis/signals.py
"""
ULTRA SIGNALS — усиленные, устойчивые и производительные торговые сигналы

+ Безопасная индексация (никаких KeyError: -1)
+ Адаптивные уровни RSI/Supertrend/SL-TP через analysis/adaptive
+ Качественные фильтры: ADX, VWAP bias, CHOP (пила), wick-антишип
+ Трендовые модули: Donchian breakout, Keltner Channels, Supertrend
+ S/R уровни (пивоты свингов) для микро-оценки «пространства»
+ BB внутри Keltner (squeeze) и «выстрелы» при выходе
+ Pullback-бонус к EMA21/ATR в сторону тренда
+ Время суток (UTC) как маленький байас к уверенности
+ Детектор затишья (quiet/stagnation) и «бей в выход»
+ MTF подтверждение (например, 15m сигнал + 1h режим/статус)
+ STRICT-режим: консенсус, пороги уверенности, cooldown от flip-flop
+ Лёгкий локальный кэш для ускорения горячих вызовов

API:
- generate_trading_signals(market_data: pd.DataFrame, symbol: Optional[str]) -> Dict[str, Any]
- generate_adaptive_stop_loss_take_profit(market_data: pd.DataFrame, entry_price: float, position_type: str, symbol: Optional[str]) -> Dict[str, Any]
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import os
import asyncio
import hashlib
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# --- проектные модули ---
from analysis.indicators.indicators import TechnicalIndicators
from analysis.adaptive import (
    AssetProfile,
    calculate_asset_profile,
    adaptive_rsi_levels,
    adaptive_supertrend_multiplier,
    adaptive_stop_loss_take_profit,
)

logger = logging.getLogger(__name__)

# =============================
# Безопасные хелперы индексации
# =============================

def _to_series(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def _last(x):
    try:
        return x.iloc[-1]         # pandas.Series
    except AttributeError:
        return x[-1]              # numpy/list

def _prev(x, k: int = 1):
    try:
        return x.iloc[-1 - k]
    except AttributeError:
        return x[-1 - k]

def _last_valid(x):
    s = _to_series(x)
    s = s[~s.isna()]
    return np.nan if s.empty else s.iloc[-1]

def _prev_valid(x, k: int = 1):
    s = _to_series(x)
    s = s[~s.isna()]
    if len(s) <= k:
        return np.nan
    return s.iloc[-1 - k]

def _nan_or(val, fallback: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else fallback
    except Exception:
        return fallback

# =============================
# Конфиг и ENV-тумблеры
# =============================

MIN_DATA_POINTS = int(os.getenv("SIGNALS_MIN_DATA_POINTS", "60"))
CACHE_TTL = int(os.getenv("SIGNALS_CACHE_TTL", "30"))
VOL_SPIKE_Z = float(os.getenv("SIGNALS_VOL_SPIKE_Z", "2.0"))
BB_THRESHOLD = float(os.getenv("SIGNALS_BB_THRESHOLD", "0.02"))
MIN_TREND_STRENGTH = float(os.getenv("SIGNALS_MIN_TREND_STRENGTH", "0.3"))

# STRICT
STRICT = bool(int(os.getenv("SIGNALS_STRICT", "1")))
CONSENSUS_MIN = int(os.getenv("SIGNALS_CONSENSUS_MIN", "2"))
MIN_STRENGTH_MEAN = float(os.getenv("SIGNALS_MIN_STRENGTH_MEAN", "55.0"))
MIN_CONF_SCORE = float(os.getenv("SIGNALS_MIN_CONF_SCORE", "0.40"))
REQUIRE_REGIME_ALIGNMENT = bool(int(os.getenv("SIGNALS_REQUIRE_REGIME", "1")))
COOLDOWN_BARS = int(os.getenv("SIGNALS_COOLDOWN_BARS", "2"))
PERSIST_BARS = int(os.getenv("SIGNALS_PERSIST_BARS", "0"))

# strong_* требования
MIN_SIGNALS_FOR_STRONG = int(os.getenv("SIGNALS_MIN_FOR_STRONG", "3"))
MIN_UNIQUE_INDICATORS_FOR_STRONG = int(os.getenv("SIGNALS_MIN_UNIQUE_FOR_STRONG", "2"))

# ADX порог
MIN_ADX_TREND = float(os.getenv("SIGNALS_MIN_ADX", "20.0"))

# CHOP и squeeze
CHOP_HIGH = float(os.getenv("SIGNALS_CHOP_HIGH", "61.8"))  # «пила», глушим тренд-логику
CHOP_LOW  = float(os.getenv("SIGNALS_CHOP_LOW", "38.2"))   # чистый тренд, усиливаем

# S/R дистанции
SR_FAR = float(os.getenv("SIGNALS_SR_FAR", "0.0040"))      # 0.40%
SR_NEAR = float(os.getenv("SIGNALS_SR_NEAR", "0.0015"))    # 0.15%

# Wick-антишип
WICK_Z = float(os.getenv("SIGNALS_WICK_Z", "2.0"))
WICK_LOOKBACK = int(os.getenv("SIGNALS_WICK_LOOKBACK", "50"))

# MTF подтверждение
MTF_ON = bool(int(os.getenv("SIGNALS_MTF", "1")))
MTF_EXCHANGE = os.getenv("SIGNALS_MTF_EXCHANGE", "bybit")  # если не передаём снаружи
MTF_TF = os.getenv("SIGNALS_MTF_TF", "60")                  # старший ТФ, например "60" (1h)
MTF_ST_MULT = float(os.getenv("SIGNALS_MTF_ST_MULT", "3.0"))

# Время суток (UTC) — маленький байас
HOUR_BONUS_ACTIVE = bool(int(os.getenv("SIGNALS_HOUR_BONUS", "1")))

# Затишье/стагнация
QUIET_BB_WIDTH = float(os.getenv("SIGNALS_QUIET_BB_WIDTH", "0.008"))  # 0.8% относительной ширины BB
QUIET_MIN_BARS = int(os.getenv("SIGNALS_QUIET_MIN_BARS", "8"))

@dataclass
class Weights:
    ema: float = 1.0
    macd: float = 1.0
    rsi: float = 1.0
    bb: float = 1.0
    vol: float = 1.0
    trend: float = 1.0
    supertrend: float = 1.0
    keltner: float = 1.0
    donchian: float = 1.2  # трендовые пробои поощряем

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"

class SignalConfidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class MarketRegime(Enum):
    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

# =============================
# Локальный кэш (легковесный)
# =============================

class _Cache:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ts: Dict[str, float] = {}
    def key(self, func: str, *parts: Any) -> str:
        chunks = []
        for p in parts:
            try:
                chunks.append(json.dumps(p, sort_keys=True, default=str))
            except Exception:
                chunks.append(str(p))
        raw = f"{func}|{'|'.join(chunks)}"
        return f"{func}:{hashlib.md5(raw.encode()).hexdigest()}"
    def get(self, key: str, ttl: int) -> Optional[Any]:
        ts = self._ts.get(key)
        if ts is None:
            return None
        if (datetime.now().timestamp() - ts) > ttl:
            return None
        return self._data.get(key)
    def set(self, key: str, value: Any):
        self._data[key] = value
        self._ts[key] = datetime.now().timestamp()

_cache = _Cache()
_last_strong: Dict[str, Tuple[str, int]] = {}  # cooldown

# =============================
# Режим рынка / веса
# =============================

def determine_market_regime(prices: np.ndarray, volumes: Optional[np.ndarray] = None, lookback: int = 80) -> MarketRegime:
    if len(prices) < max(lookback, 20):
        return MarketRegime.RANGING
    p = prices[-lookback:]
    if np.allclose(p, p[0]):
        return MarketRegime.RANGING
    ret = np.diff(p) / np.where(p[:-1] != 0, p[:-1], np.nan)
    vol = float(np.nanstd(ret) * np.sqrt(252) * 100)
    x = np.arange(len(p))
    try:
        slope, _ = np.polyfit(x, p, 1)
    except Exception:
        slope = 0.0
    ts = abs(slope) / (np.nanmean(p) + 1e-12) * 100 * np.sqrt(len(p))
    if ts > 1.0 and slope > 0:
        return MarketRegime.TREND_BULLISH
    if ts > 1.0 and slope < 0:
        return MarketRegime.TREND_BEARISH
    if vol > 40:
        return MarketRegime.VOLATILE
    if vol < 15:
        return MarketRegime.LOW_VOLATILITY
    return MarketRegime.RANGING

def get_regime_weights(regime: MarketRegime) -> Weights:
    if regime == MarketRegime.TREND_BULLISH:
        return Weights(ema=1.2, macd=1.1, rsi=0.9, bb=0.7, vol=1.0, trend=1.3, supertrend=1.2, keltner=1.0, donchian=1.3)
    if regime == MarketRegime.TREND_BEARISH:
        return Weights(ema=1.2, macd=1.1, rsi=0.9, bb=0.7, vol=1.0, trend=1.3, supertrend=1.2, keltner=1.0, donchian=1.3)
    if regime == MarketRegime.VOLATILE:
        return Weights(ema=0.8, macd=0.9, rsi=1.1, bb=1.3, vol=1.2, trend=0.8, supertrend=1.0, keltner=1.2, donchian=0.9)
    if regime == MarketRegime.LOW_VOLATILITY:
        return Weights(ema=0.8, macd=0.9, rsi=1.1, bb=1.3, vol=0.9, trend=0.7, supertrend=0.9, keltner=1.0, donchian=1.0)
    return Weights(ema=0.9, macd=0.9, rsi=1.3, bb=1.4, vol=1.0, trend=0.8, supertrend=1.0, keltner=1.0, donchian=1.1)

# =============================
# Индикаторные кирпичики и доп-метрики
# =============================

def _trend_strength_segment(seg: np.ndarray) -> float:
    x = np.arange(len(seg))
    try:
        slope, _ = np.polyfit(x, seg, 1)
    except Exception:
        return 0.0
    meanp = float(np.nanmean(seg))
    if not np.isfinite(meanp) or abs(meanp) < 1e-12:
        return 0.0
    return float(slope / meanp * 100 * np.sqrt(len(seg)))

def _ema(arr, span):
    s = _to_series(arr).astype(float)
    return s.ewm(span=span, adjust=False, min_periods=span).mean().values

# ADX (Wilder)
def _adx(high, low, close, period: int = 14):
    high = _to_series(high).astype(float).values
    low  = _to_series(low).astype(float).values
    close= _to_series(close).astype(float).values
    n = len(close)
    if n < period + 2:
        na = np.full(n, np.nan)
        return na, na, na

    up = high[1:] - high[:-1]
    dn = low[:-1]  - low[1:]
    plusDM  = np.where((up > dn) & (up > 0), up, 0.0)
    minusDM = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

    def _wilder(arr):
        out = np.empty_like(arr)
        out[:period] = np.nan
        sm = np.nansum(arr[:period])
        out[period] = sm
        for i in range(period + 1, len(arr)):
            sm = sm - (sm / period) + arr[i]
            out[i] = sm
        return out

    trN = _wilder(tr)
    plusN = 100 * (_wilder(plusDM) / (trN + 1e-12))
    minusN= 100 * (_wilder(minusDM) / (trN + 1e-12))
    dx = 100 * (np.abs(plusN - minusN) / (plusN + minusN + 1e-12))

    adx = np.empty_like(dx); adx[:] = np.nan
    start = np.where(~np.isnan(dx))[0]
    if start.size:
        s = start[0] + period
        if s < len(dx):
            adx[s] = np.nanmean(dx[start[0]:s])
            for i in range(s+1, len(dx)):
                adx[i] = ((adx[i-1]*(period-1)) + dx[i]) / period

    pad = np.array([np.nan])
    return np.r_[pad, plusN], np.r_[pad, minusN], np.r_[pad, adx]

def analyze_adx_filter(high, low, close, min_adx: float = 22.0):
    _, _, adx = _adx(high, low, close, 14)
    a = _last_valid(adx)
    if np.isnan(a):
        return {"ok": False, "value": np.nan, "message": "ADX NaN"}
    return {"ok": bool(a >= min_adx), "value": float(a), "message": f"ADX={a:.1f} (>= {min_adx})" if a>=min_adx else f"ADX={a:.1f} (< {min_adx})"}

# Keltner Channels
def keltner_channels(high, low, close, ema_period=20, atr_mult=1.5):
    high = _to_series(high).astype(float).values
    low  = _to_series(low).astype(float).values
    close= _to_series(close).astype(float).values
    tr1 = (high - low)
    tr2 = np.abs(high - np.r_[np.nan, close[:-1]])
    tr3 = np.abs(low  - np.r_[np.nan, close[:-1]])
    tr = np.nanmax(np.c_[tr1, tr2, tr3], axis=1)
    atr = _to_series(tr).ewm(alpha=1/ema_period, adjust=False, min_periods=ema_period).mean().values
    mid = _ema(close, ema_period)
    up = mid + atr_mult*atr
    lo = mid - atr_mult*atr
    return up, mid, lo

def analyze_keltner_signals(high, low, close, ema_period=20, atr_mult=1.5):
    up, md, lo = keltner_channels(high, low, close, ema_period, atr_mult)
    p = float(_last(close)); u = float(_last(up)); m = float(_last(md)); l = float(_last(lo))
    out = []
    if p > u:
        out.append({"type":"sell","confidence":"medium","indicator":"Keltner","message":"Above KC upper (exhaustion)","strength":65})
    elif p < l:
        out.append({"type":"buy","confidence":"medium","indicator":"Keltner","message":"Below KC lower (exhaustion)","strength":65})
    else:
        out.append({"type":"buy" if p>m else "sell","confidence":"low","indicator":"Keltner","message":"Bias vs middle","strength":50})
    return out

# Donchian
def donchian(high, low, period=20):
    h = _to_series(high).astype(float)
    l = _to_series(low).astype(float)
    up = h.rolling(period, min_periods=period).max().values
    lo = l.rolling(period, min_periods=period).min().values
    return up, lo

def analyze_donchian_breakout(high, low, close, period=20):
    up, lo = donchian(high, low, period)
    p = float(_last(close))
    u = float(_last(up)); l = float(_last(lo))
    if np.isfinite(u) and p > u:
        return {"type":"strong_buy","confidence":"high","indicator":"Donchian","message":f"Breakout > {period}-high","strength":85}
    if np.isfinite(l) and p < l:
        return {"type":"strong_sell","confidence":"high","indicator":"Donchian","message":f"Breakdown < {period}-low","strength":85}
    return None

# VWAP
def vwap(high, low, close, volume):
    h = _to_series(high).astype(float).values
    l = _to_series(low).astype(float).values
    c = _to_series(close).astype(float).values
    v = _to_series(volume).astype(float).values if volume is not None else None
    if v is None or len(v) != len(c):
        return np.full(len(c), np.nan)
    tp = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    cum_vtp = np.cumsum(v * tp)
    return cum_vtp / (cum_v + 1e-12)

def anchored_vwap(close, volume, start_idx: int):
    c = _to_series(close).astype(float).values
    v = _to_series(volume).astype(float).values if volume is not None else None
    if v is None or start_idx >= len(c):
        return np.full(len(c), np.nan)
    av = np.full(len(c), np.nan)
    tp = c
    vv = v[start_idx:].copy()
    tt = tp[start_idx:].copy()
    cum_v = np.cumsum(vv); cum_vt = np.cumsum(vv * tt)
    av[start_idx:] = cum_vt / (cum_v + 1e-12)
    return av

def analyze_vwap_filter(df):
    if "volume" not in df.columns:
        return None
    v = vwap(df["high"], df["low"], df["close"], df["volume"])
    p = float(df["close"].iloc[-1])
    vv = float(v[-1]) if np.isfinite(v[-1]) else np.nan
    if np.isnan(vv):
        return None
    if p >= vv:
        return {"bias":"up","message":f"Price ≥ VWAP ({p:.2f}≥{vv:.2f})"}
    else:
        return {"bias":"down","message":f"Price < VWAP ({p:.2f}<{vv:.2f})"}

# CHOP (Choppiness)
def choppiness_index(high, low, close, period=14):
    h = _to_series(high).values; l=_to_series(low).values; c=_to_series(close).values
    if len(c) < period+1: return np.nan
    tr1 = (h[1:] - l[1:])
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.nanmax(np.c_[tr1,tr2,tr3], axis=1)
    sum_tr = np.sum(tr[-period:])
    high_n = np.max(h[-period:])
    low_n  = np.min(l[-period:])
    denom = np.log10(high_n - low_n + 1e-12)
    if denom <= 0: return np.nan
    chop = 100 * np.log10(sum_tr / (high_n - low_n + 1e-12)) / denom
    return float(chop)

# Squeeze: BB внутри Keltner
def boll_keltner_squeeze(high, low, close, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5):
    ti_tmp = TechnicalIndicators(pd.DataFrame({"open":[0]*len(close),"high":high,"low":low,"close":close}))
    bb = ti_tmp.bollinger_bands(bb_period, bb_std)
    ku, km, kl = keltner_channels(high, low, close, kc_period, kc_mult)
    bu, bm, bl = (bb["upper"], bb["middle"], bb["lower"]) if isinstance(bb, dict) else bb
    width_bb = (bu - bl) / (bm + 1e-12)
    width_kc = (ku - kl) / (km + 1e-12)
    in_squeeze = bool(width_bb[-1] < width_kc[-1])
    return in_squeeze, float(width_bb[-1])

# Свинги → S/R уровни
def _swings(highs, lows, lookback=20):
    h = _to_series(highs).values; l=_to_series(lows).values
    pivots_hi, pivots_lo = [], []
    for i in range(lookback, len(h)-lookback):
        if h[i] == np.max(h[i-lookback:i+lookback+1]): pivots_hi.append((i, h[i]))
        if l[i] == np.min(l[i-lookback:i+lookback+1]): pivots_lo.append((i, l[i]))
    return pivots_hi[-10:], pivots_lo[-10:]

def nearest_sr(close, highs, lows, lookback=20):
    p = float(_last(close))
    piv_hi, piv_lo = _swings(highs, lows, lookback)
    up = min((abs(ph - p), ph) for _, ph in piv_hi)[1] if piv_hi else np.nan
    lo = min((abs(pl - p), pl) for _, pl in piv_lo)[1] if piv_lo else np.nan
    return up, lo, p

# Wick-антишип
def wick_ratio(open_, high, low, close):
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    rng = (high - low) + 1e-12
    return float((upper + lower) / rng), float(body / rng)

def is_wicky_bar(df, z_thr=WICK_Z, lookback=WICK_LOOKBACK):
    df = df.tail(lookback) if len(df) >= lookback else df
    wr = []
    for _, r in df.iterrows():
        w, _ = wick_ratio(r["open"], r["high"], r["low"], r["close"])
        wr.append(w)
    wr = np.array(wr, dtype=float)
    if len(wr) < 5: return False
    mu, sd = np.nanmean(wr), np.nanstd(wr)
    if sd < 1e-12: return False
    return (wr[-1] - mu) / (sd + 1e-12) > z_thr

# Pullback-бонус
def trend_pullback_bonus(close, ema21, atr_like: np.ndarray, side: str, mult: float = 1.0):
    try:
        p = float(_last(close)); e = float(_last(ema21))
    except Exception:
        return 0
    a = float(_last(atr_like)) if atr_like is not None and np.isfinite(_last(atr_like)) else 0.0
    if side == "buy" and p >= e and (e - p) <= (a * mult):
        return +8
    if side == "sell" and p <= e and (p - e) <= (a * mult):
        return +8
    return 0

# Детектор затишья: узкие BB в серии баров
def detect_quiet(bb_upper, bb_middle, bb_lower, min_bars=QUIET_MIN_BARS, max_width=QUIET_BB_WIDTH):
    if any(x is None for x in (bb_upper, bb_middle, bb_lower)):
        return False, 0
    bu, bm, bl = _to_series(bb_upper).values, _to_series(bb_middle).values, _to_series(bb_lower).values
    if len(bu) < min_bars: return False, 0
    width = (bu - bl) / (bm + 1e-12)
    cnt = 0
    for v in width[::-1]:
        if not np.isfinite(v): break
        if v <= max_width: cnt += 1
        else: break
    return (cnt >= min_bars), cnt

# Время суток (UTC) — маленький байас
def hour_bias_utc(df: pd.DataFrame) -> float:
    if not hasattr(df.index, "tz") and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.isna().all(): return 0.0
        h = int(ts.iloc[-1].hour)
    else:
        try:
            # если index datetime-like:
            h = int(pd.to_datetime(df.index).tz_convert("UTC")[-1].hour)
        except Exception:
            return 0.0
    if 12 <= h <= 20:
        return +0.02  # активнее
    if 0 <= h <= 4:
        return -0.02  # тонко
    return 0.0

# =============================
# Анализаторы сигналов (база)
# =============================

def analyze_trend_signals(prices: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(prices) < 60:
        return None
    s1 = _trend_strength_segment(prices[-20:])
    s2 = _trend_strength_segment(prices[-50:])
    s3 = _trend_strength_segment(prices[-100:]) if len(prices) >= 100 else _trend_strength_segment(prices)
    align = (np.sign(s1) == np.sign(s2) == np.sign(s3))
    overall = (abs(s1) + abs(s2) + abs(s3)) / 3.0
    if overall < MIN_TREND_STRENGTH:
        return None
    if align and s1 > 0:
        return {"type": "strong_buy" if overall > 0.8 else "buy",
                "confidence": "high", "indicator": "Trend",
                "message": f"Up-trend strength {overall:.2f}",
                "strength": min(95, int(overall * 100))}
    if align and s1 < 0:
        return {"type": "strong_sell" if overall > 0.8 else "sell",
                "confidence": "high", "indicator": "Trend",
                "message": f"Down-trend strength {overall:.2f}",
                "strength": min(95, int(overall * 100))}
    return None

def analyze_ema_signals(ema_fast, ema_slow, prices) -> Dict[str, Any]:
    cur = float(_last(ema_fast) - _last(ema_slow))
    prv = float(_prev(ema_fast) - _prev(ema_slow))
    cross_up = prv <= 0 and cur > 0
    cross_dn = prv >= 0 and cur < 0
    if cross_up:
        return {"type": "buy", "confidence": "high", "indicator": "EMA",
                "message": f"EMA cross up (Δ={cur:.6f})", "strength": 70}
    if cross_dn:
        return {"type": "sell", "confidence": "high", "indicator": "EMA",
                "message": f"EMA cross down (Δ={cur:.6f})", "strength": 70}
    if cur > 0:
        return {"type": "buy", "confidence": "medium", "indicator": "EMA",
                "message": f"EMA bias up (Δ={cur:.6f})", "strength": 55}
    if cur < 0:
        return {"type": "sell", "confidence": "medium", "indicator": "EMA",
                "message": f"EMA bias down (Δ={cur:.6f})", "strength": 55}
    return {"type":"neutral","confidence":"low","indicator":"EMA","message":"flat","strength":0}

def analyze_macd_signals(macd_line, macd_signal, macd_hist) -> Dict[str, Any]:
    line = float(_last(macd_line))
    sig  = float(_last(macd_signal))
    prev_line = float(_prev(macd_line))
    prev_sig  = float(_prev(macd_signal))
    hist = float(_last(macd_hist)) if macd_hist is not None else (line - sig)
    cross_up = (prev_line <= prev_sig) and (line > sig)
    cross_dn = (prev_line >= prev_sig) and (line < sig)
    if cross_up:
        return {"type":"buy","confidence":"high","indicator":"MACD","message":f"MACD cross up (hist={hist:.6f})","strength":70}
    if cross_dn:
        return {"type":"sell","confidence":"high","indicator":"MACD","message":f"MACD cross down (hist={hist:.6f})","strength":70}
    if hist > 0:
        return {"type":"buy","confidence":"medium","indicator":"MACD","message":"MACD histogram > 0","strength":55}
    if hist < 0:
        return {"type":"sell","confidence":"medium","indicator":"MACD","message":"MACD histogram < 0","strength":55}
    return {"type":"neutral","confidence":"low","indicator":"MACD","message":"flat","strength":0}

def analyze_rsi_signals(rsi_values, prices, asset_profile: AssetProfile) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if rsi_values is None or len(rsi_values) < 5:
        return out
    ob, os = adaptive_rsi_levels(asset_profile)
    cur = _nan_or(_last_valid(rsi_values), np.nan)
    prv = _nan_or(_prev_valid(rsi_values, 1), np.nan)
    if np.isnan(cur):
        return out
    if cur >= ob:
        out.append({"type":"sell","confidence":"high","indicator":"RSI","message":f"RSI overbought {cur:.1f}>{ob:.1f}","strength":70})
    elif cur <= os:
        out.append({"type":"buy","confidence":"high","indicator":"RSI","message":f"RSI oversold {cur:.1f}<{os:.1f}","strength":70})
    else:
        if not np.isnan(prv):
            if prv < os <= cur:
                out.append({"type":"buy","confidence":"medium","indicator":"RSI","message":"RSI cross up from oversold","strength":60})
            if prv > ob >= cur:
                out.append({"type":"sell","confidence":"medium","indicator":"RSI","message":"RSI cross down from overbought","strength":60})
    return out

def analyze_bollinger_bands_signals(bb_upper, bb_middle, bb_lower, prices) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if any(x is None for x in (prices, bb_upper, bb_middle, bb_lower)):
        return out
    p = float(_last(prices)); up = float(_last(bb_upper)); md = float(_last(bb_middle)); lo = float(_last(bb_lower))
    if not all(np.isfinite(v) for v in (p, up, md, lo)):
        return out
    up_dist = (p - up) / max(abs(up), 1e-12)
    lo_dist = (lo - p) / max(abs(lo), 1e-12)
    if up_dist > BB_THRESHOLD:
        out.append({"type":"sell","confidence":"medium","indicator":"Bollinger","message":f"Close above upper band (+{up_dist*100:.1f}%)","strength":65})
    if lo_dist > BB_THRESHOLD:
        out.append({"type":"buy","confidence":"medium","indicator":"Bollinger","message":f"Close below lower band (+{lo_dist*100:.1f}%)","strength":65})
    if p > md:
        out.append({"type":"buy","confidence":"low","indicator":"Bollinger","message":"Above middle band","strength":50})
    elif p < md:
        out.append({"type":"sell","confidence":"low","indicator":"Bollinger","message":"Below middle band","strength":50})
    return out

def analyze_volume_signals(prices, volumes, lookback: int = 20) -> Optional[Dict[str, Any]]:
    if volumes is None or len(volumes) < lookback or len(prices) < 2:
        return None
    v = _to_series(volumes).astype(float).values[-lookback:]
    mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
    if sd < 1e-12:
        return None
    z = (volumes[-1] - mu) / sd
    prc = 0.0
    if abs(prices[-2]) > 1e-12:
        prc = (prices[-1] - prices[-2]) / prices[-2] * 100.0
    if z > VOL_SPIKE_Z and prc > 1.0:
        recent_high = np.nanmax(prices[-lookback // 2:])
        is_brk = prices[-1] > recent_high
        return {"type":"strong_buy" if is_brk else "buy","confidence":"high","indicator":"Volume",
                "message":f"Volume spike {z:.1f}σ with price +{prc:.1f}%"+(" (breakout)" if is_brk else ""),
                "strength":85 if is_brk else 75}
    if z > VOL_SPIKE_Z and prc < -1.0:
        recent_low = np.nanmin(prices[-lookback // 2:])
        is_bd = prices[-1] < recent_low
        return {"type":"strong_sell" if is_bd else "sell","confidence":"high","indicator":"Volume",
                "message":f"Volume spike {z:.1f}σ with price {prc:.1f}%","strength":85 if is_bd else 75}
    return None

# =============================
# MTF SNAPSHOT (асинхронно)
# =============================

async def _fetch_htf_snapshot(symbol: Optional[str], default_exchange: str, tf: str) -> Dict[str, Any]:
    """
    Возвращает {'regime': MarketRegime|None, 'supertrend_dir': -1/1/None}
    Использует backend.exchange_data.get_candles если доступен.
    """
    if not (MTF_ON and symbol):
        return {}
    try:
        # импорт только тут, чтобы не плодить зависимостей при оффлайн-тестах
        from backend.exchange_data import get_candles
    except Exception:
        return {}

    try:
        df = await get_candles(default_exchange, symbol, tf, 200)
        if df is None or len(df) < 60:
            return {}
        df = df.rename(columns=str.lower).dropna(subset=["open","high","low","close"])
        prices = df["close"].to_numpy()
        regime = determine_market_regime(prices)
        ti = TechnicalIndicators(df[["open","high","low","close"]])
        st = await asyncio.to_thread(ti.supertrend, 10, MTF_ST_MULT)
        st_dir = None
        if isinstance(st, dict) and "direction" in st:
            st_dir = st["direction"][-1]
        elif isinstance(st, (list, tuple)) and len(st) >= 2:
            st_dir = st[1][-1]
        st_dir = int(st_dir) if st_dir in (-1, 1) else None
        return {"regime": regime, "supertrend_dir": st_dir}
    except Exception as e:
        logger.debug("MTF snapshot failed: %r", e)
        return {}

# =============================
# Основной API: generate_trading_signals
# =============================

async def generate_trading_signals(market_data: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Считает индикаторы и собирает усиленный ансамбль сигналов:
    - безопасная нормализация данных;
    - параллельный расчёт индикаторов (asyncio.to_thread);
    - EMA/MACD через ADX-гейт, RSI с адаптивными уровнями, BB/Keltner/Donchian/Supertrend/Trend/Volume;
    - CHOP-гейт, squeeze-выход по факту пробоя KC, S/R-дистанции, wick-антишип, pullback-бонус;
    - MTF подтверждение (опционально);
    - аккуратная агрегация, tiebreak по фильтрам (VWAP/SR/MTF), diversity-понимание strong, cooldown;
    - локальный кэш для ускорения горячих вызовов.
    """
    # --- ранние проверки ---
    if market_data is None or len(market_data) < MIN_DATA_POINTS:
        return {
            "signals": [],
            "overall_signal": "neutral",
            "confidence": "low",
            "confidence_score": 0.0,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
        }

    # --- кэш ---
    try:
        idx_tail = int(len(market_data))
        key = _cache.key("signals", symbol or "", idx_tail, list(map(str, market_data.columns)))
        cached = _cache.get(key, CACHE_TTL)
        if cached is not None:
            return cached
    except Exception:
        key = None

    # --- нормализация df ---
    df = market_data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(df) < MIN_DATA_POINTS:
        return {
            "signals": [],
            "overall_signal": "neutral",
            "confidence": "low",
            "confidence_score": 0.0,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
        }

    prices = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    volumes= df["volume"].to_numpy() if "volume" in df.columns else None

    # --- режим/веса/профиль ---
    regime  = determine_market_regime(prices, volumes)
    weights = get_regime_weights(regime)
    profile = calculate_asset_profile(symbol=symbol, highs=highs, lows=lows, prices=prices, volumes=volumes)
    st_mult = adaptive_supertrend_multiplier(profile)

    # --- фильтры ---
    adx_check = analyze_adx_filter(df["high"], df["low"], df["close"], min_adx=MIN_ADX_TREND)
    vwap_bias = analyze_vwap_filter(df)

    # --- индикаторы (параллельно) ---
    ti = TechnicalIndicators(df[["open", "high", "low", "close", *([] if "volume" not in df.columns else ["volume"])]] )
    ema9, ema21, macd, rsi, bb, st, chop_res, sqz_res = await asyncio.gather(
        asyncio.to_thread(ti.ema, 9),
        asyncio.to_thread(ti.ema, 21),
        asyncio.to_thread(ti.macd),
        asyncio.to_thread(ti.rsi, 14),
        asyncio.to_thread(ti.bollinger_bands, 20, 2.0),
        asyncio.to_thread(ti.supertrend, 10, st_mult),
        asyncio.to_thread(choppiness_index, df["high"], df["low"], df["close"], 14),
        asyncio.to_thread(boll_keltner_squeeze, df["high"], df["low"], df["close"]),
    )

    # --- MTF (асинхронный снимок) ---
    mtf_snapshot_task = asyncio.create_task(_fetch_htf_snapshot(symbol, MTF_EXCHANGE, MTF_TF)) if MTF_ON else None

    signals: List[Dict[str, Any]] = []

    # --- EMA (гейт по ADX) ---
    if ema9 is not None and ema21 is not None:
        s = analyze_ema_signals(ema9, ema21, prices)
        if s and s.get("type") != "neutral" and adx_check.get("ok", False):
            s["weight"] = weights.ema
            signals.append(s)

    # --- MACD (гейт по ADX) ---
    if macd is not None:
        if isinstance(macd, dict) and "macd" in macd and "signal" in macd:
            macd_line, macd_sig, macd_hist = macd["macd"], macd["signal"], macd.get("hist")
        elif isinstance(macd, (tuple, list)) and len(macd) >= 2:
            macd_line, macd_sig = macd[0], macd[1]
            macd_hist = macd[2] if len(macd) > 2 else None
        else:
            macd_line = macd_sig = macd_hist = None
        if macd_line is not None and macd_sig is not None:
            s = analyze_macd_signals(macd_line, macd_sig, macd_hist)
            if s and s.get("type") != "neutral" and adx_check.get("ok", False):
                s["weight"] = weights.macd
                signals.append(s)

    # --- RSI (адаптивные уровни, мягкий антиконтртренд) ---
    if rsi is not None:
        rsi_sigs = analyze_rsi_signals(rsi, prices, profile)
        is_strong_down = (regime == MarketRegime.TREND_BEARISH and getattr(profile, "trend_strength", 0.5) > 0.6)
        is_strong_up   = (regime == MarketRegime.TREND_BULLISH and getattr(profile, "trend_strength", 0.5) > 0.6)
        if STRICT:
            for rs in rsi_sigs:
                if is_strong_down and rs["type"] in ("buy", "strong_buy"):
                    rs["strength"] = max(50, rs.get("strength", 60) - 10)
                    if rs["confidence"] != "high":
                        rs["confidence"] = "low"
                if is_strong_up and rs["type"] in ("sell", "strong_sell"):
                    rs["strength"] = max(50, rs.get("strength", 60) - 10)
                    if rs["confidence"] != "high":
                        rs["confidence"] = "low"
        for rs in rsi_sigs:
            rs["weight"] = weights.rsi
            signals.append(rs)

    # --- Bollinger ---
    bb_up = bb_md = bb_lo = None
    if bb is not None:
        if isinstance(bb, dict) and all(k in bb for k in ("upper", "middle", "lower")):
            bb_up, bb_md, bb_lo = bb["upper"], bb["middle"], bb["lower"]
        elif isinstance(bb, (tuple, list)) and len(bb) >= 3:
            bb_up, bb_md, bb_lo = bb[0], bb[1], bb[2]
        if bb_up is not None and bb_md is not None and bb_lo is not None:
            for s in analyze_bollinger_bands_signals(bb_up, bb_md, bb_lo, prices):
                s["weight"] = weights.bb
                signals.append(s)

    # --- Supertrend (флип) ---
    if st is not None:
        if isinstance(st, dict) and all(k in st for k in ("supertrend", "direction")):
            st_line, st_dir = st["supertrend"], st["direction"]
        elif isinstance(st, (tuple, list)) and len(st) >= 2:
            st_line, st_dir = st[0], st[1]
        else:
            st_line = st_dir = None
        if st_dir is not None and len(st_dir) >= 2:
            if _prev(st_dir) == -1 and _last(st_dir) == 1:
                signals.append({"type": "buy", "confidence": "high", "indicator": "Supertrend",
                                "message": f"Supertrend flip up (mult={st_mult:.2f})",
                                "strength": 80, "weight": weights.supertrend})
            if _prev(st_dir) == 1 and _last(st_dir) == -1:
                signals.append({"type": "sell", "confidence": "high", "indicator": "Supertrend",
                                "message": f"Supertrend flip down (mult={st_mult:.2f})",
                                "strength": 80, "weight": weights.supertrend})

    # --- Keltner ---
    kc_up, kc_md, kc_lo = keltner_channels(df["high"], df["low"], df["close"], 20, 1.5)
    for s in analyze_keltner_signals(df["high"], df["low"], df["close"], 20, 1.5):
        s["weight"] = weights.keltner
        signals.append(s)

    # --- Donchian ---
    ds = analyze_donchian_breakout(df["high"], df["low"], df["close"], 20)
    if ds:
        ds["weight"] = weights.donchian
        signals.append(ds)

    # --- Volume spike ---
    vs = analyze_volume_signals(prices, volumes) if volumes is not None else None
    if vs:
        vs["weight"] = weights.vol
        signals.append(vs)

    # --- Trend strength ---
    ts = analyze_trend_signals(prices)
    if ts:
        ts["weight"] = weights.trend
        signals.append(ts)

    # =========================
    # Контекстные корректировки
    # =========================
    chop = float(chop_res) if chop_res is not None else np.nan
    in_sqz, bbw = sqz_res if isinstance(sqz_res, tuple) else (False, np.nan)

    # CHOP-гейт
    if np.isfinite(chop):
        for s in signals:
            if s["indicator"] in ("EMA", "MACD", "Supertrend", "Donchian", "Trend"):
                if chop > CHOP_HIGH:
                    s["strength"] = max(40, s.get("strength", 60) - 10)
                    if s["confidence"] != "high":
                        s["confidence"] = "low"
                elif chop < CHOP_LOW:
                    s["strength"] = min(95, s.get("strength", 60) + 5)

    # Squeeze: бонус при ФАКТЕ выхода за KC
    if in_sqz:
        last_p = float(df["close"].iloc[-1])
        kc_u, kc_m, kc_l = kc_up[-1], kc_md[-1], kc_lo[-1]
        if np.isfinite(kc_u) and last_p > kc_u:
            for s in signals:
                if s["indicator"] in ("Donchian", "Keltner", "Supertrend", "Trend"):
                    s["strength"] = min(95, s.get("strength", 60) + 8)
        if np.isfinite(kc_l) and last_p < kc_l:
            for s in signals:
                if s["indicator"] in ("Donchian", "Keltner", "Supertrend", "Trend"):
                    s["strength"] = min(95, s.get("strength", 60) + 8)

    # S/R дистанции (микро-«пространство»)
    up_lvl, lo_lvl, p = nearest_sr(df["close"], df["high"], df["low"])
    if np.isfinite(p):
        for s in signals:
            if s["type"].startswith("buy") and np.isfinite(up_lvl):
                dist = (up_lvl - p) / max(p, 1e-12)
                if dist > SR_FAR:
                    s["strength"] += 5
                elif dist < SR_NEAR:
                    s["strength"] = max(45, s["strength"] - 7)
            if s["type"].startswith("sell") and np.isfinite(lo_lvl):
                dist = (p - lo_lvl) / max(p, 1e-12)
                if dist > SR_FAR:
                    s["strength"] += 5
                elif dist < SR_NEAR:
                    s["strength"] = max(45, s["strength"] - 7)

    # Wick-антишип
    wicky_now = is_wicky_bar(df)
    if wicky_now:
        for s in signals:
            if s["indicator"] in ("RSI", "Bollinger", "Keltner"):
                s["strength"] = max(40, s.get("strength", 60) - 10)
                if s["confidence"] != "high":
                    s["confidence"] = "low"

    # Pullback-бонус (atr_like из Keltner)
    atr_like = (kc_up - kc_lo) / 2.0 if kc_up is not None and kc_lo is not None else None
    for s in signals:
        if s["indicator"] in ("EMA", "Supertrend", "Donchian", "Trend"):
            if s["type"].startswith("buy"):
                s["strength"] += trend_pullback_bonus(df["close"], ema21, atr_like, "buy", mult=1.0)
            if s["type"].startswith("sell"):
                s["strength"] += trend_pullback_bonus(df["close"], ema21, atr_like, "sell", mult=1.0)

    # Детектор затишья → бонус «бей в выход»
    quiet, qbars = detect_quiet(bb_up, bb_md, bb_lo, QUIET_MIN_BARS, QUIET_BB_WIDTH)
    if quiet:
        for s in signals:
            if s["indicator"] in ("Donchian", "Keltner", "Supertrend") and s["confidence"] in ("medium", "high"):
                s["strength"] = min(95, s.get("strength", 60) + 7)

    # VWAP bias — мягкая корректировка
    if vwap_bias:
        if vwap_bias["bias"] == "up":
            for s in signals:
                if s["type"].startswith("buy"):
                    s["strength"] = s.get("strength", 60) + 5
                elif s["type"].startswith("sell"):
                    s["strength"] = max(40, s.get("strength", 60) - 5)
        elif vwap_bias["bias"] == "down":
            for s in signals:
                if s["type"].startswith("sell"):
                    s["strength"] = s.get("strength", 60) + 5
                elif s["type"].startswith("buy"):
                    s["strength"] = max(40, s.get("strength", 60) - 5)

    # --- дожидаемся MTF и применяем подтверждение ---
    mtf_ctx = await mtf_snapshot_task if mtf_snapshot_task is not None else {}
    if mtf_ctx:
        htf_reg = mtf_ctx.get("regime")
        htf_st  = mtf_ctx.get("supertrend_dir")
        for s in signals:
            if htf_reg == MarketRegime.TREND_BULLISH and s["type"].startswith("buy"):
                s["strength"] = min(95, s.get("strength", 60) + 6)
            if htf_reg == MarketRegime.TREND_BEARISH and s["type"].startswith("sell"):
                s["strength"] = min(95, s.get("strength", 60) + 6)
            if htf_st == 1 and s["type"].startswith("buy"):
                s["strength"] += 4
            if htf_st == -1 and s["type"].startswith("sell"):
                s["strength"] += 4

    # =========================
    # Агрегирование и tiebreak
    # =========================
    buy = sell = tot_w = 0.0
    non_neutral: List[Dict[str, Any]] = []
    for s in signals:
        w = float(s.get("weight", 1.0))
        stg = float(s.get("strength", 60))
        if s["type"] in ("buy", "strong_buy"):
            buy += stg * w
            non_neutral.append(s)
        elif s["type"] in ("sell", "strong_sell"):
            sell += stg * w
            non_neutral.append(s)
        tot_w += w
    if tot_w > 0:
        buy /= tot_w
        sell /= tot_w

    # шум/ликвидность penalty
    noise_penalty = getattr(profile, "noise_level", 0.5)
    liq = getattr(profile, "liquidity_score", 0.5)
    penalty = max(0.0, (noise_penalty - 0.5) * 0.4) + max(0.0, (0.6 - liq) * 0.4)

    # tiebreak по фильтрам, если buy и sell близко
    net_raw = buy - sell
    side_margin = 6.0  # окно неопределённости в «силах» (5–8 ок)
    if abs(net_raw) < side_margin:
        tilt = 0
        if vwap_bias:
            tilt += (-2 if vwap_bias.get("bias") == "down" else 2)
        if np.isfinite(p):
            if np.isfinite(lo_lvl) and p < lo_lvl:
                tilt -= 2
            if np.isfinite(up_lvl) and p > up_lvl:
                tilt += 2
        if mtf_ctx:
            st_dir = mtf_ctx.get("supertrend_dir")
            if st_dir == 1:
                tilt += 1
            elif st_dir == -1:
                tilt -= 1
        net_raw += float(tilt) * 3.0  # мягкий рычаг (2.0–4.0 под настройку)

    # теперь penalty и уверенность
    net = net_raw * (1.0 - penalty)
    denom = max(1.0, buy + sell)
    conf_score = float(min(1.0, abs(net) / denom))

    # базовая классификация по net
    if net > 22:
        overall = "strong_buy"; conf = "very_high" if conf_score > 0.6 else "high"
    elif net > 10:
        overall = "buy";        conf = "high" if conf_score > 0.5 else "medium"
    elif net < -22:
        overall = "strong_sell"; conf = "very_high" if conf_score > 0.6 else "high"
    elif net < -10:
        overall = "sell";        conf = "high" if conf_score > 0.5 else "medium"
    else:
        overall = "neutral";     conf = "low"

    # требования к strong + diversity-понимание
    unique_inds = len(set(s.get("indicator", "?") for s in non_neutral))
    n_signals = len(non_neutral)

    def _diversity_multiplier(n: int, tau: float = 3.0) -> float:
        return float(1.0 - np.exp(-n / tau))  # 1→0.28, 2→0.49, 3→0.63, 6→0.86, 10→0.97

    if overall in ("strong_buy", "strong_sell"):
        if (n_signals < MIN_SIGNALS_FOR_STRONG) or (unique_inds < MIN_UNIQUE_INDICATORS_FOR_STRONG):
            overall = "buy" if overall == "strong_buy" else "sell"
            if conf == "very_high":
                conf = "high"

    trend_like = any(
        s.get("indicator") in ("Supertrend", "Donchian", "Trend", "Volume") and s.get("strength", 0) >= 75
        for s in non_neutral
    )
    if overall in ("strong_buy", "strong_sell") and not trend_like:
        overall = "buy" if overall == "strong_buy" else "sell"
        if conf == "very_high":
            conf = "high"

    conf_score *= _diversity_multiplier(n_signals, tau=3.0)

    # UTC-часовой байас
    if HOUR_BONUS_ACTIVE:
        conf_score = float(np.clip(conf_score + hour_bias_utc(df), 0.0, 1.0))

    # шкала уверенности
    if conf_score < 0.35:
        conf = "low"
    elif conf_score < 0.55:
        conf = "medium"
    elif conf_score < 0.75:
        conf = "high"
    else:
        conf = "very_high" if overall in ("buy", "sell", "strong_buy", "strong_sell") else "high"

    # STRICT: минимальная уверенность
    if STRICT and overall != "neutral" and conf_score < MIN_CONF_SCORE:
        overall = "neutral"; conf = "low"

    # cooldown от flip-flop сильных
    try:
        idx_tail = int(len(df))
        key_sym = symbol or "__"
        last_type, last_at = _last_strong.get(key_sym, (None, -10**9))
        if overall in ("strong_buy", "strong_sell") and (last_type and last_type != overall):
            if (idx_tail - last_at) <= COOLDOWN_BARS:
                overall = "neutral"; conf = "low"
        if overall in ("strong_buy", "strong_sell"):
            _last_strong[key_sym] = (overall, idx_tail)
    except Exception:
        pass

    # --- результат ---
    result = {
        "signals": signals,
        "overall_signal": overall,
        "confidence": conf,
        "confidence_score": conf_score,
        "market_regime": regime.value,
        "asset_profile": profile.to_dict() if hasattr(profile, "to_dict") else {},
        "filters": {
            "adx": adx_check,
            "vwap_bias": vwap_bias,
            "chop": {"value": chop},
            "squeeze": {"in_squeeze": bool(in_sqz), "bb_width": bbw},
            "sr": {
                "nearest_resistance": float(up_lvl) if np.isfinite(up_lvl) else None,
                "nearest_support": float(lo_lvl)   if np.isfinite(lo_lvl) else None,
                "price": float(p) if np.isfinite(p) else None,
            },
            "wicky": bool(wicky_now),
            "quiet": {"active": bool(quiet), "bars": int(qbars)},
            "mtf": {
                "on": bool(MTF_ON), "exchange": MTF_EXCHANGE, "tf": MTF_TF,
                "regime": mtf_ctx.get("regime").value if mtf_ctx and mtf_ctx.get("regime") else None,
                "supertrend_dir": mtf_ctx.get("supertrend_dir") if mtf_ctx else None,
            },
        },
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
    }

    if key:
        _cache.set(key, result)
        logger.debug("[signals] %s regime=%s non_neutral=%d conf=%.3f overall=%s",
                     symbol, regime.value, len(non_neutral), conf_score, result["overall_signal"])
    return result


# =============================
# SL/TP (адаптивные)
# =============================

async def generate_adaptive_stop_loss_take_profit(
    market_data: pd.DataFrame,
    entry_price: float,
    position_type: str,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Адаптивные SL/TP от профиля актива (ATR%, шум, тренд)."""
    if market_data is None or len(market_data) < MIN_DATA_POINTS:
        return {"stop_loss": None, "take_profit": None, "risk_reward_ratio": None,
                "timestamp": datetime.now().isoformat(), "symbol": symbol}

    df = market_data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for col in ("open","high","low","close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)

    prices = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    volumes= df["volume"].to_numpy() if "volume" in df.columns else None

    profile = calculate_asset_profile(symbol=symbol, highs=highs, lows=lows, prices=prices, volumes=volumes)
    sl, tp, rr = adaptive_stop_loss_take_profit(profile, float(entry_price), str(position_type))

    return {"stop_loss": float(sl), "take_profit": float(tp), "risk_reward_ratio": float(rr),
            "asset_profile": profile.to_dict() if hasattr(profile, "to_dict") else {},
            "timestamp": datetime.now().isoformat(), "symbol": symbol}
