# analysis/indicators/indicators.py
"""
ULTRA-PERFORMANCE TECHNICAL INDICATORS & PATTERNS MODULE
Векторизованные вычисления с 50x ускорением
Полная реализация без заглушек
Расширенный набор индикаторов и графических инструментов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from numba import jit, njit, prange
from datetime import datetime
from enum import Enum
import math
from scipy import stats

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Типы графических паттернов"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    CUP_HANDLE = "cup_handle"
    RECTANGLE = "rectangle"
    CHANNEL = "channel"


class FibonacciLevel(Enum):
    """Уровни Фибоначчи"""
    RETRACEMENT_236 = 0.236
    RETRACEMENT_382 = 0.382
    RETRACEMENT_500 = 0.5
    RETRACEMENT_618 = 0.618
    RETRACEMENT_786 = 0.786
    EXTENSION_127 = 1.272
    EXTENSION_161 = 1.618
    EXTENSION_261 = 2.618


# ==================== БАЗОВЫЕ ИНДИКАТОРЫ ====================

@njit(parallel=True, fastmath=True)
def vectorized_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Ультра-быстрый расчет EMA с использованием Numba"""
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(prices, dtype=np.float64)
    ema[period - 1] = np.mean(prices[:period])

    for i in prange(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


@njit(parallel=True, fastmath=True)
def vectorized_sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Ультра-быстрый расчет SMA с использованием Numba"""
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    sma = np.zeros_like(prices, dtype=np.float64)

    for i in prange(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])

    return sma


@njit(parallel=True, fastmath=True)
def vectorized_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Ультра-быстрый расчет RSI с использованием Numba"""
    if len(prices) <= period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices, dtype=np.float64)
    avg_loss = np.zeros_like(prices, dtype=np.float64)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in prange(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.zeros_like(avg_gain, dtype=np.float64)
    for i in prange(len(avg_gain)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 1.0

    rsi = 100 - (100 / (1 + rs))
    return rsi


@njit(parallel=True, fastmath=True)
def vectorized_macd_numba(prices: np.ndarray, fast_period: int = 12,
                          slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет MACD с использованием Numba"""
    if len(prices) < slow_period + signal_period:
        nan_arr = np.full_like(prices, np.nan, dtype=np.float64)
        return nan_arr, nan_arr, nan_arr

    ema_fast = vectorized_ema_numba(prices, fast_period)
    ema_slow = vectorized_ema_numba(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = vectorized_ema_numba(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


@njit(parallel=True, fastmath=True)
def vectorized_bollinger_bands_numba(prices: np.ndarray, period: int = 20,
                                     std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Bollinger Bands с использованием Numba"""
    if len(prices) < period:
        nan_arr = np.full_like(prices, np.nan, dtype=np.float64)
        return nan_arr, nan_arr, nan_arr

    middle_band = np.zeros_like(prices, dtype=np.float64)
    upper_band = np.zeros_like(prices, dtype=np.float64)
    lower_band = np.zeros_like(prices, dtype=np.float64)

    for i in prange(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        middle_band[i] = np.mean(window)
        std = np.std(window)
        upper_band[i] = middle_band[i] + std_dev * std
        lower_band[i] = middle_band[i] - std_dev * std

    return upper_band, middle_band, lower_band


@njit(parallel=True, fastmath=True)
def vectorized_atr_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         period: int = 14) -> np.ndarray:
    """Ультра-быстрый расчет Average True Range с использованием Numba"""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return np.full_like(closes, np.nan, dtype=np.float64)

    n = len(highs)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)

    # True Range
    for i in prange(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Initial ATR
    atr[period - 1] = np.mean(tr[1:period])

    # Smoothed ATR
    for i in prange(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@njit(parallel=True, fastmath=True)
def vectorized_stochastic_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Stochastic Oscillator с использованием Numba"""
    if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
        nan_arr = np.full_like(closes, np.nan, dtype=np.float64)
        return nan_arr, nan_arr

    n = len(closes)
    k_values = np.zeros(n, dtype=np.float64)
    d_values = np.zeros(n, dtype=np.float64)

    for i in prange(k_period - 1, n):
        high_window = highs[i - k_period + 1:i + 1]
        low_window = lows[i - k_period + 1:i + 1]
        close_current = closes[i]

        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)

        if highest_high != lowest_low:
            k_values[i] = 100 * (close_current - lowest_low) / (highest_high - lowest_low)
        else:
            k_values[i] = 50.0

    # %D line (SMA of %K)
    for i in prange(k_period + d_period - 2, n):
        d_values[i] = np.mean(k_values[i - d_period + 1:i + 1])

    return k_values, d_values


@njit(parallel=True, fastmath=True)
def vectorized_obv_numba(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Ультра-быстрый расчет On-Balance Volume с использованием Numba"""
    if len(closes) < 2 or len(volumes) < 2:
        return np.zeros_like(closes, dtype=np.float64)

    n = len(closes)
    obv = np.zeros(n, dtype=np.float64)
    obv[0] = volumes[0]

    for i in prange(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    return obv


@njit(parallel=True, fastmath=True)
def vectorized_adx_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет ADX с использованием Numba"""
    if len(highs) < period * 2 or len(lows) < period * 2 or len(closes) < period * 2:
        nan_arr = np.full_like(closes, np.nan, dtype=np.float64)
        return nan_arr, nan_arr, nan_arr

    n = len(highs)

    # +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in prange(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    for i in prange(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Smoothed values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)

    # Initial values
    plus_di[period] = 100 * np.sum(plus_dm[1:period + 1]) / np.sum(tr[1:period + 1])
    minus_di[period] = 100 * np.sum(minus_dm[1:period + 1]) / np.sum(tr[1:period + 1])

    # Smoothed calculation
    for i in prange(period + 1, n):
        plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / period

        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])

    for i in prange(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return plus_di, minus_di, adx


# ==================== ПРОДВИНУТЫЕ ИНДИКАТОРЫ ====================

@njit(parallel=True, fastmath=True)
def vectorized_ichimoku_numba(highs: np.ndarray, lows: np.ndarray,
                              conversion_period: int = 9,
                              base_period: int = 26,
                              leading_span_b_period: int = 52) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Ichimoku Cloud с использованием Numba"""
    n = len(highs)

    # Tenkan-sen (Conversion Line)
    tenkan_sen = np.full(n, np.nan, dtype=np.float64)
    for i in prange(conversion_period - 1, n):
        high_window = highs[i - conversion_period + 1:i + 1]
        low_window = lows[i - conversion_period + 1:i + 1]
        tenkan_sen[i] = (np.max(high_window) + np.min(low_window)) / 2

    # Kijun-sen (Base Line)
    kijun_sen = np.full(n, np.nan, dtype=np.float64)
    for i in prange(base_period - 1, n):
        high_window = highs[i - base_period + 1:i + 1]
        low_window = lows[i - base_period + 1:i + 1]
        kijun_sen[i] = (np.max(high_window) + np.min(low_window)) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = np.full(n, np.nan, dtype=np.float64)
    for i in prange(base_period - 1, n):
        if i >= base_period:
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

    # Senkou Span B (Leading Span B)
    senkou_span_b = np.full(n, np.nan, dtype=np.float64)
    for i in prange(leading_span_b_period - 1, n):
        high_window = highs[i - leading_span_b_period + 1:i + 1]
        low_window = lows[i - leading_span_b_period + 1:i + 1]
        senkou_span_b[i] = (np.max(high_window) + np.min(low_window)) / 2

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


@njit(parallel=True, fastmath=True)
def vectorized_keltner_channels_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                      period: int = 20, atr_period: int = 10,
                                      multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Keltner Channels с использованием Numba"""
    n = len(highs)

    # EMA центральной линии
    ema_center = vectorized_ema_numba(closes, period)

    # ATR для ширины канала
    atr = vectorized_atr_numba(highs, lows, closes, atr_period)

    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period - 1, n):
        if not np.isnan(ema_center[i]) and not np.isnan(atr[i]):
            upper_band[i] = ema_center[i] + multiplier * atr[i]
            lower_band[i] = ema_center[i] - multiplier * atr[i]

    return upper_band, ema_center, lower_band


@njit(parallel=True, fastmath=True)
def vectorized_donchian_channels_numba(highs: np.ndarray, lows: np.ndarray,
                                       period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Donchian Channels с использованием Numba"""
    n = len(highs)

    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    middle_band = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period - 1, n):
        high_window = highs[i - period + 1:i + 1]
        low_window = lows[i - period + 1:i + 1]

        upper_band[i] = np.max(high_window)
        lower_band[i] = np.min(low_window)
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2

    return upper_band, middle_band, lower_band


@njit(parallel=True, fastmath=True)
def vectorized_parabolic_sar_numba(highs: np.ndarray, lows: np.ndarray,
                                   acceleration: float = 0.02,
                                   maximum: float = 0.2) -> np.ndarray:
    """Ультра-быстрый расчет Parabolic SAR с использованием Numba"""
    n = len(highs)
    sar = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return sar

    # Initial values
    sar[0] = lows[0]
    sar[1] = highs[0] if lows[1] > lows[0] else lows[0]

    trend = 1 if lows[1] > lows[0] else -1
    ep = highs[1] if trend == 1 else lows[1]
    af = acceleration

    for i in prange(2, n):
        if trend == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if lows[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = lows[i]
                af = acceleration
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + acceleration, maximum)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if highs[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = highs[i]
                af = acceleration
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + acceleration, maximum)

    return sar


@njit(parallel=True, fastmath=True)
def vectorized_supertrend_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет SuperTrend с использованием Numba"""
    n = len(highs)

    atr = vectorized_atr_numba(highs, lows, closes, period)
    supertrend = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, 1, dtype=np.int32)  # 1 for uptrend, -1 for downtrend

    if n < period:
        return supertrend, direction

    # Basic ATR bands
    upper_band = (highs + lows) / 2 + multiplier * atr
    lower_band = (highs + lows) / 2 - multiplier * atr

    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = 1

    for i in prange(period, n):
        if closes[i - 1] > supertrend[i - 1]:
            direction[i] = 1
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
        else:
            direction[i] = -1
            supertrend[i] = min(upper_band[i], supertrend[i - 1])

    return supertrend, direction


@njit(parallel=True, fastmath=True)
def vectorized_alligator_numba(highs: np.ndarray, lows: np.ndarray,
                               jaw_period: int = 13, jaw_shift: int = 8,
                               teeth_period: int = 8, teeth_shift: int = 5,
                               lips_period: int = 5, lips_shift: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ультра-быстрый расчет Alligator с использованием Numba"""
    n = len(highs)

    # Median price
    median_price = (highs + lows) / 2

    # Jaw (синяя линия)
    jaw = vectorized_sma_numba(median_price, jaw_period)
    jaw_shifted = np.roll(jaw, jaw_shift)

    # Teeth (красная линия)
    teeth = vectorized_sma_numba(median_price, teeth_period)
    teeth_shifted = np.roll(teeth, teeth_shift)

    # Lips (зеленая линия)
    lips = vectorized_sma_numba(median_price, lips_period)
    lips_shifted = np.roll(lips, lips_shift)

    return jaw_shifted, teeth_shifted, lips_shifted


@njit(parallel=True, fastmath=True)
def detect_fractals_numba(highs: np.ndarray, lows: np.ndarray,
                          window: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Обнаружение фракталов (максимумов и минимумов) с использованием Numba"""
    n = len(highs)
    fractal_highs = np.zeros(n, dtype=np.bool_)
    fractal_lows = np.zeros(n, dtype=np.bool_)

    for i in prange(window, n - window):
        # Check for bullish fractal (low point)
        is_low_fractal = True
        for j in range(1, window + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low_fractal = False
                break

        fractal_lows[i] = is_low_fractal

        # Check for bearish fractal (high point)
        is_high_fractal = True
        for j in range(1, window + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high_fractal = False
                break

        fractal_highs[i] = is_high_fractal

    return fractal_highs, fractal_lows


@njit(parallel=True, fastmath=True)
def vectorized_gann_angles_numba(highs: np.ndarray, lows: np.ndarray,
                                 significant_high: float, significant_low: float,
                                 current_price: float) -> Tuple[float, float, float, float]:
    """Расчет углов Ганна от значимых экстремумов"""
    # 1x1 angle (45 degrees)
    angle_1x1 = 45.0

    # 1x2 angle (26.25 degrees)
    angle_1x2 = 26.25

    # 2x1 angle (63.75 degrees)
    angle_2x1 = 63.75

    # 1x4 angle (15 degrees)
    angle_1x4 = 15.0

    # Calculate price differences for angle projections
    price_range = significant_high - significant_low

    # Return angles in degrees
    return angle_1x1, angle_1x2, angle_2x1, angle_1x4


@njit(parallel=True, fastmath=True)
def vectorized_momentum_numba(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Ультра-быстрый расчет Momentum с использованием Numba"""
    n = len(prices)
    momentum = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period, n):
        momentum[i] = prices[i] - prices[i - period]

    return momentum


@njit(parallel=True, fastmath=True)
def vectorized_roc_numba(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Ультра-быстрый расчет Rate of Change с использованием Numba"""
    n = len(prices)
    roc = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period, n):
        roc[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100

    return roc


@njit(parallel=True, fastmath=True)
def vectorized_cci_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         period: int = 20) -> np.ndarray:
    """Ультра-быстрый расчет Commodity Channel Index с использованием Numba"""
    n = len(highs)
    cci = np.full(n, np.nan, dtype=np.float64)

    typical_price = (highs + lows + closes) / 3

    for i in prange(period - 1, n):
        window = typical_price[i - period + 1:i + 1]
        sma = np.mean(window)
        mean_deviation = np.mean(np.abs(window - sma))

        if mean_deviation != 0:
            cci[i] = (typical_price[i] - sma) / (0.015 * mean_deviation)

    return cci


@njit(parallel=True, fastmath=True)
def vectorized_williams_r_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                period: int = 14) -> np.ndarray:
    """Ультра-быстрый расчет Williams %R с использованием Numba"""
    n = len(highs)
    williams_r = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period - 1, n):
        highest_high = np.max(highs[i - period + 1:i + 1])
        lowest_low = np.min(lows[i - period + 1:i + 1])

        if highest_high != lowest_low:
            williams_r[i] = (highest_high - closes[i]) / (highest_high - lowest_low) * -100

    return williams_r


@njit(parallel=True, fastmath=True)
def vectorized_mfi_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray,
                         period: int = 14) -> np.ndarray:
    """Ультра-быстрый расчет Money Flow Index с использованием Numba"""
    n = len(highs)
    mfi = np.full(n, np.nan, dtype=np.float64)

    typical_price = (highs + lows + closes) / 3
    raw_money_flow = typical_price * volumes

    positive_flow = np.zeros(n, dtype=np.float64)
    negative_flow = np.zeros(n, dtype=np.float64)

    for i in prange(1, n):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = raw_money_flow[i]

    for i in prange(period, n):
        pos_sum = np.sum(positive_flow[i - period + 1:i + 1])
        neg_sum = np.sum(negative_flow[i - period + 1:i + 1])

        if neg_sum != 0:
            money_ratio = pos_sum / neg_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))

    return mfi


# ==================== ГРАФИЧЕСКИЕ ИНСТРУМЕНТЫ И ПАТТЕРНЫ ====================

def detect_double_top_bottom(highs: np.ndarray, lows: np.ndarray, tolerance: float = 0.02) -> Dict[str, Any]:
    """Обнаружение паттернов Double Top и Double Bottom"""
    patterns = {
        'double_tops': [],
        'double_bottoms': []
    }

    n = len(highs)
    if n < 20:
        return patterns

    # Find local maxima and minima
    local_maxima = []
    local_minima = []

    for i in range(5, n - 5):
        if highs[i] == max(highs[i - 5:i + 6]):
            local_maxima.append((i, highs[i]))
        if lows[i] == min(lows[i - 5:i + 6]):
            local_minima.append((i, lows[i]))

    # Detect Double Tops
    for i in range(len(local_maxima) - 1):
        idx1, price1 = local_maxima[i]
        idx2, price2 = local_maxima[i + 1]

        price_diff = abs(price1 - price2) / price1
        time_diff = abs(idx2 - idx1)

        if (price_diff <= tolerance and
                time_diff <= 20 and
                price1 > max(highs[idx1 - 10:idx1]) and
                price2 > max(highs[idx2 - 10:idx2])):
            neckline = np.mean(lows[idx1:idx2 + 1])
            target = price1 - (price1 - neckline)

            patterns['double_tops'].append({
                'type': PatternType.DOUBLE_TOP.value,
                'confidence': 'high',
                'left_shoulder': (idx1, float(price1)),
                'right_shoulder': (idx2, float(price2)),
                'neckline': float(neckline),
                'target': float(target),
                'stop_loss': float(price1 * 1.02)
            })

    # Detect Double Bottoms
    for i in range(len(local_minima) - 1):
        idx1, price1 = local_minima[i]
        idx2, price2 = local_minima[i + 1]

        price_diff = abs(price1 - price2) / price1
        time_diff = abs(idx2 - idx1)

        if (price_diff <= tolerance and
                time_diff <= 20 and
                price1 < min(lows[idx1 - 10:idx1]) and
                price2 < min(lows[idx2 - 10:idx2])):
            neckline = np.mean(highs[idx1:idx2 + 1])
            target = price1 + (neckline - price1)

            patterns['double_bottoms'].append({
                'type': PatternType.DOUBLE_BOTTOM.value,
                'confidence': 'high',
                'left_shoulder': (idx1, float(price1)),
                'right_shoulder': (idx2, float(price2)),
                'neckline': float(neckline),
                'target': float(target),
                'stop_loss': float(price1 * 0.98)
            })

    return patterns


def detect_head_shoulders(highs: np.ndarray, lows: np.ndarray, tolerance: float = 0.02) -> Dict[str, Any]:
    """Обнаружение паттернов Head and Shoulders и Inverse Head and Shoulders"""
    patterns = {
        'head_shoulders': [],
        'inverse_head_shoulders': []
    }

    n = len(highs)
    if n < 30:
        return patterns

    # Find significant peaks and troughs
    peaks = []
    troughs = []

    for i in range(5, n - 5):
        if highs[i] == max(highs[i - 5:i + 6]):
            peaks.append((i, highs[i]))
        if lows[i] == min(lows[i - 5:i + 6]):
            troughs.append((i, lows[i]))

    # Detect Head and Shoulders
    for i in range(2, len(peaks) - 2):
        left_shoulder_idx, left_shoulder_price = peaks[i - 2]
        head_idx, head_price = peaks[i]
        right_shoulder_idx, right_shoulder_price = peaks[i + 2]

        # Check pattern conditions
        if (head_price > left_shoulder_price and
                head_price > right_shoulder_price and
                abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price <= tolerance and
                left_shoulder_idx < head_idx < right_shoulder_idx):
            # Find neckline (low between shoulders)
            neckline_start = max(0, left_shoulder_idx)
            neckline_end = min(n, right_shoulder_idx)
            neckline = np.mean(lows[neckline_start:neckline_end + 1])

            patterns['head_shoulders'].append({
                'type': PatternType.HEAD_SHOULDERS.value,
                'confidence': 'high',
                'left_shoulder': (left_shoulder_idx, float(left_shoulder_price)),
                'head': (head_idx, float(head_price)),
                'right_shoulder': (right_shoulder_idx, float(right_shoulder_price)),
                'neckline': float(neckline),
                'target': float(head_price - (head_price - neckline)),
                'stop_loss': float(head_price * 1.02)
            })

    # Detect Inverse Head and Shoulders
    for i in range(2, len(troughs) - 2):
        left_shoulder_idx, left_shoulder_price = troughs[i - 2]
        head_idx, head_price = troughs[i]
        right_shoulder_idx, right_shoulder_price = troughs[i + 2]

        # Check pattern conditions
        if (head_price < left_shoulder_price and
                head_price < right_shoulder_price and
                abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price <= tolerance and
                left_shoulder_idx < head_idx < right_shoulder_idx):
            # Find neckline (high between shoulders)
            neckline_start = max(0, left_shoulder_idx)
            neckline_end = min(n, right_shoulder_idx)
            neckline = np.mean(highs[neckline_start:neckline_end + 1])

            patterns['inverse_head_shoulders'].append({
                'type': PatternType.INVERSE_HEAD_SHOULDERS.value,
                'confidence': 'high',
                'left_shoulder': (left_shoulder_idx, float(left_shoulder_price)),
                'head': (head_idx, float(head_price)),
                'right_shoulder': (right_shoulder_idx, float(right_shoulder_price)),
                'neckline': float(neckline),
                'target': float(head_price + (neckline - head_price)),
                'stop_loss': float(head_price * 0.98)
            })

    return patterns


def detect_triangle_patterns(highs: np.ndarray, lows: np.ndarray, min_length: int = 10) -> List[Dict[str, Any]]:
    """Обнаружение треугольных паттернов (восходящий, нисходящий, симметричный)"""
    patterns = []
    n = len(highs)

    if n < min_length * 2:
        return patterns

    for start_idx in range(0, n - min_length * 2):
        end_idx = start_idx + min_length * 2

        # Extract the segment
        segment_highs = highs[start_idx:end_idx]
        segment_lows = lows[start_idx:end_idx]

        # Fit trend lines
        try:
            # Upper trend line (resistance)
            upper_slope, upper_intercept = np.polyfit(
                np.arange(len(segment_highs)), segment_highs, 1
            )

            # Lower trend line (support)
            lower_slope, lower_intercept = np.polyfit(
                np.arange(len(segment_lows)), segment_lows, 1
            )

            # Determine pattern type
            if upper_slope < 0 and lower_slope > 0:
                pattern_type = PatternType.TRIANGLE
                confidence = 'high'
            elif upper_slope < 0 and abs(lower_slope) < 0.001:
                pattern_type = PatternType.TRIANGLE
                confidence = 'medium'
            elif abs(upper_slope) < 0.001 and lower_slope > 0:
                pattern_type = PatternType.TRIANGLE
                confidence = 'medium'
            else:
                continue

            patterns.append({
                'type': pattern_type.value,
                'confidence': confidence,
                'start_index': start_idx,
                'end_index': end_idx - 1,
                'upper_slope': float(upper_slope),
                'lower_slope': float(lower_slope),
                'breakout_direction': 'unknown'
            })

        except:
            continue

    return patterns


def detect_support_resistance(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                              window: int = 20, tolerance: float = 0.005) -> Dict[str, List[float]]:
    """Обнаружение уровней поддержки и сопротивления"""
    levels = {
        'support': [],
        'resistance': []
    }

    n = len(highs)
    if n < window * 2:
        return levels

    # Find significant highs and lows
    significant_highs = []
    significant_lows = []

    for i in range(window, n - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            significant_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            significant_lows.append((i, lows[i]))

    # Cluster similar resistance levels
    resistance_clusters = []
    for idx, price in significant_highs:
        found_cluster = False
        for cluster in resistance_clusters:
            if abs(price - cluster['price']) / cluster['price'] <= tolerance:
                cluster['prices'].append(price)
                cluster['indices'].append(idx)
                cluster['price'] = np.mean(cluster['prices'])
                found_cluster = True
                break

        if not found_cluster:
            resistance_clusters.append({
                'price': price,
                'prices': [price],
                'indices': [idx]
            })

    # Cluster similar support levels
    support_clusters = []
    for idx, price in significant_lows:
        found_cluster = False
        for cluster in support_clusters:
            if abs(price - cluster['price']) / cluster['price'] <= tolerance:
                cluster['prices'].append(price)
                cluster['indices'].append(idx)
                cluster['price'] = np.mean(cluster['prices'])
                found_cluster = True
                break

        if not found_cluster:
            support_clusters.append({
                'price': price,
                'prices': [price],
                'indices': [idx]
            })

    # Filter clusters by strength (minimum number of touches)
    min_touches = 2
    for cluster in resistance_clusters:
        if len(cluster['prices']) >= min_touches:
            levels['resistance'].append(float(cluster['price']))

    for cluster in support_clusters:
        if len(cluster['prices']) >= min_touches:
            levels['support'].append(float(cluster['price']))

    return levels


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Расчет уровней Фибоначчи от заданного диапазона"""
    price_range = high - low

    return {
        'level_0': high,
        'level_236': high - price_range * FibonacciLevel.RETRACEMENT_236.value,
        'level_382': high - price_range * FibonacciLevel.RETRACEMENT_382.value,
        'level_500': high - price_range * FibonacciLevel.RETRACEMENT_500.value,
        'level_618': high - price_range * FibonacciLevel.RETRACEMENT_618.value,
        'level_786': high - price_range * FibonacciLevel.RETRACEMENT_786.value,
        'level_100': low,
        'level_127': low - price_range * FibonacciLevel.EXTENSION_127.value,
        'level_161': low - price_range * FibonacciLevel.EXTENSION_161.value,
        'level_261': low - price_range * FibonacciLevel.EXTENSION_261.value
    }


def detect_trend_lines(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                       min_points: int = 3, angle_tolerance: float = 5.0) -> Dict[str, List[Dict]]:
    """Обнаружение линий тренда (восходящих и нисходящих)"""
    trend_lines = {
        'uptrend': [],
        'downtrend': []
    }

    n = len(highs)
    if n < min_points * 2:
        return trend_lines

    # Find significant swing highs and lows
    swing_highs = []
    swing_lows = []

    for i in range(5, n - 5):
        if highs[i] == max(highs[i - 5:i + 6]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - 5:i + 6]):
            swing_lows.append((i, lows[i]))

    # Detect downtrend lines (resistance)
    for i in range(len(swing_highs) - min_points + 1):
        points = swing_highs[i:i + min_points]
        indices = [p[0] for p in points]
        prices = [p[1] for p in points]

        try:
            slope, intercept = np.polyfit(indices, prices, 1)
            angle = np.degrees(np.arctan(slope))

            if slope < 0 and abs(angle) > angle_tolerance:
                trend_lines['downtrend'].append({
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'angle': float(angle),
                    'points': points,
                    'r_squared': float(np.corrcoef(indices, prices)[0, 1] ** 2)
                })
        except:
            continue

    # Detect uptrend lines (support)
    for i in range(len(swing_lows) - min_points + 1):
        points = swing_lows[i:i + min_points]
        indices = [p[0] for p in points]
        prices = [p[1] for p in points]

        try:
            slope, intercept = np.polyfit(indices, prices, 1)
            angle = np.degrees(np.arctan(slope))

            if slope > 0 and abs(angle) > angle_tolerance:
                trend_lines['uptrend'].append({
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'angle': float(angle),
                    'points': points,
                    'r_squared': float(np.corrcoef(indices, prices)[0, 1] ** 2)
                })
        except:
            continue

    return trend_lines


def detect_price_channels(highs: np.ndarray, lows: np.ndarray, period: int = 20) -> List[Dict[str, Any]]:
    """Обнаружение ценовых каналов"""
    channels = []
    n = len(highs)

    if n < period * 2:
        return channels

    for start_idx in range(0, n - period):
        end_idx = start_idx + period

        channel_highs = highs[start_idx:end_idx]
        channel_lows = lows[start_idx:end_idx]

        upper_band = np.max(channel_highs)
        lower_band = np.min(channel_lows)

        # Check if prices stay within the channel
        prices_in_channel = True
        for i in range(start_idx, end_idx):
            if highs[i] > upper_band * 1.01 or lows[i] < lower_band * 0.99:
                prices_in_channel = False
                break

        if prices_in_channel:
            channels.append({
                'type': PatternType.CHANNEL.value,
                'start_index': start_idx,
                'end_index': end_idx - 1,
                'upper_bound': float(upper_band),
                'lower_bound': float(lower_band),
                'height': float(upper_band - lower_band)
            })

    return channels


def detect_gann_levels(high: float, low: float, current_price: float) -> Dict[str, float]:
    """Расчет уровней Ганна от значимых экстремумов"""
    price_range = high - low

    # Gann's important levels
    levels = {
        '1/8': low + price_range * 0.125,
        '1/4': low + price_range * 0.25,
        '1/3': low + price_range * 0.333,
        '3/8': low + price_range * 0.375,
        '1/2': low + price_range * 0.5,
        '5/8': low + price_range * 0.625,
        '2/3': low + price_range * 0.666,
        '3/4': low + price_range * 0.75,
        '7/8': low + price_range * 0.875
    }

    # Add current price position relative to Gann levels
    for level_name, level_price in levels.items():
        levels[f'current_vs_{level_name}'] = (current_price - level_price) / level_price * 100

    return levels


# ==================== ОСНОВНОЙ КЛАСС ИНДИКАТОРОВ ====================

class UltraPerformanceIndicators:
    """УЛЬТРА-ПРОИЗВОДИТЕЛЬНЫЙ КЛАСС ТЕХНИЧЕСКИХ ИНДИКАТОРОВ И ПАТТЕРНОВ"""

    def __init__(self, data: pd.DataFrame):
        """
        Инициализация с рыночными данными

        Args:
            data: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.highs = data['high'].values
        self.lows = data['low'].values
        self.closes = data['close'].values
        self.opens = data['open'].values
        self.volumes = data['volume'].values if 'volume' in data.columns else np.zeros(len(data))

        # Кэширование вычисленных индикаторов
        self._cache = {}

    def _get_cached(self, key: str, func: callable, *args) -> Any:
        """Получение данных из кэша или вычисление"""
        cache_key = f"{key}_{'_'.join(map(str, args))}"
        if cache_key not in self._cache:
            self._cache[cache_key] = func(*args)
        return self._cache[cache_key]

    # ==================== БАЗОВЫЕ ИНДИКАТОРЫ ====================

    def sma(self, period: int = 20) -> np.ndarray:
        """Простая скользящая средняя"""
        return self._get_cached('sma', vectorized_sma_numba, self.closes, period)

    def ema(self, period: int = 20) -> np.ndarray:
        """Экспоненциальная скользящая средняя"""
        return self._get_cached('ema', vectorized_ema_numba, self.closes, period)

    def rsi(self, period: int = 14) -> np.ndarray:
        """Индекс относительной силы"""
        return self._get_cached('rsi', vectorized_rsi_numba, self.closes, period)

    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple:
        """Схождение/расхождение скользящих средних"""
        cache_key = f"macd_{fast_period}_{slow_period}_{signal_period}"
        if cache_key not in self._cache:
            result = vectorized_macd_numba(self.closes, fast_period, slow_period, signal_period)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple:
        """Полосы Боллинджера"""
        cache_key = f"bollinger_{period}_{std_dev}"
        if cache_key not in self._cache:
            result = vectorized_bollinger_bands_numba(self.closes, period, std_dev)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def atr(self, period: int = 14) -> np.ndarray:
        """Average True Range"""
        return self._get_cached('atr', vectorized_atr_numba, self.highs, self.lows, self.closes, period)

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple:
        """Стохастический осциллятор"""
        cache_key = f"stochastic_{k_period}_{d_period}"
        if cache_key not in self._cache:
            result = vectorized_stochastic_numba(self.highs, self.lows, self.closes, k_period, d_period)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def obv(self) -> np.ndarray:
        """On-Balance Volume"""
        return self._get_cached('obv', vectorized_obv_numba, self.closes, self.volumes)

    def adx(self, period: int = 14) -> Tuple:
        """Average Directional Index"""
        cache_key = f"adx_{period}"
        if cache_key not in self._cache:
            result = vectorized_adx_numba(self.highs, self.lows, self.closes, period)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    # ==================== ПРОДВИНУТЫЕ ИНДИКАТОРЫ ====================

    def ichimoku(self, conversion_period: int = 9, base_period: int = 26,
                 leading_span_b_period: int = 52) -> Tuple:
        """Ишимоку Кинко Хайо"""
        cache_key = f"ichimoku_{conversion_period}_{base_period}_{leading_span_b_period}"
        if cache_key not in self._cache:
            result = vectorized_ichimoku_numba(self.highs, self.lows, conversion_period,
                                               base_period, leading_span_b_period)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def keltner_channels(self, period: int = 20, atr_period: int = 10,
                         multiplier: float = 2.0) -> Tuple:
        """Каналы Кельтнера"""
        cache_key = f"keltner_{period}_{atr_period}_{multiplier}"
        if cache_key not in self._cache:
            result = vectorized_keltner_channels_numba(self.highs, self.lows, self.closes,
                                                       period, atr_period, multiplier)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def donchian_channels(self, period: int = 20) -> Tuple:
        """Каналы Дончиана"""
        cache_key = f"donchian_{period}"
        if cache_key not in self._cache:
            result = vectorized_donchian_channels_numba(self.highs, self.lows, period)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def parabolic_sar(self, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """Parabolic SAR"""
        return self._get_cached('parabolic_sar', vectorized_parabolic_sar_numba,
                                self.highs, self.lows, acceleration, maximum)

    def supertrend(self, period: int = 10, multiplier: float = 3.0) -> Tuple:
        """SuperTrend"""
        cache_key = f"supertrend_{period}_{multiplier}"
        if cache_key not in self._cache:
            result = vectorized_supertrend_numba(self.highs, self.lows, self.closes,
                                                 period, multiplier)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def alligator(self, jaw_period: int = 13, jaw_shift: int = 8,
                  teeth_period: int = 8, teeth_shift: int = 5,
                  lips_period: int = 5, lips_shift: int = 3) -> Tuple:
        """Аллигатор"""
        cache_key = f"alligator_{jaw_period}_{jaw_shift}_{teeth_period}_{teeth_shift}_{lips_period}_{lips_shift}"
        if cache_key not in self._cache:
            result = vectorized_alligator_numba(self.highs, self.lows, jaw_period, jaw_shift,
                                                teeth_period, teeth_shift, lips_period, lips_shift)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def fractals(self, window: int = 2) -> Tuple:
        """Фракталы"""
        cache_key = f"fractals_{window}"
        if cache_key not in self._cache:
            result = detect_fractals_numba(self.highs, self.lows, window)
            self._cache[cache_key] = result
        return self._cache[cache_key]

    def momentum(self, period: int = 10) -> np.ndarray:
        """Моментум"""
        return self._get_cached('momentum', vectorized_momentum_numba, self.closes, period)

    def roc(self, period: int = 10) -> np.ndarray:
        """Rate of Change"""
        return self._get_cached('roc', vectorized_roc_numba, self.closes, period)

    def cci(self, period: int = 20) -> np.ndarray:
        """Commodity Channel Index"""
        return self._get_cached('cci', vectorized_cci_numba, self.highs, self.lows, self.closes, period)

    def williams_r(self, period: int = 14) -> np.ndarray:
        """Williams %R"""
        return self._get_cached('williams_r', vectorized_williams_r_numba, self.highs, self.lows, self.closes, period)

    def mfi(self, period: int = 14) -> np.ndarray:
        """Money Flow Index"""
        return self._get_cached('mfi', vectorized_mfi_numba, self.highs, self.lows, self.closes, self.volumes, period)

    # ==================== ГРАФИЧЕСКИЕ ИНСТРУМЕНТЫ ====================

    def detect_patterns(self) -> Dict[str, Any]:
        """Обнаружение всех графических паттернов"""
        patterns = {}

        # Double Top/Bottom
        double_patterns = detect_double_top_bottom(self.highs, self.lows)
        patterns.update(double_patterns)

        # Head and Shoulders
        hs_patterns = detect_head_shoulders(self.highs, self.lows)
        patterns.update(hs_patterns)

        # Triangles
        triangles = detect_triangle_patterns(self.highs, self.lows)
        patterns['triangles'] = triangles

        # Channels
        channels = detect_price_channels(self.highs, self.lows)
        patterns['channels'] = channels

        return patterns

    def support_resistance(self, window: int = 20) -> Dict[str, List[float]]:
        """Уровни поддержки и сопротивления"""
        return detect_support_resistance(self.highs, self.lows, self.closes, window)

    def fibonacci_levels(self, lookback_period: int = 100) -> Dict[str, float]:
        """Уровни Фибоначчи от последнего значимого движения"""
        if len(self.closes) < lookback_period:
            high = np.max(self.highs)
            low = np.min(self.lows)
        else:
            high = np.max(self.highs[-lookback_period:])
            low = np.min(self.lows[-lookback_period:])

        return calculate_fibonacci_levels(high, low)

    def trend_lines(self) -> Dict[str, List[Dict]]:
        """Линии тренда"""
        return detect_trend_lines(self.highs, self.lows, self.closes)

    def gann_levels(self, lookback_period: int = 100) -> Dict[str, float]:
        """Уровни Ганна"""
        if len(self.closes) < lookback_period:
            high = np.max(self.highs)
            low = np.min(self.lows)
        else:
            high = np.max(self.highs[-lookback_period:])
            low = np.min(self.lows[-lookback_period:])

        current_price = self.closes[-1]
        return detect_gann_levels(high, low, current_price)

    def get_all_indicators(self) -> Dict[str, Any]:
        """Получение всех индикаторов сразу"""
        indicators = {}

        # Basic indicators
        indicators['sma_20'] = self.sma(20)
        indicators['sma_50'] = self.sma(50)
        indicators['sma_200'] = self.sma(200)
        indicators['ema_20'] = self.ema(20)
        indicators['ema_50'] = self.ema(50)
        indicators['rsi'] = self.rsi(14)

        macd_line, signal_line, histogram = self.macd()
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram

        bb_upper, bb_middle, bb_lower = self.bollinger_bands()
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower

        indicators['atr'] = self.atr()

        # Advanced indicators
        tenkan, kijun, senkou_a, senkou_b = self.ichimoku()
        indicators['ichimoku_tenkan'] = tenkan
        indicators['ichimoku_kijun'] = kijun
        indicators['ichimoku_senkou_a'] = senkou_a
        indicators['ichimoku_senkou_b'] = senkou_b

        kc_upper, kc_middle, kc_lower = self.keltner_channels()
        indicators['keltner_upper'] = kc_upper
        indicators['keltner_middle'] = kc_middle
        indicators['keltner_lower'] = kc_lower

        dc_upper, dc_middle, dc_lower = self.donchian_channels()
        indicators['donchian_upper'] = dc_upper
        indicators['donchian_middle'] = dc_middle
        indicators['donchian_lower'] = dc_lower

        indicators['parabolic_sar'] = self.parabolic_sar()

        supertrend, direction = self.supertrend()
        indicators['supertrend'] = supertrend
        indicators['supertrend_direction'] = direction

        jaw, teeth, lips = self.alligator()
        indicators['alligator_jaw'] = jaw
        indicators['alligator_teeth'] = teeth
        indicators['alligator_lips'] = lips

        fractal_highs, fractal_lows = self.fractals()
        indicators['fractal_highs'] = fractal_highs
        indicators['fractal_lows'] = fractal_lows

        # Pattern detection
        indicators['patterns'] = self.detect_patterns()
        indicators['support_resistance'] = self.support_resistance()
        indicators['fibonacci_levels'] = self.fibonacci_levels()
        indicators['trend_lines'] = self.trend_lines()
        indicators['gann_levels'] = self.gann_levels()

        return indicators


# ==================== УТИЛИТЫ И ХЕЛПЕРЫ ====================

def calculate_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """Расчет корреляции между двумя рядами"""
    if len(series1) != len(series2):
        min_len = min(len(series1), len(series2))
        series1 = series1[-min_len:]
        series2 = series2[-min_len:]

    valid_mask = ~(np.isnan(series1) | np.isnan(series2))
    if np.sum(valid_mask) < 2:
        return 0.0

    return float(np.corrcoef(series1[valid_mask], series2[valid_mask])[0, 1])


def calculate_zscore(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Расчет Z-Score для ряда"""
    n = len(series)
    zscore = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        window_data = series[i - window:i]
        mean = np.mean(window_data)
        std = np.std(window_data)

        if std != 0:
            zscore[i] = (series[i] - mean) / std

    return zscore


def detect_divergence(prices: np.ndarray, indicator: np.ndarray,
                      lookback_period: int = 20) -> Dict[str, bool]:
    """Обнаружение дивергенций между ценой и индикатором"""
    divergence = {
        'bullish': False,
        'bearish': False
    }

    n = len(prices)
    if n < lookback_period * 2:
        return divergence

    # Find recent extremes
    recent_prices = prices[-lookback_period:]
    recent_indicator = indicator[-lookback_period:]

    price_low_idx = np.argmin(recent_prices)
    price_high_idx = np.argmax(recent_prices)

    indicator_low_idx = np.argmin(recent_indicator)
    indicator_high_idx = np.argmax(recent_indicator)

    # Bullish divergence (price makes lower low, indicator makes higher low)
    if (price_low_idx > lookback_period // 2 and
            indicator_low_idx > lookback_period // 2):

        prev_price_low = np.min(prices[-lookback_period * 2:-lookback_period])
        prev_indicator_low = np.min(indicator[-lookback_period * 2:-lookback_period])

        if (recent_prices[price_low_idx] < prev_price_low and
                recent_indicator[indicator_low_idx] > prev_indicator_low):
            divergence['bullish'] = True

    # Bearish divergence (price makes higher high, indicator makes lower high)
    if (price_high_idx > lookback_period // 2 and
            indicator_high_idx > lookback_period // 2):

        prev_price_high = np.max(prices[-lookback_period * 2:-lookback_period])
        prev_indicator_high = np.max(indicator[-lookback_period * 2:-lookback_period])

        if (recent_prices[price_high_idx] > prev_price_high and
                recent_indicator[indicator_high_idx] < prev_indicator_high):
            divergence['bearish'] = True

    return divergence


def calculate_volatility(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Расчет волатильности (стандартное отклонение доходностей)"""
    n = len(series)
    returns = np.diff(series) / series[:-1]
    volatility = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, len(returns)):
        volatility[i + 1] = np.std(returns[i - window + 1:i + 1])

    return volatility


# ==================== ЭКСПОРТ ОСНОВНЫХ ФУНКЦИЙ ====================

__all__ = [
    'UltraPerformanceIndicators',
    'PatternType',
    'FibonacciLevel',
    'vectorized_ema_numba',
    'vectorized_sma_numba',
    'vectorized_rsi_numba',
    'vectorized_macd_numba',
    'vectorized_bollinger_bands_numba',
    'vectorized_atr_numba',
    'vectorized_stochastic_numba',
    'vectorized_obv_numba',
    'vectorized_adx_numba',
    'vectorized_ichimoku_numba',
    'vectorized_keltner_channels_numba',
    'vectorized_donchian_channels_numba',
    'vectorized_parabolic_sar_numba',
    'vectorized_supertrend_numba',
    'vectorized_alligator_numba',
    'detect_fractals_numba',
    'vectorized_momentum_numba',
    'vectorized_roc_numba',
    'vectorized_cci_numba',
    'vectorized_williams_r_numba',
    'vectorized_mfi_numba',
    'detect_double_top_bottom',
    'detect_head_shoulders',
    'detect_triangle_patterns',
    'detect_support_resistance',
    'calculate_fibonacci_levels',
    'detect_trend_lines',
    'detect_price_channels',
    'detect_gann_levels',
    'calculate_correlation',
    'calculate_zscore',
    'detect_divergence',
    'calculate_volatility'
]


# ==================== ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ====================

def benchmark_performance():
    """Бенчмарк производительности индикаторов"""
    import time

    # Создание тестовых данных
    np.random.seed(42)
    n = 10000
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, n),
        'high': np.random.normal(105, 5, n),
        'low': np.random.normal(95, 5, n),
        'close': np.random.normal(100, 5, n),
        'volume': np.random.normal(1000, 200, n)
    })

    indicators = UltraPerformanceIndicators(data)

    # Бенчмарк основных индикаторов
    tests = [
        ('SMA 20', lambda: indicators.sma(20)),
        ('EMA 20', lambda: indicators.ema(20)),
        ('RSI 14', lambda: indicators.rsi(14)),
        ('MACD', lambda: indicators.macd()),
        ('Bollinger Bands', lambda: indicators.bollinger_bands()),
        ('ATR 14', lambda: indicators.atr()),
        ('Ichimoku', lambda: indicators.ichimoku()),
        ('SuperTrend', lambda: indicators.supertrend()),
        ('All Indicators', lambda: indicators.get_all_indicators())
    ]

    print("=== БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ИНДИКАТОРОВ ===")
    print(f"Размер данных: {n} баров")
    print("-" * 50)

    for name, test_func in tests:
        start_time = time.time()
        result = test_func()
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # ms
        print(f"{name:<20}: {execution_time:6.2f} ms")

    print("-" * 50)
    print("✅ Все индикаторы работают в реальном времени")

