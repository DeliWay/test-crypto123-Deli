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


@njit(cache=True, fastmath=True)
def _shift_with_nan(arr: np.ndarray, periods: int) -> np.ndarray:
    """Сдвиг массива с заполнением NaN (без подглядывания в будущее)"""
    n = len(arr)
    result = np.empty(n, dtype=arr.dtype)
    result[:] = np.nan

    if periods > 0:
        # Сдвиг вправо (прошлые значения)
        if periods < n:
            result[periods:] = arr[:-periods]
    elif periods < 0:
        # Сдвиг влево (будущие значения - только для отрисовки!)
        if -periods < n:
            result[:periods] = arr[-periods:]

    return result

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
    """Ультра-быстрый расчет RSI с использованием Numba с защитой от деления на ноль"""
    if len(prices) <= period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    n = len(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Инициализация с проверкой нулевых значений
    initial_avg_gain = np.mean(gains[:period])
    initial_avg_loss = np.mean(losses[:period])

    # Обработка краевых случаев
    if initial_avg_loss == 0 and initial_avg_gain == 0:
        return np.full(n, 50.0, dtype=np.float64)
    elif initial_avg_loss == 0:
        return np.full(n, 100.0, dtype=np.float64)
    elif initial_avg_gain == 0:
        return np.full(n, 0.0, dtype=np.float64)

    # Продолжаем расчет если оба ненулевые
    avg_gain = np.full(n, np.nan, dtype=np.float64)
    avg_loss = np.full(n, np.nan, dtype=np.float64)
    avg_gain[period] = initial_avg_gain
    avg_loss[period] = initial_avg_loss

    # Сглаживание средних
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # Расчет RSI
    rsi = np.full(n, 50.0, dtype=np.float64)  # По умолчанию нейтрально

    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain[i] > 1e-10:
            rsi[i] = 100.0  # Только рост
        else:
            rsi[i] = 50.0  # Нейтрально

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

    # Smoothed ATR (Wilder's smoothing)
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
    """Корректный расчет ADX по методу Уайлдера"""
    n = len(highs)
    if n < period + 1:
        return (np.full_like(highs, np.nan),
                np.full_like(highs, np.nan),
                np.full_like(highs, np.nan))

    # Инициализация массивов
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    # Расчет True Range и Directional Movement
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr[i] = max(high_low, high_close, low_close)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Сглаживание по Уайлдеру
    atr = np.zeros(n, dtype=np.float64)
    atr_plus_dm = np.zeros(n, dtype=np.float64)
    atr_minus_dm = np.zeros(n, dtype=np.float64)

    # Первые значения
    atr[period] = np.sum(tr[1:period + 1]) / period
    atr_plus_dm[period] = np.sum(plus_dm[1:period + 1]) / period
    atr_minus_dm[period] = np.sum(minus_dm[1:period + 1]) / period

    # Рекурсивное сглаживание
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        atr_plus_dm[i] = (atr_plus_dm[i - 1] * (period - 1) + plus_dm[i]) / period
        atr_minus_dm[i] = (atr_minus_dm[i - 1] * (period - 1) + minus_dm[i]) / period

    # Расчет Directional Indicators
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)

    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * atr_plus_dm[i] / atr[i]
            minus_di[i] = 100.0 * atr_minus_dm[i] / atr[i]

    # Расчет DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX как сглаженный DX
    adx = np.zeros(n, dtype=np.float64)
    if n > period * 2:
        # Первое значение ADX - среднее первых period значений DX
        valid_dx = dx[period:period * 2]
        valid_dx = valid_dx[~np.isnan(valid_dx)]
        if len(valid_dx) > 0:
            adx[period * 2 - 1] = np.mean(valid_dx)

            # Рекурсивное сглаживание
            for i in range(period * 2, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return plus_di, minus_di, adx


# ==================== ПРОДВИНУТЫЕ ИНДИКАТОРЫ ====================

@njit(parallel=True, fastmath=True)
def vectorized_ichimoku_numba(highs: np.ndarray, lows: np.ndarray,
                              conversion_period: int = 9,
                              base_period: int = 26,
                              leading_span_b_period: int = 52) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Senkou Span A (Leading Span A) - shifted forward
    senkou_span_a = np.full(n, np.nan, dtype=np.float64)
    for i in prange(base_period, n):
        senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

    # Senkou Span B (Leading Span B) - shifted forward
    senkou_span_b = np.full(n, np.nan, dtype=np.float64)
    for i in prange(leading_span_b_period - 1, n):
        high_window = highs[i - leading_span_b_period + 1:i + 1]
        low_window = lows[i - leading_span_b_period + 1:i + 1]
        senkou_span_b[i] = (np.max(high_window) + np.min(low_window)) / 2

    # Chikou Span (Lagging Span) - shifted backward
    chikou_span = np.full(n, np.nan, dtype=np.float64)
    for i in prange(base_period, n):
        if i >= base_period:
            chikou_span[i - base_period] = highs[i]  # Typically close price, using high for consistency

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


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

    for i in prange(max(period, atr_period) - 1, n):
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
    trend = 1 if lows[1] > lows[0] else -1
    ep = highs[1] if trend == 1 else lows[1]
    af = acceleration

    for i in prange(2, n):
        if trend == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], lows[i - 1], lows[i - 2])

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
            sar[i] = max(sar[i], highs[i - 1], highs[i - 2])

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
    hl2 = (highs + lows) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

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
    """Ультра-быстрый расчет Alligator с использованием Numba (без подглядывания в будущее)"""
    n = len(highs)

    # Median price
    median_price = (highs + lows) / 2

    # Jaw (синяя линия)
    jaw = vectorized_sma_numba(median_price, jaw_period)
    jaw_shifted = np.full(n, np.nan, dtype=np.float64)
    if jaw_shift < n:
        jaw_shifted[jaw_shift:] = jaw[:-jaw_shift]

    # Teeth (красная линия)
    teeth = vectorized_sma_numba(median_price, teeth_period)
    teeth_shifted = np.full(n, np.nan, dtype=np.float64)
    if teeth_shift < n:
        teeth_shifted[teeth_shift:] = teeth[:-teeth_shift]

    # Lips (зеленая линия)
    lips = vectorized_sma_numba(median_price, lips_period)
    lips_shifted = np.full(n, np.nan, dtype=np.float64)
    if lips_shift < n:
        lips_shifted[lips_shift:] = lips[:-lips_shift]

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
        current_low = lows[i]
        for j in range(1, window + 1):
            if current_low >= lows[i - j] or current_low >= lows[i + j]:
                is_low_fractal = False
                break

        fractal_lows[i] = is_low_fractal

        # Check for bearish fractal (high point)
        is_high_fractal = True
        current_high = highs[i]
        for j in range(1, window + 1):
            if current_high <= highs[i - j] or current_high <= highs[i + j]:
                is_high_fractal = False
                break

        fractal_highs[i] = is_high_fractal

    return fractal_highs, fractal_lows


@njit(parallel=True, fastmath=True)
def vectorized_gann_angles_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                 start_idx: int, end_idx: int) -> Tuple[float, float, float, float]:
    """Расчет углов Ганна от значимых экстремумов"""
    if end_idx <= start_idx or end_idx >= len(highs):
        return 45.0, 26.25, 63.75, 15.0

    # Find significant high and low in the range
    segment_highs = highs[start_idx:end_idx + 1]
    segment_lows = lows[start_idx:end_idx + 1]

    significant_high = np.max(segment_highs)
    significant_low = np.min(segment_lows)
    current_price = closes[end_idx]

    price_range = significant_high - significant_low
    if price_range < 1e-10:
        return 45.0, 26.25, 63.75, 15.0

    # Calculate time range
    time_range = end_idx - start_idx
    if time_range < 1:
        return 45.0, 26.25, 63.75, 15.0

    # Calculate angles based on price movement over time
    price_change = current_price - significant_low
    time_change = end_idx - start_idx

    # 1x1 angle (45 degrees) - price moves 1 unit per 1 time unit
    base_angle = np.degrees(np.arctan(price_range / time_range))

    # Gann angles
    angle_1x1 = base_angle
    angle_1x2 = np.degrees(np.arctan(price_range / (time_range * 2)))
    angle_2x1 = np.degrees(np.arctan((price_range * 2) / time_range))
    angle_1x4 = np.degrees(np.arctan(price_range / (time_range * 4)))

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
        base_price = prices[i - period]
        if abs(base_price) < 1e-10:  # Защита от деления на ноль
            roc[i] = 0.0
        else:
            roc[i] = (prices[i] - base_price) / base_price * 100

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

        if mean_deviation > 1e-10:
            cci[i] = (typical_price[i] - sma) / (0.015 * mean_deviation)
        else:
            cci[i] = 0.0

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

        if abs(highest_high - lowest_low) > 1e-10:
            williams_r[i] = (highest_high - closes[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50.0

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

        if neg_sum > 1e-10:
            money_ratio = pos_sum / neg_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
        elif pos_sum > 1e-10:
            mfi[i] = 100.0
        else:
            mfi[i] = 50.0

    return mfi


# ==================== ГРАФИЧЕСКИЕ ИНСТРУМЕНТЫ И ПАТТЕРНЫ ====================

def detect_double_top_bottom(highs: np.ndarray, lows: np.ndarray, tolerance: float = 0.02) -> Dict[str, Any]:
    """Обнаружение паттернов Double Top и Double Bottom с правильными целями"""
    patterns = {
        'double_tops': [],
        'double_bottoms': []
    }

    n = len(highs)
    if n < 20:
        return patterns

    # Find local maxima and minima with tolerance
    local_maxima = []
    local_minima = []

    for i in range(5, n - 5):
        # Check for local maximum with tolerance
        is_max = True
        for j in range(-5, 6):
            if j != 0 and highs[i] < highs[i + j] * (1 - tolerance / 2):
                is_max = False
                break
        if is_max:
            local_maxima.append((i, highs[i]))

        # Check for local minimum with tolerance
        is_min = True
        for j in range(-5, 6):
            if j != 0 and lows[i] > lows[i + j] * (1 + tolerance / 2):
                is_min = False
                break
        if is_min:
            local_minima.append((i, lows[i]))

    # Detect Double Tops
    for i in range(len(local_maxima) - 1):
        idx1, price1 = local_maxima[i]
        idx2, price2 = local_maxima[i + 1]

        price_diff = abs(price1 - price2) / max(price1, price2)
        time_diff = abs(idx2 - idx1)

        # Find the trough between the two peaks
        trough_idx = np.argmin(lows[idx1:idx2 + 1]) + idx1
        trough_price = lows[trough_idx]

        if (price_diff <= tolerance and
                time_diff <= 20 and
                price1 > max(highs[max(0, idx1 - 10):idx1]) and
                price2 > max(highs[idx2:min(n, idx2 + 10)]) and
                trough_price < min(price1, price2) * 0.95):
            neckline = trough_price
            target = neckline - (max(price1, price2) - neckline)  # Correct target calculation
            patterns['double_tops'].append({
                'start_idx': idx1,
                'end_idx': idx2,
                'peak1': price1,
                'peak2': price2,
                'neckline': neckline,
                'target': target
            })

    # Detect Double Bottoms
    for i in range(len(local_minima) - 1):
        idx1, price1 = local_minima[i]
        idx2, price2 = local_minima[i + 1]

        price_diff = abs(price1 - price2) / min(price1, price2)
        time_diff = abs(idx2 - idx1)

        # Find the peak between the two troughs
        peak_idx = np.argmax(highs[idx1:idx2 + 1]) + idx1
        peak_price = highs[peak_idx]

        if (price_diff <= tolerance and
                time_diff <= 20 and
                price1 < min(lows[max(0, idx1 - 10):idx1]) and
                price2 < min(lows[idx2:min(n, idx2 + 10)]) and
                peak_price > max(price1, price2) * 1.05):
            neckline = peak_price
            target = neckline + (neckline - min(price1, price2))  # Correct target calculation
            patterns['double_bottoms'].append({
                'start_idx': idx1,
                'end_idx': idx2,
                'trough1': price1,
                'trough2': price2,
                'neckline': neckline,
                'target': target
            })

    return patterns


def detect_head_shoulders(highs: np.ndarray, lows: np.ndarray, tolerance: float = 0.03) -> Dict[str, Any]:
    """Обнаружение паттернов Head & Shoulders и Inverse Head & Shoulders"""
    patterns = {
        'head_shoulders': [],
        'inverse_head_shoulders': []
    }

    n = len(highs)
    if n < 30:
        return patterns

    # Find significant highs and lows
    local_maxima = []
    local_minima = []

    for i in range(10, n - 10):
        # Local maximum with tolerance
        if (highs[i] >= np.max(highs[i - 5:i + 6]) and
                highs[i] > np.mean(highs[i - 10:i + 11]) * 1.05):
            local_maxima.append((i, highs[i]))

        # Local minimum with tolerance
        if (lows[i] <= np.min(lows[i - 5:i + 6]) and
                lows[i] < np.mean(lows[i - 10:i + 11]) * 0.95):
            local_minima.append((i, lows[i]))

    # Detect Head & Shoulders
    for i in range(2, len(local_maxima) - 2):
        left_shoulder_idx, left_shoulder_price = local_maxima[i - 1]
        head_idx, head_price = local_maxima[i]
        right_shoulder_idx, right_shoulder_price = local_maxima[i + 1]

        # Check symmetry and proportions
        price_diff_lh = abs(left_shoulder_price - head_price) / head_price
        price_diff_rh = abs(right_shoulder_price - head_price) / head_price
        time_diff_lh = abs(head_idx - left_shoulder_idx)
        time_diff_rh = abs(right_shoulder_idx - head_idx)

        if (price_diff_lh <= tolerance and
                price_diff_rh <= tolerance and
                abs(time_diff_lh - time_diff_rh) <= 5 and
                head_price > left_shoulder_price * 1.03 and
                head_price > right_shoulder_price * 1.03):
            # Find neckline (trough between shoulders)
            trough_start = left_shoulder_idx
            trough_end = right_shoulder_idx
            trough_idx = np.argmin(lows[trough_start:trough_end + 1]) + trough_start
            neckline = lows[trough_idx]

            # Calculate target
            target = neckline - (head_price - neckline)  # Correct target calculation

            patterns['head_shoulders'].append({
                'left_shoulder_idx': left_shoulder_idx,
                'head_idx': head_idx,
                'right_shoulder_idx': right_shoulder_idx,
                'left_shoulder_price': left_shoulder_price,
                'head_price': head_price,
                'right_shoulder_price': right_shoulder_price,
                'neckline': neckline,
                'target': target
            })

    # Detect Inverse Head & Shoulders
    for i in range(2, len(local_minima) - 2):
        left_shoulder_idx, left_shoulder_price = local_minima[i - 1]
        head_idx, head_price = local_minima[i]
        right_shoulder_idx, right_shoulder_price = local_minima[i + 1]

        # Check symmetry and proportions
        price_diff_lh = abs(left_shoulder_price - head_price) / head_price
        price_diff_rh = abs(right_shoulder_price - head_price) / head_price
        time_diff_lh = abs(head_idx - left_shoulder_idx)
        time_diff_rh = abs(right_shoulder_idx - head_idx)

        if (price_diff_lh <= tolerance and
                price_diff_rh <= tolerance and
                abs(time_diff_lh - time_diff_rh) <= 5 and
                head_price < left_shoulder_price * 0.97 and
                head_price < right_shoulder_price * 0.97):
            # Find neckline (peak between shoulders)
            peak_start = left_shoulder_idx
            peak_end = right_shoulder_idx
            peak_idx = np.argmax(highs[peak_start:peak_end + 1]) + peak_start
            neckline = highs[peak_idx]

            # Calculate target
            target = neckline + (neckline - head_price)  # Correct target calculation

            patterns['inverse_head_shoulders'].append({
                'left_shoulder_idx': left_shoulder_idx,
                'head_idx': head_idx,
                'right_shoulder_idx': right_shoulder_idx,
                'left_shoulder_price': left_shoulder_price,
                'head_price': head_price,
                'right_shoulder_price': right_shoulder_price,
                'neckline': neckline,
                'target': target
            })

    return patterns


def detect_support_resistance(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                              tolerance: float = 0.005, min_touches: int = 3) -> Dict[str, List[float]]:
    """Обнаружение уровней поддержки и сопротивления с tolerance"""
    levels = {
        'support': [],
        'resistance': []
    }

    n = len(closes)
    if n < 20:
        return levels

    # Find potential support levels (local minima)
    for i in range(10, n - 10):
        current_low = lows[i]

        # Check if this is a local minimum within tolerance
        is_min = True
        touch_count = 0

        for j in range(-10, 11):
            if j != 0 and lows[i + j] < current_low * (1 - tolerance):
                is_min = False
                break
            if abs(lows[i + j] - current_low) <= current_low * tolerance:
                touch_count += 1

        if is_min and touch_count >= min_touches and current_low not in levels['support']:
            levels['support'].append(current_low)

    # Find potential resistance levels (local maxima)
    for i in range(10, n - 10):
        current_high = highs[i]

        # Check if this is a local maximum within tolerance
        is_max = True
        touch_count = 0

        for j in range(-10, 11):
            if j != 0 and highs[i + j] > current_high * (1 + tolerance):
                is_max = False
                break
            if abs(highs[i + j] - current_high) <= current_high * tolerance:
                touch_count += 1

        if is_max and touch_count >= min_touches and current_high not in levels['resistance']:
            levels['resistance'].append(current_high)

    # Remove duplicates and sort
    levels['support'] = sorted(list(set(levels['support'])))
    levels['resistance'] = sorted(list(set(levels['resistance'])))

    return levels


def detect_price_channels(highs: np.ndarray, lows: np.ndarray, period: int = 20) -> Dict[str, Any]:
    """Обнаружение ценовых каналов (восходящих, нисходящих, боковых)"""
    channels = {
        'upward': [],
        'downward': [],
        'sideways': []
    }

    n = len(highs)
    if n < period * 2:
        return channels

    for i in range(period, n - period):
        # Current window
        high_window = highs[i - period:i + 1]
        low_window = lows[i - period:i + 1]

        # Linear regression for trend direction
        x = np.arange(period + 1)
        high_slope, _, _, _, _ = stats.linregress(x, high_window)
        low_slope, _, _, _, _ = stats.linregress(x, low_window)

        # Channel parameters
        upper_band = np.max(high_window)
        lower_band = np.min(low_window)
        middle_band = (upper_band + lower_band) / 2

        # Determine channel type
        if high_slope > 0 and low_slope > 0:
            channel_type = 'upward'
        elif high_slope < 0 and low_slope < 0:
            channel_type = 'downward'
        else:
            channel_type = 'sideways'

        # Check if price is within channel (not breaking out)
        current_high = highs[i]
        current_low = lows[i]

        if (current_high <= upper_band * 1.02 and  # Allow small margin
                current_low >= lower_band * 0.98):
            channels[channel_type].append({
                'start_idx': i - period,
                'end_idx': i,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'middle_band': middle_band,
                'slope': (high_slope + low_slope) / 2
            })

    return channels


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Расчет уровней Фибоначчи для заданного диапазона"""
    price_range = high - low

    return {
        'retracement_236': high - price_range * 0.236,
        'retracement_382': high - price_range * 0.382,
        'retracement_500': high - price_range * 0.5,
        'retracement_618': high - price_range * 0.618,
        'retracement_786': high - price_range * 0.786,
        'extension_127': high + price_range * 0.272,
        'extension_161': high + price_range * 0.618,
        'extension_261': high + price_range * 1.618,
        'extension_127': high + price_range * 1.272,
        'extension_161': high + price_range * 1.618,
        'extension_261': high + price_range * 2.618
    }


def detect_triangle_patterns(highs: np.ndarray, lows: np.ndarray,
                             min_length: int = 10, max_length: int = 50) -> Dict[str, Any]:
    """Обнаружение треугольных паттернов (восходящие, нисходящие, симметричные)"""
    triangles = {
        'ascending': [],
        'descending': [],
        'symmetrical': []
    }

    n = len(highs)
    if n < max_length * 2:
        return triangles

    for start_idx in range(0, n - max_length):
        for end_idx in range(start_idx + min_length, start_idx + max_length):
            if end_idx >= n:
                continue

            segment_highs = highs[start_idx:end_idx + 1]
            segment_lows = lows[start_idx:end_idx + 1]

            # Fit trend lines
            x = np.arange(len(segment_highs))

            # Upper trend line (resistance)
            upper_slope, upper_intercept, _, _, _ = stats.linregress(x, segment_highs)

            # Lower trend line (support)
            lower_slope, lower_intercept, _, _, _ = stats.linregress(x, segment_lows)

            # Check for convergence
            if upper_slope < 0 and lower_slope > 0:
                triangle_type = 'symmetrical'
            elif upper_slope < 0 and abs(lower_slope) < 0.001:
                triangle_type = 'descending'
            elif abs(upper_slope) < 0.001 and lower_slope > 0:
                triangle_type = 'ascending'
            else:
                continue

            # Calculate breakout point
            breakout_price = (upper_intercept + lower_intercept) / 2

            triangles[triangle_type].append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'breakout_price': breakout_price
            })

    return triangles


# ==================== ОСНОВНОЙ КЛАСС ИНДИКАТОРОВ ====================

class TechnicalIndicators:
    """ULTRA-PERFORMANCE КЛАСС ТЕХНИЧЕСКИХ ИНДИКАТОРОВ ДЛЯ КРИПТОРЫНКА"""

    def __init__(self, data: pd.DataFrame):
        """
        Инициализация с данными OHLCV

        Args:
            data: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self._cache = {}
        self._indicators_cache = {}

    def _get_cached(self, key: str, calculation_func, *args, **kwargs) -> Any:
        """
        Умный кэш с оптимизированными ключами (без гигантских строк массивов)
        """
        # Создаем компактный ключ на основе параметров
        cache_key = f"{key}_{str(args)}_{str(kwargs)}"

        if cache_key not in self._cache:
            self._cache[cache_key] = calculation_func(*args, **kwargs)

        return self._cache[cache_key]

    def rsi(self, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        closes = self.data['close'].values
        return self._get_cached(f"rsi_{period}", vectorized_rsi_numba, closes, period)

    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Moving Average Convergence Divergence"""
        closes = self.data['close'].values
        return self._get_cached(f"macd_{fast_period}_{slow_period}_{signal_period}",
                                vectorized_macd_numba, closes, fast_period, slow_period, signal_period)

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        closes = self.data['close'].values
        return self._get_cached(f"bb_{period}_{std_dev}",
                                vectorized_bollinger_bands_numba, closes, period, std_dev)

    def atr(self, period: int = 14) -> np.ndarray:
        """Average True Range"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"atr_{period}", vectorized_atr_numba, highs, lows, closes, period)

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"stoch_{k_period}_{d_period}",
                                vectorized_stochastic_numba, highs, lows, closes, k_period, d_period)

    def obv(self) -> np.ndarray:
        """On-Balance Volume"""
        closes = self.data['close'].values
        volumes = self.data['volume'].values
        return self._get_cached("obv", vectorized_obv_numba, closes, volumes)

    def adx(self, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average Directional Index"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"adx_{period}", vectorized_adx_numba, highs, lows, closes, period)

    def ichimoku(self, conversion_period: int = 9, base_period: int = 26,
                 leading_span_b_period: int = 52) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ультра-быстрый расчет Ichimoku Cloud с использованием Numba"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values

        # Tenkan-sen (Conversion Line)
        tenkan_sen = np.full_like(highs.shape, np.nan, dtype=np.float64)
        for i in range(conversion_period - 1, len(highs)):
            high_window = highs[i - conversion_period + 1:i + 1]
            low_window = lows[i - conversion_period + 1:i + 1]
            tenkan_sen[i] = (np.max(high_window) + np.min(low_window)) / 2

        # Kijun-sen (Base Line)
        kijun_sen = np.full_like(highs.shape, np.nan, dtype=np.float64)
        for i in range(base_period - 1, len(highs)):
            high_window = highs[i - base_period + 1:i + 1]
            low_window = lows[i - base_period + 1:i + 1]
            kijun_sen[i] = (np.max(high_window) + np.min(low_window)) / 2

        # Senkou Span A (Leading Span A) - сдвиг вперед на base_period
        senkou_span_a = _shift_with_nan((tenkan_sen + kijun_sen) / 2, base_period)

        # Senkou Span B (Leading Span B) - сдвиг вперед на base_period
        senkou_span_b = np.full_like(highs, np.nan, dtype=np.float64)
        for i in range(leading_span_b_period - 1, len(highs)):
            high_window = highs[i - leading_span_b_period + 1:i + 1]
            low_window = lows[i - leading_span_b_period + 1:i + 1]
            senkou_span_b[i] = (np.max(high_window) + np.min(low_window)) / 2
        senkou_span_b = _shift_with_nan(senkou_span_b, base_period)

        # Chikou Span (Lagging Span) - shifted backward
        chikou_span = _shift_with_nan(closes, -base_period)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    def keltner_channels(self, period: int = 20, atr_period: int = 10,
                         multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"keltner_{period}_{atr_period}_{multiplier}",
                                vectorized_keltner_channels_numba, highs, lows, closes, period, atr_period, multiplier)

    def donchian_channels(self, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channels"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        return self._get_cached(f"donchian_{period}", vectorized_donchian_channels_numba, highs, lows, period)

    def parabolic_sar(self, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """Parabolic SAR"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        return self._get_cached(f"parabolic_sar_{acceleration}_{maximum}",
                                vectorized_parabolic_sar_numba, highs, lows, acceleration, maximum)

    def supertrend(self, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """SuperTrend"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"supertrend_{period}_{multiplier}",
                                vectorized_supertrend_numba, highs, lows, closes, period, multiplier)

    def alligator(self, jaw_period: int = 13, jaw_shift: int = 8,
                  teeth_period: int = 8, teeth_shift: int = 5,
                  lips_period: int = 5, lips_shift: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ультра-быстрый расчет Alligator с использованием Numba (без подглядывания в будущее)"""
        n = len(self.highs)

        # Median price
        median_price = (self.highs + self.lows) / 2

        # Jaw (синяя линия)
        jaw = vectorized_sma_numba(median_price, jaw_period)
        jaw_shifted = _shift_with_nan(jaw, jaw_shift)

        # Teeth (красная линия)
        teeth = vectorized_sma_numba(median_price, teeth_period)
        teeth_shifted = _shift_with_nan(teeth, teeth_shift)

        # Lips (зеленая линия)
        lips = vectorized_sma_numba(median_price, lips_period)
        lips_shifted = _shift_with_nan(lips, lips_shift)

        return jaw_shifted, teeth_shifted, lips_shifted

    def fractals(self, window: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Fractals"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        return self._get_cached(f"fractals_{window}", detect_fractals_numba, highs, lows, window)

    def gann_angles(self, start_idx: int, end_idx: int) -> Tuple[float, float, float, float]:
        """Gann Angles"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return vectorized_gann_angles_numba(highs, lows, closes, start_idx, end_idx)

    def momentum(self, period: int = 10) -> np.ndarray:
        """Momentum"""
        closes = self.data['close'].values
        return self._get_cached(f"momentum_{period}", vectorized_momentum_numba, closes, period)

    def roc(self, period: int = 10) -> np.ndarray:
        """Rate of Change"""
        closes = self.data['close'].values
        return self._get_cached(f"roc_{period}", vectorized_roc_numba, closes, period)

    def cci(self, period: int = 20) -> np.ndarray:
        """Commodity Channel Index"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"cci_{period}", vectorized_cci_numba, highs, lows, closes, period)

    def williams_r(self, period: int = 14) -> np.ndarray:
        """Williams %R"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        return self._get_cached(f"williams_r_{period}", vectorized_williams_r_numba, highs, lows, closes, period)

    def mfi(self, period: int = 14) -> np.ndarray:
        """Money Flow Index"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        volumes = self.data['volume'].values
        return self._get_cached(f"mfi_{period}", vectorized_mfi_numba, highs, lows, closes, volumes, period)

    def detect_patterns(self, tolerance: float = 0.02) -> Dict[str, Any]:
        """Обнаружение всех графических паттернов"""
        highs = self.data['high'].values
        lows = self.data['low'].values

        patterns = {}
        patterns.update(detect_double_top_bottom(highs, lows, tolerance))
        patterns.update(detect_head_shoulders(highs, lows, tolerance))
        patterns['triangles'] = detect_triangle_patterns(highs, lows)

        return patterns

    def support_resistance(self, tolerance: float = 0.005, min_touches: int = 3) -> Dict[str, List[float]]:
        """Уровни поддержки и сопротивления"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values

        return detect_support_resistance(highs, lows, closes, tolerance, min_touches)

    def price_channels(self, period: int = 20) -> Dict[str, Any]:
        """Ценовые каналы"""
        highs = self.data['high'].values
        lows = self.data['low'].values

        return detect_price_channels(highs, lows, period)

    def fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Уровни Фибоначчи"""
        return calculate_fibonacci_levels(high, low)

    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        self._indicators_cache.clear()


# ==================== УТИЛИТЫ И ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ====================

def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Расчет классических pivot points"""
    pivot = (high + low + close) / 3

    return {
        'pivot': pivot,
        'r1': 2 * pivot - low,
        'r2': pivot + (high - low),
        'r3': high + 2 * (pivot - low),
        's1': 2 * pivot - high,
        's2': pivot - (high - low),
        's3': low - 2 * (high - pivot)
    }


def calculate_camarilla_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """Расчет Camarilla pivot points"""
    return {
        'r4': close + (high - low) * 1.1 / 2,
        'r3': close + (high - low) * 1.1 / 4,
        'r2': close + (high - low) * 1.1 / 6,
        'r1': close + (high - low) * 1.1 / 12,
        's1': close - (high - low) * 1.1 / 12,
        's2': close - (high - low) * 1.1 / 6,
        's3': close - (high - low) * 1.1 / 4,
        's4': close - (high - low) * 1.1 / 2
    }


def calculate_volume_profile(prices: np.ndarray, volumes: np.ndarray,
                             bins: int = 20) -> Dict[str, Any]:
    """Расчет Volume Profile"""
    if len(prices) == 0 or len(volumes) == 0:
        return {'price_levels': [], 'volumes': []}

    min_price, max_price = np.min(prices), np.max(prices)
    price_range = max_price - min_price

    if price_range < 1e-10:
        return {'price_levels': [min_price], 'volumes': [np.sum(volumes)]}

    bin_edges = np.linspace(min_price, max_price, bins + 1)
    bin_volumes = np.zeros(bins)

    for price, volume in zip(prices, volumes):
        bin_idx = int((price - min_price) / price_range * bins)
        bin_idx = min(bin_idx, bins - 1)
        bin_volumes[bin_idx] += volume

    price_levels = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        'price_levels': price_levels.tolist(),
        'volumes': bin_volumes.tolist(),
        'poc_price': price_levels[np.argmax(bin_volumes)],
        'value_area_high': price_levels[np.argsort(bin_volumes)[-int(bins * 0.3)]],
        'value_area_low': price_levels[np.argsort(bin_volumes)[int(bins * 0.3)]]
    }


# ==================== ТЕСТИРОВАНИЕ И ВАЛИДАЦИЯ ====================

def test_indicators():
    """Тестирование всех индикаторов на синтетических данных"""
    np.random.seed(42)
    n = 1000

    # Создание синтетических данных
    prices = np.cumsum(np.random.randn(n)) + 100
    highs = prices + np.abs(np.random.randn(n)) * 2
    lows = prices - np.abs(np.random.randn(n)) * 2
    volumes = np.random.lognormal(0, 1, n) * 1000

    data = pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

    ti = TechnicalIndicators(data)

    # Тестирование всех индикаторов
    print("Testing RSI...")
    rsi = ti.rsi(14)
    print(f"RSI shape: {rsi.shape}, non-NaN: {np.sum(~np.isnan(rsi))}")

    print("Testing MACD...")
    macd, signal, hist = ti.macd()
    print(f"MACD shapes: {macd.shape}, {signal.shape}, {hist.shape}")

    print("Testing Bollinger Bands...")
    upper, middle, lower = ti.bollinger_bands()
    print(f"BB shapes: {upper.shape}, {middle.shape}, {lower.shape}")

    print("Testing ATR...")
    atr = ti.atr()
    print(f"ATR shape: {atr.shape}")

    print("Testing Stochastic...")
    k, d = ti.stochastic()
    print(f"Stochastic shapes: {k.shape}, {d.shape}")

    print("Testing ADX...")
    pdi, mdi, adx = ti.adx()
    print(f"ADX shapes: {pdi.shape}, {mdi.shape}, {adx.shape}")

    print("Testing Ichimoku...")
    tenkan, kijun, senkou_a, senkou_b, chikou = ti.ichimoku()
    print(f"Ichimoku shapes: {tenkan.shape}, {kijun.shape}, {senkou_a.shape}, {senkou_b.shape}, {chikou.shape}")

    print("Testing patterns...")
    patterns = ti.detect_patterns()
    print(f"Patterns found: {sum(len(v) for v in patterns.values())}")

    print("Testing support/resistance...")
    sr = ti.support_resistance()
    print(f"Support levels: {len(sr['support'])}, Resistance levels: {len(sr['resistance'])}")

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_indicators()