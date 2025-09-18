# analysis/signals.py
"""
ULTRA-PERFORMANCE TRADING SIGNALS MODULE V2
Улучшенное обнаружение сигналов с адаптивными порогами
Интеграция с реальными рыночными данными и ChatGPT-API для калибровки
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import dataclass
import hashlib
import time
import pandas as pd

# Импорт мощных индикаторов
from .indicators.indicators import (
    TechnicalIndicators,
    vectorized_ema_numba,
    vectorized_rsi_numba,
    vectorized_macd_numba,
    detect_support_resistance
)

# Импорт адаптивного модуля
from .adaptive import (
    AssetProfile,
    calculate_asset_profile,
    adaptive_rsi_levels,
    adaptive_supertrend_multiplier,
    adaptive_stop_loss_take_profit
)

logger = logging.getLogger(__name__)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ АНАЛИЗА СИГНАЛОВ ====================

def detect_crossover(line1: np.ndarray, line2: np.ndarray, lookback: int = 2) -> str:
    """
    Обнаружение пересечения двух линий с подтверждением
    """
    if len(line1) < lookback + 1 or len(line2) < lookback + 1:
        return "no_cross"

    # Проверяем пересечение в последних барах
    current_diff = line1[-1] - line2[-1]
    prev_diff = line1[-2] - line2[-2]

    # Бычье пересечение (line1 пересекает line2 снизу вверх)
    if prev_diff <= 0 and current_diff > 0:
        # Подтверждение: проверяем, что пересечение сохраняется
        if all(line1[-i] > line2[-i] for i in range(1, min(lookback + 1, len(line1)))):
            return "bullish_cross"

    # Медвежье пересечение (line1 пересекает line2 сверху вниз)
    elif prev_diff >= 0 and current_diff < 0:
        # Подтверждение: проверяем, что пересечение сохраняется
        if all(line1[-i] < line2[-i] for i in range(1, min(lookback + 1, len(line1)))):
            return "bearish_cross"

    return "no_cross"


def detect_divergence(prices: np.ndarray, indicator: np.ndarray, lookback: int = 20) -> str:
    """
    Обнаружение дивергенции между ценой и индикатором
    """
    if len(prices) < lookback or len(indicator) < lookback:
        return "no_divergence"

    # Находим экстремумы цены
    price_highs = []
    price_lows = []

    for i in range(len(prices) - lookback, len(prices)):
        if i < 2 or i >= len(prices) - 2:
            continue

        # Локальный максимум цены
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            price_highs.append((i, prices[i]))

        # Локальный минимум цены
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            price_lows.append((i, prices[i]))

    # Находим экстремумы индикатора
    indicator_highs = []
    indicator_lows = []

    for i in range(len(indicator) - lookback, len(indicator)):
        if i < 2 or i >= len(indicator) - 2:
            continue

        # Локальный максимум индикатора
        if indicator[i] > indicator[i - 1] and indicator[i] > indicator[i + 1]:
            indicator_highs.append((i, indicator[i]))

        # Локальный минимум индикатора
        if indicator[i] < indicator[i - 1] and indicator[i] < indicator[i + 1]:
            indicator_lows.append((i, indicator[i]))

    # Проверяем бычью дивергенцию (цена делает нижние минимумы, индикатор - более высокие)
    if len(price_lows) >= 2 and len(indicator_lows) >= 2:
        price_low1, price_low2 = price_lows[-2][1], price_lows[-1][1]
        indicator_low1, indicator_low2 = indicator_lows[-2][1], indicator_lows[-1][1]

        if price_low2 < price_low1 and indicator_low2 > indicator_low1:
            return "bullish_divergence"

    # Проверяем медвежью дивергенцию (цена делает более высокие максимумы, индикатор - более низкие)
    if len(price_highs) >= 2 and len(indicator_highs) >= 2:
        price_high1, price_high2 = price_highs[-2][1], price_highs[-1][1]
        indicator_high1, indicator_high2 = indicator_highs[-2][1], indicator_highs[-1][1]

        if price_high2 > price_high1 and indicator_high2 < indicator_high1:
            return "bearish_divergence"

    return "no_divergence"


def detect_overbought_oversold(values: np.ndarray, overbought: float = 70.0, oversold: float = 30.0) -> str:
    """
    Определение состояния перекупленности/перепроданности
    """
    if len(values) == 0:
        return "neutral"

    current_value = values[-1]

    if current_value > overbought:
        return "overbought"
    elif current_value < oversold:
        return "oversold"
    else:
        return "neutral"


def calculate_momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Расчет момента ценового движения
    """
    if len(prices) < period:
        return np.array([], dtype=float)

    momentum = np.full(len(prices), np.nan, dtype=float)

    for i in range(period, len(prices)):
        momentum[i] = prices[i] - prices[i - period]

    return momentum


def calculate_rate_of_change(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Расчет скорости изменения цены (ROC)
    """
    if len(prices) < period:
        return np.array([], dtype=float)

    roc = np.full(len(prices), np.nan, dtype=float)

    for i in range(period, len(prices)):
        if abs(prices[i - period]) > 1e-10:  # Защита от деления на ноль
            roc[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100
        else:
            roc[i] = 0

    return roc


# ==================== ОСНОВНАЯ ЛОГИКА СИГНАЛОВ ====================

# Конфигурация
CACHE_TTL = 30  # секунды
LLM_CACHE_TTL = 3600  # 1 час для LLM калибровок
MIN_DATA_POINTS = 20


class SignalType(Enum):
    """Типы торговых сигналов"""
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"
    WATCH = "watch"


class SignalConfidence(Enum):
    """Уверенность в сигнале"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketRegime(Enum):
    """Режимы рынка"""
    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class SignalWeightConfig:
    """Конфигурация весов сигналов для разных режимов рынка"""
    ema_weight: float = 1.0
    macd_weight: float = 1.0
    rsi_weight: float = 1.0
    bollinger_weight: float = 1.0
    volume_weight: float = 1.0
    trend_weight: float = 1.0
    supertrend_weight: float = 1.0


@dataclass
class SignalThresholds:
    """Адаптивные пороги для сигналов"""
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bollinger_breakout_threshold: float = 0.02  # 2% от полосы
    volume_spike_threshold: float = 2.0  # 2x от среднего
    min_trend_strength: float = 0.3
    confirmation_bars: int = 2  # Бары подтверждения


class SignalCache:
    """Кэш для сигналов и калибровок"""

    def __init__(self):
        self._llm_timestamps = {}  # Добавляем отслеживание времени для LLM калибровок
        self._cache = {}
        self._timestamps = {}
        self._llm_calibrations = {}

    def get_key(self, func_name: str, *args) -> str:
        """Генерация ключа кэша с защитой от bytes"""
        parts = []
        for a in args:
            if isinstance(a, (bytes, bytearray)):
                parts.append(hashlib.md5(a).hexdigest())
            else:
                try:
                    parts.append(json.dumps(a, sort_keys=True, default=str))
                except Exception:
                    parts.append(str(a))
        args_str = "|".join(parts)
        return f"{func_name}:{hashlib.md5(args_str.encode()).hexdigest()}"

    def get(self, key: str, ttl: int = CACHE_TTL) -> Optional[Any]:
        """Получение данных из кэша"""
        if key in self._cache and time.time() - self._timestamps[key] < ttl:
            return self._cache[key]
        return None

    def set(self, key: str, data: Any, ttl: int = CACHE_TTL):
        """Сохранение данных в кэш"""
        self._cache[key] = data
        self._timestamps[key] = time.time()

    def get_llm_calibration(self, symbol: str, regime: str) -> Optional[SignalWeightConfig]:
        """Получение калибровки от LLM с проверкой TTL"""
        key = f"llm_calibration:{symbol}:{regime}"
        # Проверяем наличие и свежесть данных, как в основном кэше
        if key in self._llm_calibrations and time.time() - self._llm_timestamps[key] < LLM_CACHE_TTL:
            return self._llm_calibrations[key]
        return None

    def set_llm_calibration(self, symbol: str, regime: str, calibration: SignalWeightConfig):
        """Сохранение калибровки от LLM"""
        key = f"llm_calibration:{symbol}:{regime}"
        self._llm_calibrations[key] = calibration
        self._llm_timestamps[key] = time.time()  # Сохраняем временную метку


# Глобальный кэш
signal_cache = SignalCache()


def determine_market_regime(prices: np.ndarray, volumes: Optional[np.ndarray] = None,
                            lookback: int = 50) -> MarketRegime:
    """
    Определение режима рынка на основе цены и объема
    """
    if len(prices) < lookback:
        return MarketRegime.RANGING

    recent_prices = prices[-lookback:]

    # Защита от одинаковых цен
    if np.all(recent_prices == recent_prices[0]):
        return MarketRegime.RANGING

    try:
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Волатильность
        volatility = np.std(returns) * np.sqrt(252) * 100  # Годовая волатильность в %

        # Тренд
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        trend_strength = abs(slope) / np.mean(recent_prices) * 100 * np.sqrt(len(recent_prices))

        # Определение режима
        if trend_strength > 1.0 and slope > 0:
            return MarketRegime.TREND_BULLISH
        elif trend_strength > 1.0 and slope < 0:
            return MarketRegime.TREND_BEARISH
        elif volatility > 40:
            return MarketRegime.VOLATILE
        elif volatility < 15:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGING

    except (ZeroDivisionError, np.linalg.LinAlgError):
        return MarketRegime.RANGING


def get_regime_weights(regime: MarketRegime, symbol: str = None) -> SignalWeightConfig:
    """
    Получение весов индикаторов для текущего режима рынка
    """
    # Попытка получить калибровку от LLM
    if symbol:
        llm_calibration = signal_cache.get_llm_calibration(symbol, regime.value)
        if llm_calibration:
            return llm_calibration

    # Базовые веса по умолчанию для разных режимов
    if regime == MarketRegime.TREND_BULLISH:
        return SignalWeightConfig(
            ema_weight=1.2, macd_weight=1.1, rsi_weight=0.8,
            bollinger_weight=0.7, volume_weight=1.0, trend_weight=1.3,
            supertrend_weight=1.2
        )
    elif regime == MarketRegime.TREND_BEARISH:
        return SignalWeightConfig(
            ema_weight=1.2, macd_weight=1.1, rsi_weight=0.8,
            bollinger_weight=0.7, volume_weight=1.0, trend_weight=1.3,
            supertrend_weight=1.2
        )
    elif regime == MarketRegime.VOLATILE:
        return SignalWeightConfig(
            ema_weight=0.8, macd_weight=0.9, rsi_weight=1.1,
            bollinger_weight=1.3, volume_weight=1.2, trend_weight=0.7,
            supertrend_weight=1.1
        )
    elif regime == MarketRegime.LOW_VOLATILITY:
        return SignalWeightConfig(
            ema_weight=0.7, macd_weight=0.8, rsi_weight=1.2,
            bollinger_weight=1.4, volume_weight=0.9, trend_weight=0.6,
            supertrend_weight=0.9
        )
    else:  # RANGING
        return SignalWeightConfig(
            ema_weight=0.9, macd_weight=0.9, rsi_weight=1.3,
            bollinger_weight=1.5, volume_weight=1.0, trend_weight=0.8,
            supertrend_weight=1.0
        )


def analyze_ema_signals(ema_fast: np.ndarray, ema_slow: np.ndarray,
                        prices: np.ndarray, confirmation_bars: int = 2) -> Optional[Dict[str, Any]]:
    if len(ema_fast) < confirmation_bars + 1 or len(ema_slow) < confirmation_bars + 1:
        return None

    current_diff = ema_fast[-1] - ema_slow[-1]
    prev_diff = ema_fast[-2] - ema_slow[-2]

    recent_diffs = ema_fast[-20:] - ema_slow[-20:] if len(ema_fast) >= 20 else np.array([current_diff])
    diff_std = np.std(recent_diffs) if len(recent_diffs) > 1 else abs(current_diff) * 0.1

    conf_ok_buy = all(ema_fast[-i] > ema_slow[-i] for i in range(1, max(2, confirmation_bars)))
    is_bullish_cross = (prev_diff <= 0 and current_diff > diff_std * 0.5 and conf_ok_buy)

    conf_ok_sell = all(ema_fast[-i] < ema_slow[-i] for i in range(1, max(2, confirmation_bars)))
    is_bearish_cross = (prev_diff >= 0 and current_diff < -diff_std * 0.5 and conf_ok_sell)

    if is_bullish_cross:
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "EMA",
            "message": f"Бычье пересечение EMA с подтверждением ({confirmation_bars} бара)",
            "strength": 75,
            "details": {
                "fast_ema": float(ema_fast[-1]),
                "slow_ema": float(ema_slow[-1]),
                "gap_percent": float((current_diff / ema_slow[-1]) * 100) if abs(ema_slow[-1]) > 1e-10 else 0.0
            }
        }

    if is_bearish_cross:
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "EMA",
            "message": f"Медвежье пересечение EMA с подтверждением ({confirmation_bars} бара)",
            "strength": 75,
            "details": {
                "fast_ema": float(ema_fast[-1]),
                "slow_ema": float(ema_slow[-1]),
                "gap_percent": float((current_diff / ema_slow[-1]) * 100) if abs(ema_slow[-1]) > 1e-10 else 0.0
            }
        }

    return None


def analyze_macd_signals(macd_line: np.ndarray, macd_signal_line: np.ndarray,
                         prices: np.ndarray, confirmation_bars: int = 2) -> Optional[Dict[str, Any]]:
    """
    Улучшенный анализ сигналов MACD с подтверждением.
    Подтверждение проверяем ПОСЛЕ кросса (не включаем бар -2 в проверку),
    чтобы кроссы на хвосте не отбрасывались.
    """
    if macd_line is None or macd_signal_line is None:
        return None
    if len(macd_line) < confirmation_bars + 1 or len(macd_signal_line) < confirmation_bars + 1:
        return None

    current_diff = macd_line[-1] - macd_signal_line[-1]
    prev_diff = macd_line[-2] - macd_signal_line[-2]

    conf_ok_buy = all(macd_line[-i] > macd_signal_line[-i] for i in range(1, max(2, confirmation_bars)))
    is_bullish_cross = (prev_diff <= 0 and current_diff > 0 and conf_ok_buy)

    conf_ok_sell = all(macd_line[-i] < macd_signal_line[-i] for i in range(1, max(2, confirmation_bars)))
    is_bearish_cross = (prev_diff >= 0 and current_diff < 0 and conf_ok_sell)

    if is_bullish_cross:
        divergence = detect_divergence(prices, macd_line)
        divergence_type = "bullish_divergence" if divergence == "bullish_divergence" else "none"
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "MACD",
            "message": f"Бычье пересечение MACD с подтверждением ({confirmation_bars} бара)"
                       + (", с бычьей дивергенцией" if divergence_type == "bullish_divergence" else ""),
            "strength": 80 if divergence_type == "bullish_divergence" else 70,
            "details": {
                "macd_line": float(macd_line[-1]),
                "macd_signal": float(macd_signal_line[-1]),
                "histogram": float(current_diff),
                "divergence": divergence_type
            }
        }

    if is_bearish_cross:
        divergence = detect_divergence(prices, macd_line)
        divergence_type = "bearish_divergence" if divergence == "bearish_divergence" else "none"
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "MACD",
            "message": f"Медвежье пересечение MACD с подтверждением ({confirmation_bars} бара)"
                       + (", с медвежьей дивергенцией" if divergence_type == "bearish_divergence" else ""),
            "strength": 80 if divergence_type == "bearish_divergence" else 70,
            "details": {
                "macd_line": float(macd_line[-1]),
                "macd_signal": float(macd_signal_line[-1]),
                "histogram": float(current_diff),
                "divergence": divergence_type
            }
        }

    return None


def analyze_rsi_signals(rsi_values: np.ndarray, prices: np.ndarray,
                        asset_profile: AssetProfile, regime: MarketRegime) -> List[Dict[str, Any]]:
    """
    Анализ сигналов RSI с адаптивными уровнями
    """
    signals = []

    if len(rsi_values) < 2:
        return signals

    # Адаптивные уровни перекупленности/перепроданности
    overbought, oversold = adaptive_rsi_levels(asset_profile)

    current_rsi = rsi_values[-1]
    prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else current_rsi

    # Перекупленность/перепроданность
    if current_rsi > overbought:
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "RSI",
            "message": f"Перекупленность RSI ({current_rsi:.1f} > {overbought:.1f})",
            "strength": 75,
            "details": {
                "rsi_value": float(current_rsi),
                "threshold": float(overbought),
                "trend": "overbought"
            }
        })
    elif current_rsi < oversold:
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "RSI",
            "message": f"Перепроданность RSI ({current_rsi:.1f} < {oversold:.1f})",
            "strength": 75,
            "details": {
                "rsi_value": float(current_rsi),
                "threshold": float(oversold),
                "trend": "oversold"
            }
        })

    # Анализ дивергенции
    divergence = detect_divergence(prices, rsi_values)
    if divergence == "bullish_divergence":
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "RSI",
            "message": "Бычья дивергенция RSI",
            "strength": 65,
            "details": {"divergence_type": "bullish"}
        })
    elif divergence == "bearish_divergence":
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "RSI",
            "message": "Медвежья дивергенция RSI",
            "strength": 65,
            "details": {"divergence_type": "bearish"}
        })

    return signals


def analyze_bollinger_bands_signals(prices: np.ndarray, bb_upper: np.ndarray,
                                    bb_lower: np.ndarray, bb_middle: np.ndarray,
                                    threshold: float = 0.02) -> List[Dict[str, Any]]:
    """
    Анализ сигналов Bollinger Bands с адаптивным порогом
    """
    signals = []

    if not prices.size or not bb_upper.size or not bb_lower.size or len(prices) < 3:
        return signals

    current_price = prices[-1]
    current_upper = bb_upper[-1]
    current_lower = bb_lower[-1]
    current_middle = bb_middle[-1]

    # Проверка на NaN значения и защита от деления на ноль
    if (np.isnan(current_upper) or np.isnan(current_lower) or
            np.isnan(bb_upper[-2]) or np.isnan(bb_lower[-2]) or
            abs(current_upper) < 1e-10 or abs(current_lower) < 1e-10):
        return signals

    # Относительное расстояние до полос с защитой от деления на ноль
    upper_distance = (current_price - current_upper) / max(current_upper, 1e-10)
    lower_distance = (current_lower - current_price) / max(current_lower, 1e-10)

    # Пробитие верхней полосы
    if upper_distance > threshold:
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": f"Цена выше верхней полосы Bollinger Bands на {upper_distance * 100:.1f}%",
            "strength": 70,
            "details": {
                "price": float(current_price),
                "upper_band": float(current_upper),
                "deviation_percent": float(upper_distance * 100)
            }
        })

    # Пробитие нижней полосы
    if lower_distance > threshold:
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": f"Цена ниже нижней полосы Bollinger Bands на {lower_distance * 100:.1f}%",
            "strength": 70,
            "details": {
                "price": float(current_price),
                "lower_band": float(current_lower),
                "deviation_percent": float(lower_distance * 100)
            }
        })

    # Отскок от полос
    if (abs(upper_distance) < threshold * 0.5 and
            prices[-2] > bb_upper[-2] and
            current_price <= current_upper):
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": "Отскок от верхней полосы Bollinger Bands",
            "strength": 65,
            "details": {"pattern": "upper_band_rejection"}
        })

    if (abs(lower_distance) < threshold * 0.5 and
            prices[-2] < bb_lower[-2] and
            current_price >= current_lower):
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": "Отскок от нижней полосы Bollinger Bands",
            "strength": 65,
            "details": {"pattern": "lower_band_bounce"}
        })

    return signals


def analyze_volume_signals(prices: np.ndarray, volumes: np.ndarray,
                           lookback: int = 20, threshold: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Анализ сигналов объема с z-score нормализацией
    """
    if len(prices) < lookback or len(volumes) < lookback:
        return None

    # Расчет z-score объема
    recent_volumes = volumes[-lookback:]
    volume_mean = np.mean(recent_volumes)
    volume_std = np.std(recent_volumes)

    if volume_std < 1e-10:
        return None

    current_volume = volumes[-1]
    volume_z = (current_volume - volume_mean) / volume_std

    # Ценовое движение
    price_change = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 and abs(prices[-2]) > 1e-10 else 0

    # Сигналы на основе объема и цены
    if volume_z > threshold and price_change > 1.0:
        # Сильный объем при росте цены
        recent_high = np.max(prices[-lookback // 2:])
        is_breakout = prices[-1] > recent_high

        return {
            "type": SignalType.STRONG_BUY.value if is_breakout else SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Volume",
            "message": f"Сильный объем ({volume_z:.1f}σ) при росте цены ({price_change:.1f}%)" +
                       (" с пробоем сопротивления" if is_breakout else ""),
            "strength": 85 if is_breakout else 75,
            "details": {
                "volume_z_score": float(volume_z),
                "price_change_percent": float(price_change),
                "is_breakout": is_breakout,
                "breakout_level": float(recent_high) if is_breakout else None
            }
        }

    elif volume_z > threshold and price_change < -1.0:
        # Сильный объем при падении цены
        recent_low = np.min(prices[-lookback // 2:])
        is_breakdown = prices[-1] < recent_low

        return {
            "type": SignalType.STRONG_SELL.value if is_breakdown else SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Volume",
            "message": f"Сильный объем ({volume_z:.1f}σ) при падении цены ({price_change:.1f}%)" +
                       (" с пробоем поддержки" if is_breakdown else ""),
            "strength": 85 if is_breakdown else 75,
            "details": {
                "volume_z_score": float(volume_z),
                "price_change_percent": float(price_change),
                "is_breakdown": is_breakdown,
                "breakdown_level": float(recent_low) if is_breakdown else None
            }
        }

    return None


def analyze_trend_signals(prices: np.ndarray, min_strength: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Анализ сигналов тренда с определением силы
    """
    if len(prices) < 20:
        return None

    # Анализ краткосрочного и долгосрочного трендов
    short_term = prices[-20:]  # 20 баров
    medium_term = prices[-50:] if len(prices) >= 50 else prices
    long_term = prices[-100:] if len(prices) >= 100 else prices

    def calculate_trend_strength(price_data):
        try:
            x = np.arange(len(price_data))
            slope, _ = np.polyfit(x, price_data, 1)
            mean_price = np.mean(price_data)
            if abs(mean_price) < 1e-10:
                return 0.0
            return slope / mean_price * 100 * np.sqrt(len(price_data))
        except (ZeroDivisionError, np.linalg.LinAlgError):
            return 0.0

    short_strength = calculate_trend_strength(short_term)
    medium_strength = calculate_trend_strength(medium_term)
    long_strength = calculate_trend_strength(long_term)

    # Совпадение трендов
    trend_alignment = (np.sign(short_strength) == np.sign(medium_strength) == np.sign(long_strength))
    overall_strength = (abs(short_strength) + abs(medium_strength) + abs(long_strength)) / 3

    if overall_strength < min_strength:
        return None

    if trend_alignment and short_strength > 0:
        return {
            "type": SignalType.STRONG_BUY.value if overall_strength > 0.8 else SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Trend",
            "message": f"Восходящий тренд (сила: {overall_strength:.2f})",
            "strength": min(95, int(overall_strength * 100)),
            "details": {
                "short_term_strength": float(short_strength),
                "medium_term_strength": float(medium_strength),
                "long_term_strength": float(long_strength),
                "alignment": True
            }
        }
    elif trend_alignment and short_strength < 0:
        return {
            "type": SignalType.STRONG_SELL.value if overall_strength > 0.8 else SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Trend",
            "message": f"Нисходящий тренд (сила: {overall_strength:.2f})",
            "strength": min(95, int(overall_strength * 100)),
            "details": {
                "short_term_strength": float(short_strength),
                "medium_term_strength": float(medium_strength),
                "long_term_strength": float(long_strength),
                "alignment": True
            }
        }

    return None


async def generate_trading_signals(market_data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
    if market_data is None or len(market_data) < MIN_DATA_POINTS:
        return {
            "signals": [],
            "overall_signal": SignalType.NEUTRAL.value,
            "confidence": SignalConfidence.LOW.value,
            "confidence_score": 0.0,
            "timestamp": datetime.now().isoformat()
        }

    cache_key = signal_cache.get_key('generate_signals', symbol, market_data.values.tobytes())
    cached = signal_cache.get(cache_key)
    if cached:
        return cached

    try:
        prices  = market_data['close'].values
        highs   = market_data['high'].values
        lows    = market_data['low'].values
        volumes = market_data['volume'].values if 'volume' in market_data.columns else None

        regime  = determine_market_regime(prices, volumes)
        weights = get_regime_weights(regime, symbol)

        asset_profile = calculate_asset_profile(
            symbol=symbol, highs=highs, lows=lows, prices=prices, volumes=volumes
        )
        multiplier = adaptive_supertrend_multiplier(asset_profile)

        ti = TechnicalIndicators(market_data)
        indicators_tasks = [
            asyncio.to_thread(ti.ema, 9),
            asyncio.to_thread(ti.ema, 21),
            asyncio.to_thread(ti.macd),
            asyncio.to_thread(ti.rsi),
            asyncio.to_thread(ti.bollinger_bands),
            asyncio.to_thread(ti.supertrend, 10, multiplier),
        ]
        ema9, ema21, macd_data, rsi, bb_data, supertrend_data = await asyncio.gather(*indicators_tasks)

        all_signals: List[Dict[str, Any]] = []

        # EMA
        if ema9 is not None and ema21 is not None:
            ema_signal = analyze_ema_signals(ema9, ema21, prices)
            if ema_signal:
                ema_signal['weight'] = weights.ema_weight
                all_signals.append(ema_signal)

        # MACD (dict|tuple), confirmation_bars=1
        if macd_data is not None:
            if isinstance(macd_data, dict) and 'macd' in macd_data and 'signal' in macd_data:
                macd_line, macd_sig = macd_data['macd'], macd_data['signal']
            elif isinstance(macd_data, (tuple, list)) and len(macd_data) >= 2:
                macd_line, macd_sig = macd_data[0], macd_data[1]
            else:
                macd_line = macd_sig = None
            macd_sig_obj = analyze_macd_signals(macd_line, macd_sig, prices, confirmation_bars=1) \
                           if (macd_line is not None and macd_sig is not None) else None
            if macd_sig_obj:
                macd_sig_obj['weight'] = weights.macd_weight
                all_signals.append(macd_sig_obj)

        # RSI
        if rsi is not None:
            for s in analyze_rsi_signals(rsi, prices, asset_profile, regime):
                s['weight'] = weights.rsi_weight
                all_signals.append(s)

        # Bollinger (dict|tuple)
        if bb_data is not None:
            if isinstance(bb_data, dict) and all(k in bb_data for k in ('upper', 'middle', 'lower')):
                bb_upper, bb_middle, bb_lower = bb_data['upper'], bb_data['middle'], bb_data['lower']
            elif isinstance(bb_data, (tuple, list)) and len(bb_data) >= 3:
                bb_upper, bb_middle, bb_lower = bb_data[0], bb_data[1], bb_data[2]
            else:
                bb_upper = bb_middle = bb_lower = None
            if bb_upper is not None and bb_lower is not None and bb_middle is not None:
                for s in analyze_bollinger_bands_signals(prices, bb_upper, bb_lower, bb_middle):
                    s['weight'] = weights.bollinger_weight
                    all_signals.append(s)

        # Supertrend (dict|tuple)
        if supertrend_data is not None:
            if isinstance(supertrend_data, dict) and all(k in supertrend_data for k in ('supertrend', 'direction')):
                st_line, st_dir = supertrend_data['supertrend'], supertrend_data['direction']
            elif isinstance(supertrend_data, (tuple, list)) and len(supertrend_data) >= 2:
                st_line, st_dir = supertrend_data[0], supertrend_data[1]
            else:
                st_line = st_dir = None

            if st_line is not None and st_dir is not None and len(st_dir) > 1:
                prev_dir, curr_dir = st_dir[-2], st_dir[-1]
                if prev_dir == -1 and curr_dir == 1:
                    all_signals.append({
                        "type": SignalType.BUY.value,
                        "confidence": SignalConfidence.HIGH.value,
                        "indicator": "Supertrend",
                        "message": f"Разворот Supertrend вверх (множитель {multiplier:.2f})",
                        "strength": 80,
                        "weight": weights.supertrend_weight,
                        "details": {"supertrend_value": float(st_line[-1])}
                    })
                elif prev_dir == 1 and curr_dir == -1:
                    all_signals.append({
                        "type": SignalType.SELL.value,
                        "confidence": SignalConfidence.HIGH.value,
                        "indicator": "Supertrend",
                        "message": f"Разворот Supertrend вниз (множитель {multiplier:.2f})",
                        "strength": 80,
                        "weight": weights.supertrend_weight,
                        "details": {"supertrend_value": float(st_line[-1])}
                    })

        # Объём
        if volumes is not None:
            vol_sig = analyze_volume_signals(prices, volumes)
            if vol_sig:
                vol_sig['weight'] = weights.volume_weight
                all_signals.append(vol_sig)

        # Тренд
        trend_sig = analyze_trend_signals(prices)
        if trend_sig:
            trend_sig['weight'] = weights.trend_weight
            all_signals.append(trend_sig)

        # Агрегация
        buy_strength = sell_strength = total_weight = 0.0
        for s in all_signals:
            w = float(s.get('weight', 1.0))
            st = float(s.get('strength', 50))
            if s['type'] in (SignalType.BUY.value, SignalType.STRONG_BUY.value):
                buy_strength += st * w
            elif s['type'] in (SignalType.SELL.value, SignalType.STRONG_SELL.value):
                sell_strength += st * w
            total_weight += w

        if total_weight > 0:
            buy_strength /= total_weight
            sell_strength /= total_weight

        net_strength = buy_strength - sell_strength
        denom = max(1.0, buy_strength + sell_strength)
        confidence_score = min(1.0, abs(net_strength) / denom)

        if net_strength > 20:
            overall = SignalType.STRONG_BUY.value
            conf = SignalConfidence.VERY_HIGH.value if confidence_score > 0.6 else SignalConfidence.HIGH.value
        elif net_strength > 10:
            overall = SignalType.BUY.value
            conf = SignalConfidence.HIGH.value if confidence_score > 0.5 else SignalConfidence.MEDIUM.value
        elif net_strength < -20:
            overall = SignalType.STRONG_SELL.value
            conf = SignalConfidence.VERY_HIGH.value if confidence_score > 0.6 else SignalConfidence.HIGH.value
        elif net_strength < -10:
            overall = SignalType.SELL.value
            conf = SignalConfidence.HIGH.value if confidence_score > 0.5 else SignalConfidence.MEDIUM.value
        else:
            overall = SignalType.NEUTRAL.value
            conf = SignalConfidence.LOW.value

        result = {
            "signals": all_signals,
            "overall_signal": overall,
            "confidence": conf,
            "confidence_score": float(confidence_score),
            "market_regime": regime.value,
            "asset_profile": asset_profile.to_dict() if hasattr(asset_profile, 'to_dict') else {},
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }
        signal_cache.set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Ошибка генерации сигналов для {symbol}: {e}", exc_info=True)
        return {
            "signals": [],
            "overall_signal": SignalType.NEUTRAL.value,
            "confidence": SignalConfidence.LOW.value,
            "confidence_score": 0.0,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }



async def generate_adaptive_stop_loss_take_profit(market_data: pd.DataFrame,
                                                  entry_price: float,
                                                  position_type: str,
                                                  symbol: str = None) -> Dict[str, Any]:
    """
    Генерация адаптивных уровней стоп-лосса и тейк-профита
    """
    if market_data is None or len(market_data) < MIN_DATA_POINTS:
        return {
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "timestamp": datetime.now().isoformat()
        }

    try:
        prices = market_data['close'].values
        highs = market_data['high'].values
        lows = market_data['low'].values
        volumes = market_data['volume'].values if 'volume' in market_data.columns else None

        # Расчет профиля актива (исправленный вызов)
        asset_profile = calculate_asset_profile(
            symbol=symbol,
            highs=highs,
            lows=lows,
            prices=prices,
            volumes=volumes
        )

        # Адаптивные уровни стоп-лосса и тейк-профита
        stop_loss, take_profit, risk_reward_ratio = adaptive_stop_loss_take_profit(
            asset_profile, entry_price, position_type
        )

        return {
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "take_profit": float(take_profit) if take_profit is not None else None,
            "risk_reward_ratio": float(risk_reward_ratio) if risk_reward_ratio is not None else None,
            "asset_profile": asset_profile.to_dict() if hasattr(asset_profile, 'to_dict') else {},
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }

    except Exception as e:
        logger.error(f"Ошибка генерации стоп-лосса/тейк-профита для {symbol}: {str(e)}", exc_info=True)
        return {
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }