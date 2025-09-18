"""
АДАПТИВНЫЙ МОДУЛЬ ДЛЯ ТЕХНИЧЕСКОГО АНАЛИЗА
Динамическая подстройка параметров под характеристики актива
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from numba import njit


@dataclass
class AssetProfile:
    """Профиль характеристик торгового актива"""
    symbol: str
    volatility: float  # Средняя волатильность (ATR/Close в %)
    liquidity: float  # Относительная ликвидность
    trend_strength: float  # Сила тренда (0-1)
    noise_level: float  # Уровень шума (0-1)


def calculate_asset_profile(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray, volumes: np.ndarray) -> Tuple[float, float, float, float]:
    """Расчет характеристик актива для адаптивной настройки"""
    n = len(closes)
    if n < 20:
        return 0.01, 1.0, 0.5, 0.5

    # Волатильность (стандартное отклонение доходностей)
    returns = np.diff(closes) / closes[:-1]
    volatility = float(np.std(returns)) * 100 if len(returns) > 1 else 1.0

    # Ликвидность (нормированный объем)
    avg_volume = float(np.mean(volumes[-50:])) if n >= 50 else 1.0
    liquidity = min(1.0, avg_volume / 1e6)  # Нормируем к 1M

    # Сила тренда (наклон линейного тренда)
    if n >= 20:
        x = np.arange(len(closes[-20:]))
        y = closes[-20:]
        slope = float(np.polyfit(x, y, 1)[0]) / np.mean(y) * 100
        trend_strength = min(1.0, abs(slope) / 10.0)
    else:
        trend_strength = 0.5

    # Уровень шума (отношение диапазона к Close)
    daily_ranges = (highs - lows) / closes
    noise_level = float(np.mean(daily_ranges[-20:])) if n >= 20 else 0.02

    return volatility, liquidity, trend_strength, noise_level


def adaptive_rsi_levels(volatility: float = 0.02,
                        trend_strength: float = 0.5) -> Tuple[float, float]:
    """Адаптивные уровни перекупленности/перепроданности для RSI"""
    # Базовые уровни
    overbought = 70.0
    oversold = 30.0

    # Корректировка по волатильности
    if volatility > 3.0:  # Высокая волатильность (>3%)
        overbought += 5.0
        oversold -= 5.0
    elif volatility < 1.0:  # Низкая волатильность (<1%)
        overbought -= 5.0
        oversold += 5.0

    # Корректировка по силе тренда
    if trend_strength > 0.7:  # Сильный тренд
        overbought += 3.0
        oversold -= 3.0

    return min(85.0, max(55.0, overbought)), min(45.0, max(15.0, oversold))


def adaptive_supertrend_multiplier(volatility: float) -> float:
    """Адаптивный мультипликатор для SuperTrend"""
    # Базовый множитель
    multiplier = 3.0

    # Корректировка по волатильности
    if volatility > 3.0:
        multiplier += 1.0
    elif volatility < 1.0:
        multiplier -= 1.0

    return max(2.0, min(5.0, multiplier))


def adaptive_stop_loss_take_profit(current_price: float,
                                   volatility: float,
                                   trend_direction: int = 1) -> Tuple[float, float]:
    """Адаптивные уровни стоп-лосса и тейк-профита"""
    # Базовые расстояния в %
    stop_loss_pct = 2.0
    take_profit_pct = 4.0

    # Корректировка по волатильности
    stop_loss_pct += volatility * 0.5
    take_profit_pct += volatility * 1.0

    # Расчет абсолютных значений
    if trend_direction > 0:  # Бычий тренд
        stop_loss = current_price * (1 - stop_loss_pct / 100)
        take_profit = current_price * (1 + take_profit_pct / 100)
    else:  # Медвежий тренд
        stop_loss = current_price * (1 + stop_loss_pct / 100)
        take_profit = current_price * (1 - take_profit_pct / 100)

    return stop_loss, take_profit


class AdaptiveTechnicalAnalysis:
    """Класс для адаптивного технического анализа"""

    def __init__(self, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, volumes: np.ndarray):
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes

        # Расчет профиля актива
        self.volatility, self.liquidity, self.trend_strength, self.noise_level = \
            calculate_asset_profile(highs, lows, closes, volumes)

    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """Получение всех адаптивных параметров"""
        rsi_overbought, rsi_oversold = adaptive_rsi_levels(
            self.volatility, self.trend_strength
        )

        supertrend_multiplier = adaptive_supertrend_multiplier(self.volatility)

        return {
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'supertrend_multiplier': supertrend_multiplier,
            'volatility': self.volatility,
            'liquidity': self.liquidity,
            'trend_strength': self.trend_strength,
            'noise_level': self.noise_level
        }