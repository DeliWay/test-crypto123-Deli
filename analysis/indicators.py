"""
Минимальный модуль индикаторов для обратной совместимости
Основной функционал перенесен в analysis_engine.py
"""

import numpy as np
from typing import List, Optional, Dict, Any

# Добавь этот импорт в начало analysis_engine.py
from typing import List, Dict, Optional, Any

def calculate_ema(prices: List[float], period: int = 9) -> List[float]:
    """Расчет EMA (обертка для совместимости)"""
    from .analysis_engine import calculate_indicators_sync
    indicators = calculate_indicators_sync(prices)
    return indicators.get(f'ema{period}', []) if period in [9, 21] else []

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Расчет RSI (обертка для совместимости)"""
    from .analysis_engine import calculate_indicators_sync
    indicators = calculate_indicators_sync(prices)
    return indicators.get('rsi', [])

def calculate_macd(prices: List[float]) -> Dict[str, List[float]]:
    """Расчет MACD (обертка для совместимости)"""
    from .analysis_engine import calculate_indicators_sync
    indicators = calculate_indicators_sync(prices)
    return {
        'macd': indicators.get('macd', []),
        'signal': indicators.get('macd_signal', []),
        'histogram': indicators.get('macd_hist', [])
    }

def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, List[float]]:
    """Расчет Bollinger Bands (обертка для совместимости)"""
    from .analysis_engine import calculate_indicators_sync
    indicators = calculate_indicators_sync(prices)
    return {
        'upper': indicators.get('bb_upper', []),
        'middle': indicators.get('bb_mid', []),
        'lower': indicators.get('bb_lower', [])
    }

def calculate_stochastic(high: List[float], low: List[float], close: List[float]) -> Dict[str, List[float]]:
    """Расчет Stochastic (обертка для совместимости)"""
    from .analysis_engine import calculate_indicators_sync
    indicators = calculate_indicators_sync(close)
    return {
        'k': indicators.get('stoch_k', []),
        'd': indicators.get('stoch_d', [])
    }

# Быстрые версии для отдельных расчетов
def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Быстрый расчет EMA"""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

    return ema

def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Быстрый расчет RSI"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]

    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100 - 100 / (1 + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]

        if delta > 0:
            up_val = delta
            down_val = 0
        else:
            up_val = 0
            down_val = -delta

        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period

        rs = up / down if down != 0 else 0
        rsi[i] = 100 - 100 / (1 + rs)

    return rsi