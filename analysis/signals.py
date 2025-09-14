"""
Минимальный модуль сигналов для обратной совместимости
Основной функционал перенесен в analysis_engine.py
"""

from typing import List, Dict, Any

def detect_patterns(prices: List[float], pattern_type: str = "all") -> List[Dict]:
    """Обнаружение паттернов (обертка для совместимости)"""
    from .analysis_engine import detect_signals_sync
    result = detect_signals_sync(prices)
    return result.get('alerts', [])

def find_support_resistance(prices: List[float], window: int = 20) -> Dict[str, List[float]]:
    """Поиск уровней поддержки и сопротивления"""
    if not prices:
        return {'support': [], 'resistance': []}

    # Упрощенный расчет уровней
    min_price = min(prices[-window:]) if len(prices) > window else min(prices)
    max_price = max(prices[-window:]) if len(prices) > window else max(prices)

    return {
        'support': [min_price * 0.98, min_price * 0.99],
        'resistance': [max_price * 1.01, max_price * 1.02]
    }

def calculate_trend_strength(prices: List[float]) -> float:
    """Расчет силы тренда"""
    if len(prices) < 10:
        return 0.0

    # Простой расчет силы тренда
    recent_prices = prices[-10:]
    price_change = recent_prices[-1] - recent_prices[0]
    volatility = max(recent_prices) - min(recent_prices)

    if volatility == 0:
        return 0.0

    return abs(price_change) / volatility

def generate_trading_signals(prices: List[float], strategy: str = "classic") -> Dict[str, Any]:
    """Генерация торговых сигналов (обертка для совместимости)"""
    from .analysis_engine import detect_signals_sync
    return detect_signals_sync(prices, strategy)

# Простые функции для базового анализа
def is_oversold(rsi_value: float, threshold: float = 30) -> bool:
    """Проверка перепроданности"""
    return rsi_value < threshold

def is_overbought(rsi_value: float, threshold: float = 70) -> bool:
    """Проверка перекупленности"""
    return rsi_value > threshold

def has_macd_crossover(macd_line: List[float], signal_line: List[float]) -> bool:
    """Проверка пересечения MACD"""
    if len(macd_line) < 2 or len(signal_line) < 2:
        return False

    return (macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]) or \
           (macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1])