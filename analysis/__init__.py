"""
Пакет технического анализа
"""

from typing import List, Dict, Any  # ← ДОБАВЬ ЭТОТ ИМПОРТ

from .analysis_engine import detect_signals_sync, calculate_indicators_sync, analysis_engine
from . import indicators
from . import signals

# Экспорт основных функций для обратной совместимости
def detect_signals(closes: List[float], strategy: str = 'classic') -> Dict[str, Any]:
    """Обнаружение сигналов (совместимость со старым кодом)"""
    return detect_signals_sync(closes, strategy)

# Реэкспорт функций
__all__ = [
    'detect_signals_sync',
    'calculate_indicators_sync', 
    'analysis_engine',
    'indicators',
    'signals',
    'detect_signals'  # для обратной совместимости
]