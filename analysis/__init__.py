# analysis/__init__.py
"""
ULTRA-PERFORMANCE TECHNICAL ANALYSIS MODULE
Унифицированные синхронные и асинхронные интерфейсы
Реальные рыночные данные без заглушек
Многократное улучшение производительности и точности
"""

from .analysis_engine import (
    # Основные классы
    UltraAnalysisEngine,
    analysis_engine,

    # Асинхронные функции
    detect_signals,
    calculate_indicators,
    analyze_symbol,
    detect_patterns,
    calculate_profit_potential,

    # Синхронные функции
    detect_signals_sync,
    calculate_indicators_sync,
    analyze_symbol_sync,
    detect_patterns_sync,
    calculate_profit_potential_sync,

    # Утилиты
    init_analysis_engine,
    close_analysis_engine,

    # Константы
    AnalysisStrategy,
    SignalStrength
)

from .indicators.indicators import logger
from . import signals

__all__ = [
    'UltraAnalysisEngine',
    'analysis_engine',
    'detect_signals',
    'calculate_indicators',
    'analyze_symbol',
    'detect_patterns',
    'calculate_profit_potential',
    'detect_signals_sync',
    'calculate_indicators_sync',
    'analyze_symbol_sync',
    'detect_patterns_sync',
    'calculate_profit_potential_sync',
    'init_analysis_engine',
    'close_analysis_engine',
    'AnalysisStrategy',
    'SignalStrength',
    'indicators',
    'signals'
]