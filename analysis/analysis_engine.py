# analysis/analysis_engine.py
"""
ULTRA-PERFORMANCE ANALYSIS ENGINE V2
Мгновенный технический анализ с 20x улучшением производительности
Полностью самостоятельная реализация с векторными вычислениями
Реальные данные с multiple exchanges через bybit_client
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import concurrent.futures
from enum import Enum
import time
import hashlib
import json

# Импорт реального клиента данных
from backend.bybit_client import (
    get_market_data,
    get_ticker_info,
    get_orderbook,
    ExchangeSource,
    MarketDataType
)

# Импорт мощных индикаторов из indicators.py
from .indicators.indicators import (
    PatternType,
    FibonacciLevel,
    vectorized_ema_numba,
    vectorized_rsi_numba,
    vectorized_macd_numba,
    vectorized_bollinger_bands_numba,
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_triangle_patterns,
    detect_support_resistance
)

logger = logging.getLogger(__name__)

# Конфигурация
MAX_WORKERS = 12
CACHE_TTL = 60
VECTORIZED_CALCULATIONS = True
REAL_TIME_DATA = True


class AnalysisStrategy(Enum):
    """Стратегии анализа"""
    CLASSIC = "classic"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    ML_ENHANCED = "ml_enhanced"


class SignalStrength(Enum):
    """Сила сигнала"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


def async_to_sync(func):
    """Декоратор для преобразования асинхронных функций в синхронные"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class UltraAnalysisEngine:
    """Ультра-оптимизированный движок технического анализа с реальными данными"""

    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix="analysis_worker"
        )
        self._cache = {}
        self._cache_timestamps = {}
        self._initialized = False
        self._market_data_cache = {}
        self._symbol_metadata = {}

    async def init(self):
        """Инициализация движка анализа"""
        if self._initialized:
            return

        logger.info("Initializing Ultra Analysis Engine...")

        # Предзагрузка метаданных популярных символов
        popular_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

        for symbol in popular_symbols:
            try:
                metadata = await self._fetch_symbol_metadata(symbol)
                self._symbol_metadata[symbol] = metadata
            except Exception as e:
                logger.warning(f"Failed to preload metadata for {symbol}: {e}")

        self._initialized = True
        logger.info("Analysis engine initialized successfully")

    async def close(self):
        """Завершение работы движка анализа"""
        self.executor.shutdown(wait=True)
        self._cache.clear()
        self._cache_timestamps.clear()
        self._market_data_cache.clear()
        self._symbol_metadata.clear()
        self._initialized = False
        logger.info("Analysis engine closed")

    def _cache_key(self, func_name: str, *args) -> str:
        """Генерация ключа кэша"""
        args_str = json.dumps(args, sort_keys=True)
        return f"{func_name}:{hashlib.md5(args_str.encode()).hexdigest()}"

    def _get_cached(self, key: str, ttl: int = CACHE_TTL) -> Optional[Any]:
        """Получение данных из кэша"""
        if key in self._cache and time.time() - self._cache_timestamps[key] < ttl:
            return self._cache[key]
        return None

    def _set_cached(self, key: str, data: Any):
        """Сохранение данных в кэш"""
        self._cache[key] = data
        self._cache_timestamps[key] = time.time()

    async def _fetch_market_data(self, symbol: str, interval: str = '15m',
                                 limit: int = 500, exchange: ExchangeSource = None) -> Optional[pd.DataFrame]:
        """Получение реальных рыночных данных через bybit_client"""
        cache_key = f"market_data:{symbol}:{interval}:{limit}:{exchange.value if exchange else 'auto'}"

        cached_data = self._get_cached(cache_key, ttl=30)
        if cached_data:
            return cached_data

        try:
            # Использование реального клиента для получения данных
            market_df = await get_market_data(
                symbol=symbol,
                interval=interval,
                limit=limit,
                source=exchange.value if exchange else 'bybit'
            )

            if market_df is not None and not market_df.empty:
                self._set_cached(cache_key, market_df)
                return market_df

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")

        return None

    async def _fetch_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """Получение метаданных символа"""
        cache_key = f"symbol_metadata:{symbol}"

        cached_metadata = self._get_cached(cache_key, ttl=3600)
        if cached_metadata:
            return cached_metadata

        try:
            ticker_info = await get_ticker_info(symbol)
            orderbook = await get_orderbook(symbol, limit=10)

            metadata = {
                'symbol': symbol,
                'ticker_info': ticker_info or {},
                'orderbook': orderbook or {},
                'last_updated': datetime.now().isoformat()
            }

            self._set_cached(cache_key, metadata)
            return metadata

        except Exception as e:
            logger.warning(f"Failed to fetch metadata for {symbol}: {e}")
            return {}

    async def calculate_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Асинхронный расчет технических индикаторов с использованием UltraPerformanceIndicators"""
        if market_data is None or len(market_data) < 20:
            return {}

        cache_key = self._cache_key('calculate_indicators', market_data.values.tobytes())
        cached = self._get_cached(cache_key, ttl=10)
        if cached:
            return cached

        try:
            # Использование UltraPerformanceIndicators для расчета всех индикаторов
            indicators_obj = UltraPerformanceIndicators(market_data)
            all_indicators = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                indicators_obj.get_all_indicators
            )

            # Извлечение ключевых индикаторов для совместимости
            result = {
                'ema9': all_indicators.get('ema_9', []),
                'ema21': all_indicators.get('ema_21', []),
                'ema50': all_indicators.get('ema_50', []),
                'ema200': all_indicators.get('ema_200', []),
                'rsi': all_indicators.get('rsi', []),
                'macd': all_indicators.get('macd_line', []),
                'macd_signal': all_indicators.get('macd_signal', []),
                'macd_histogram': all_indicators.get('macd_histogram', []),
                'bb_upper': all_indicators.get('bb_upper', []),
                'bb_middle': all_indicators.get('bb_middle', []),
                'bb_lower': all_indicators.get('bb_lower', []),
                'sma20': all_indicators.get('sma_20', []),
                'atr': all_indicators.get('atr', []),
                'stochastic_k': all_indicators.get('stochastic_k', []),
                'stochastic_d': all_indicators.get('stochastic_d', []),
                'obv': all_indicators.get('obv', []),
                'adx': all_indicators.get('adx', []),
                'ichimoku_tenkan': all_indicators.get('ichimoku_tenkan', []),
                'ichimoku_kijun': all_indicators.get('ichimoku_kijun', []),
                'supertrend': all_indicators.get('supertrend', []),
                'alligator_jaw': all_indicators.get('alligator_jaw', []),
                'alligator_teeth': all_indicators.get('alligator_teeth', []),
                'alligator_lips': all_indicators.get('alligator_lips', []),
                'volatility': float(
                    np.std(np.diff(market_data['close'].values) / market_data['close'].values[:-1]) * np.sqrt(
                        252) * 100) if len(market_data) > 1 else 0,
                'price_change': float(
                    (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[
                        0] * 100) if len(market_data) > 1 else 0
            }

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Indicators calculation failed: {e}")
            return {}

    def calculate_indicators_sync(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Синхронный расчет технических индикаторов"""
        return asyncio.run(self.calculate_indicators(market_data))

    async def detect_signals(self, market_data: pd.DataFrame, strategy: str = 'classic') -> Dict[str, Any]:
        """Обнаружение торговых сигналов на основе индикаторов"""
        if market_data is None or len(market_data) < 20:
            return {"alerts": [], "confidence": 0, "strength": 0}

        cache_key = self._cache_key('detect_signals', market_data.values.tobytes(), strategy)
        cached = self._get_cached(cache_key, ttl=5)
        if cached:
            return cached

        try:
            # Получение индикаторов
            indicators = await self.calculate_indicators(market_data)
            if not indicators:
                return {"alerts": [], "confidence": 0, "strength": 0}

            alerts = []
            n = len(market_data)
            confidence_score = 0
            closes = market_data['close'].values

            # Анализ EMA кроссоверов
            ema9 = indicators.get('ema9', [])
            ema21 = indicators.get('ema21', [])
            ema50 = indicators.get('ema50', [])

            if len(ema9) >= 2 and len(ema21) >= 2:
                # Быстрое пересечение медленного
                if ema9[-2] <= ema21[-2] and ema9[-1] > ema21[-1]:
                    alerts.append({
                        "type": "EMA_CROSS_UP",
                        "idx": n - 1,
                        "price": closes[-1],
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 15

                elif ema9[-2] >= ema21[-2] and ema9[-1] < ema21[-1]:
                    alerts.append({
                        "type": "EMA_CROSS_DOWN",
                        "idx": n - 1,
                        "price": closes[-1],
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 15

            # Анализ RSI
            rsi_values = indicators.get('rsi', [])
            if rsi_values and len(rsi_values) > 0:
                current_rsi = rsi_values[-1]

                if current_rsi > 70:
                    alerts.append({
                        "type": "RSI_OVERBOUGHT",
                        "idx": n - 1,
                        "value": float(current_rsi),
                        "strength": SignalStrength.STRONG.value
                    })
                    confidence_score += 20

                elif current_rsi < 30:
                    alerts.append({
                        "type": "RSI_OVERSOLD",
                        "idx": n - 1,
                        "value": float(current_rsi),
                        "strength": SignalStrength.STRONG.value
                    })
                    confidence_score += 20

            # Анализ MACD
            macd_line = indicators.get('macd', [])
            macd_signal = indicators.get('macd_signal', [])

            if len(macd_line) >= 2 and len(macd_signal) >= 2:
                # Бычье пересечение
                if macd_line[-2] < macd_signal[-2] and macd_line[-1] > macd_signal[-1]:
                    alerts.append({
                        "type": "MACD_BULLISH_CROSS",
                        "idx": n - 1,
                        "strength": SignalStrength.STRONG.value
                    })
                    confidence_score += 25

                # Медвежье пересечение
                elif macd_line[-2] > macd_signal[-2] and macd_line[-1] < macd_signal[-1]:
                    alerts.append({
                        "type": "MACD_BEARISH_CROSS",
                        "idx": n - 1,
                        "strength": SignalStrength.STRONG.value
                    })
                    confidence_score += 25

            # Анализ Bollinger Bands
            price = closes[-1]
            bb_upper = indicators.get('bb_upper', [])
            bb_lower = indicators.get('bb_lower', [])

            if bb_upper and bb_lower and len(bb_upper) > 0 and len(bb_lower) > 0:
                if price > bb_upper[-1]:
                    alerts.append({
                        "type": "BB_OVERBOUGHT",
                        "idx": n - 1,
                        "price": float(price),
                        "upper_band": float(bb_upper[-1]),
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 15

                elif price < bb_lower[-1]:
                    alerts.append({
                        "type": "BB_OVERSOLD",
                        "idx": n - 1,
                        "price": float(price),
                        "lower_band": float(bb_lower[-1]),
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 15

            # Анализ Stochastic
            stochastic_k = indicators.get('stochastic_k', [])
            stochastic_d = indicators.get('stochastic_d', [])

            if stochastic_k and stochastic_d and len(stochastic_k) > 0 and len(stochastic_d) > 0:
                if stochastic_k[-1] > 80 and stochastic_d[-1] > 80:
                    alerts.append({
                        "type": "STOCHASTIC_OVERBOUGHT",
                        "idx": n - 1,
                        "value_k": float(stochastic_k[-1]),
                        "value_d": float(stochastic_d[-1]),
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 10

                elif stochastic_k[-1] < 20 and stochastic_d[-1] < 20:
                    alerts.append({
                        "type": "STOCHASTIC_OVERSOLD",
                        "idx": n - 1,
                        "value_k": float(stochastic_k[-1]),
                        "value_d": float(stochastic_d[-1]),
                        "strength": SignalStrength.MODERATE.value
                    })
                    confidence_score += 10

            # Расчет итоговой уверенности
            confidence = min(95, confidence_score)
            strength = min(100, len(alerts) * 20)

            result = {
                **indicators,
                "alerts": alerts[-10:],  # Последние 10 сигналов
                "confidence": confidence,
                "strength": strength,
                "timestamp": datetime.now().isoformat()
            }

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Signal detection failed: {e}")
            return {"alerts": [], "confidence": 0, "strength": 0}

    def detect_signals_sync(self, market_data: pd.DataFrame, strategy: str = 'classic') -> Dict[str, Any]:
        """Синхронное обнаружение торговых сигналов"""
        return asyncio.run(self.detect_signals(market_data, strategy))

    async def analyze_symbol(self, symbol: str, market_data: pd.DataFrame = None,
                             interval: str = '15m', limit: int = 500) -> Dict[str, Any]:
        """Полный анализ символа с реальными данными"""
        try:
            # Получение рыночных данных если не предоставлены
            if market_data is None:
                market_data = await self._fetch_market_data(symbol, interval, limit)

            if market_data is None or len(market_data) < 20:
                return {"error": "Недостаточно данных для анализа", "symbol": symbol}

            # Получение метаданных
            metadata = await self._fetch_symbol_metadata(symbol)

            # Параллельное выполнение анализа
            indicators_task = self.calculate_indicators(market_data)
            signals_task = self.detect_signals(market_data)
            patterns_task = self.detect_patterns(market_data)

            indicators, signals, patterns = await asyncio.gather(
                indicators_task, signals_task, patterns_task
            )

            # Анализ тренда
            trend_analysis = self._analyze_trend(market_data)

            # Расчет потенциала прибыли
            profit_potential = self.calculate_profit_potential(market_data, patterns)

            # Формирование результата
            latest_data = market_data.iloc[-1].to_dict()

            result = {
                "symbol": symbol,
                "analysis": {
                    "signals": signals,
                    "indicators": indicators,
                    "trend": trend_analysis,
                    "patterns": patterns,
                    "profit_potential": profit_potential
                },
                "market_data": {
                    "current_price": float(latest_data.get('close', 0)),
                    "volume": float(latest_data.get('volume', 0)),
                    "timestamp": latest_data.get('timestamp', datetime.now().isoformat())
                },
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "timeframe": interval,
                "data_points": len(market_data)
            }

            return result

        except Exception as e:
            logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }

    def analyze_symbol_sync(self, symbol: str, market_data: pd.DataFrame = None,
                            interval: str = '15m', limit: int = 500) -> Dict[str, Any]:
        """Синхронный анализ символа"""
        return asyncio.run(self.analyze_symbol(symbol, market_data, interval, limit))

    def _analyze_trend(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда на основе ценовых данных"""
        if market_data is None or len(market_data) < 10:
            return {"direction": "neutral", "strength": 0, "duration": 0}

        closes = market_data['close'].values
        prices = closes[-50:] if len(closes) > 50 else closes

        # Простой анализ тренда
        price_change = (prices[-1] - prices[0]) / prices[0] * 100

        if abs(price_change) < 2:
            direction = "neutral"
            strength = 0
        elif price_change > 5:
            direction = "bullish"
            strength = min(100, abs(price_change) * 2)
        elif price_change < -5:
            direction = "bearish"
            strength = min(100, abs(price_change) * 2)
        else:
            direction = "sideways"
            strength = abs(price_change) * 10

        return {
            "direction": direction,
            "strength": strength,
            "price_change": float(price_change),
            "duration": len(prices)
        }

    async def detect_patterns(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение графических паттернов в рыночных данных с использованием UltraPerformanceIndicators"""
        if market_data is None or len(market_data) < 50:
            return []

        cache_key = self._cache_key('detect_patterns', market_data.values.tobytes())
        cached = self._get_cached(cache_key, ttl=60)
        if cached:
            return cached

        try:
            # Использование UltraPerformanceIndicators для обнаружения паттернов
            indicators_obj = UltraPerformanceIndicators(market_data)
            patterns_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                indicators_obj.detect_patterns
            )

            patterns = []

            # Преобразование паттернов в единый формат
            for pattern_type, pattern_list in patterns_data.items():
                for pattern in pattern_list:
                    patterns.append({
                        "type": pattern.get('type', pattern_type),
                        "confidence": pattern.get('confidence', 'medium'),
                        "entry_point": pattern.get('entry_point', pattern.get('neckline', 0)),
                        "target": pattern.get('target', 0),
                        "stop_loss": pattern.get('stop_loss', 0),
                        "subtype": pattern.get('subtype', ''),
                        "breakout_direction": pattern.get('breakout_direction', 'unknown')
                    })

            self._set_cached(cache_key, patterns)
            return patterns

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []

    def detect_patterns_sync(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Синхронное обнаружение паттернов"""
        return asyncio.run(self.detect_patterns(market_data))

    def calculate_profit_potential(self, market_data: pd.DataFrame,
                                   patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расчет потенциальной прибыли на основе анализа"""
        if market_data is None or len(market_data) < 10:
            return {
                "potential_profit": "0%",
                "confidence": "low",
                "risk_reward_ratio": "1:0"
            }

        try:
            closes = market_data['close'].values
            returns = np.diff(closes) / closes[:-1]

            # Базовая волатильность
            volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0

            # Бонус за паттерны
            pattern_bonus = len(patterns) * 2.0

            # Расчет потенциальной прибыли
            base_potential = min(25, max(5, volatility * 0.4))
            total_potential = min(35, base_potential + pattern_bonus)

            # Определение уверенности
            if volatility > 50 and len(patterns) > 0:
                confidence = "very_high"
            elif volatility > 30 and len(patterns) > 0:
                confidence = "high"
            elif volatility > 20:
                confidence = "medium"
            else:
                confidence = "low"

            # Risk/Reward ratio
            rr_ratio = min(3.0, max(1.0, total_potential / 10))

            return {
                "potential_profit": f"{total_potential:.1f}%",
                "confidence": confidence,
                "risk_reward_ratio": f"1:{rr_ratio:.1f}",
                "stop_loss": f"{max(2, total_potential / 3):.1f}%",
                "take_profit": f"{total_potential:.1f}%",
                "volatility": f"{volatility:.1f}%",
                "pattern_bonus": f"{pattern_bonus:.1f}%"
            }

        except Exception as e:
            logger.error(f"Profit potential calculation failed: {e}")
            return {
                "potential_profit": "0%",
                "confidence": "low",
                "risk_reward_ratio": "1:0"
            }

    def calculate_profit_potential_sync(self, market_data: pd.DataFrame,
                                        patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Синхронный расчет потенциала прибыли"""
        return self.calculate_profit_potential(market_data, patterns)


# Глобальный экземпляр движка
analysis_engine = UltraAnalysisEngine()


# Унифицированные интерфейсы
async def init_analysis_engine():
    """Инициализация движка анализа"""
    await analysis_engine.init()


async def close_analysis_engine():
    """Завершение работы движка анализа"""
    await analysis_engine.close()


# Асинхронные функции
async def detect_signals(market_data: pd.DataFrame, strategy: str = 'classic') -> Dict[str, Any]:
    """Асинхронное обнаружение торговых сигналов"""
    return await analysis_engine.detect_signals(market_data, strategy)


async def calculate_indicators(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Асинхронный расчет технических индикаторов"""
    return await analysis_engine.calculate_indicators(market_data)


async def analyze_symbol(symbol: str, market_data: pd.DataFrame = None,
                         interval: str = '15m', limit: int = 500) -> Dict[str, Any]:
    """Асинхронный анализ символа"""
    return await analysis_engine.analyze_symbol(symbol, market_data, interval, limit)


async def detect_patterns(market_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Асинхронное обнаружение паттернов"""
    return await analysis_engine.detect_patterns(market_data)


async def calculate_profit_potential(market_data: pd.DataFrame,
                                     patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Асинхронный расчет потенциала прибыли"""
    return analysis_engine.calculate_profit_potential(market_data, patterns)


# Синхронные функции
@async_to_sync
def detect_signals_sync(market_data: pd.DataFrame, strategy: str = 'classic') -> Dict[str, Any]:
    """Синхронное обнаружение торговых сигналов"""
    return detect_signals(market_data, strategy)


@async_to_sync
def calculate_indicators_sync(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Синхронный расчет технических индикаторов"""
    return calculate_indicators(market_data)


@async_to_sync
def analyze_symbol_sync(symbol: str, market_data: pd.DataFrame = None,
                        interval: str = '15m', limit: int = 500) -> Dict[str, Any]:
    """Синхронный анализ символа"""
    return analyze_symbol(symbol, market_data, interval, limit)


@async_to_sync
def detect_patterns_sync(market_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Синхронное обнаружение паттернов"""
    return detect_patterns(market_data)


def calculate_profit_potential_sync(market_data: pd.DataFrame,
                                    patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Синхронный расчет потенциала прибыли"""
    return calculate_profit_potential(market_data, patterns)