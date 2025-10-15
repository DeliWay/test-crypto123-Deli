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

# Новый провайдер реальных данных
from backend.exchange_data import (
    get_candles,          # async (exchange: str, symbol: str, timeframe: str, limit: int)
    get_ticker,           # async (exchange: str, symbol: str)
    get_orderbook_data,   # async (exchange: str, symbol: str, limit: int)
    Exchange,             # enum ('binance' | 'bybit')
    TimeFrame             # enum ('1m','3m','5m','15m','30m','1h','2h','4h','6h','12h','1d','1w','1M')
)

# Импорт мощных индикаторов из indicators.py
from analysis.indicators.indicators import TechnicalIndicators

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

    def _normalize_exchange(self, exchange) -> str:
        """
        Превращает Exchange|str|None в строку для провайдера данных.
        По умолчанию используем 'bybit'.
        """
        try:
            # если это Enum Exchange — у него есть .value
            val = (exchange.value if hasattr(exchange, "value") else exchange) or "bybit"
        except Exception:
            val = "bybit"
        s = str(val).strip().lower()
        # разрешённые алиасы
        if s in ("binance", "bn", "bnb"):
            return "binance"
        if s in ("bybit", "bb"):
            return "bybit"
        return "bybit"

    def _normalize_interval(self, interval: str) -> str:
        """
        Приводим старые обозначения ('15', '60', '15min', ...) к формату провайдера ('15m','1h',...).
        """
        key = (interval or "").strip().lower()
        tf_alias = {
            "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
            "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
            "d": "1d", "w": "1w", "m": "1M",
            "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
            "1d": "1d", "1w": "1w", "1mth": "1M", "1mo": "1M"
        }
        return tf_alias.get(key, key or "15m")

    async def _fetch_market_data(self, symbol: str, interval: str = '15m',
                                 limit: int = 500, exchange: Optional[Exchange] = None) -> Optional[pd.DataFrame]:
        """
        Получение реальных рыночных данных через наш exchange_data
        """
        ex = self._normalize_exchange(exchange)
        tf = self._normalize_interval(interval)
        cache_key = f"market_data:{symbol}:{tf}:{limit}:{ex}"

        cached_data = self._get_cached(cache_key, ttl=30)
        if cached_data is not None:
            return cached_data

        try:
            df = await get_candles(ex, symbol, tf, limit)
            if df is not None and not df.empty:
                # df уже в формате: timestamp, open, high, low, close, volume, symbol, timeframe
                self._set_cached(cache_key, df)
                return df
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol} ({ex}, {tf}): {e}")

        return None


    async def _fetch_symbol_metadata(self, symbol: str, exchange: Optional[Exchange] = None) -> Dict[str, Any]:
        """
        Получение метаданных символа (тикер + ордербук) в унифицированный dict.
        """
        ex = self._normalize_exchange(exchange)
        cache_key = f"symbol_metadata:{ex}:{symbol}"

        cached_metadata = self._get_cached(cache_key, ttl=3600)
        if cached_metadata:
            return cached_metadata

        try:
            t = await get_ticker(ex, symbol)
            ob = await get_orderbook_data(ex, symbol, limit=10)

            ticker_info = {}
            if t:
                ticker_info = {
                    'symbol': t.symbol,
                    'price': float(t.price),
                    'volume': float(t.volume),
                    'change_24h': float(t.change_24h),
                    'change_percent_24h': float(t.change_percent_24h),
                    'high_24h': float(t.high_24h),
                    'low_24h': float(t.low_24h),
                    'timestamp': t.timestamp.isoformat()
                }

            orderbook = {}
            if ob:
                orderbook = {
                    'symbol': ob.symbol,
                    'bids': ob.bids,
                    'asks': ob.asks,
                    'timestamp': ob.timestamp.isoformat()
                }

            metadata = {
                'symbol': symbol,
                'exchange': ex,
                'ticker_info': ticker_info,
                'orderbook': orderbook,
                'last_updated': datetime.now().isoformat()
            }

            self._set_cached(cache_key, metadata)
            return metadata

        except Exception as e:
            logger.warning(f"Failed to fetch metadata for {symbol} on {ex}: {e}")
            return {
                'symbol': symbol,
                'exchange': ex,
                'ticker_info': {},
                'orderbook': {},
                'last_updated': datetime.now().isoformat()
            }

    # внутри UltraAnalysisEngine
    async def calculate_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Унифицированный расчёт индикаторов для OHLCV DataFrame.
        Поддерживает разные сигнатуры методов TechnicalIndicators:
          - вариант A: ti.sma(close, 20)
          - вариант B: ti.sma(20)      # берёт series из self.data внутри класса
        И альтернативные имена: bbands/bollinger_bands, stoch/stochastic.
        """
        import numpy as np
        import pandas as pd
        from inspect import signature

        if market_data is None or len(market_data) < 5:
            return {"ok": False, "error": "not_enough_data", "series": {}, "last": {}}

        # нормализуем колонки
        df = market_data.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                return {"ok": False, "error": f"missing_column:{c}", "series": {}, "last": {}}
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        n = len(df)
        if n < 5:
            return {"ok": False, "error": "not_enough_clean_rows", "series": {}, "last": {}}

        close = df["close"]

        # импорт индикаторов
        try:
            from analysis.indicators.indicators import TechnicalIndicators as _TI
        except Exception:
            from analysis.indicators.indicators import UltraPerformanceIndicators as _TI
        ti = _TI(df[["open", "high", "low", "close", "volume"]])

        # универсальный адаптер вызова: сначала пробуем с series, если TypeError — без series
        def _call(f, *args_with_series, fallback_args=()):
            try:
                return f(*args_with_series)
            except TypeError:
                return f(*fallback_args)
            except Exception:
                return None

        # helper: привести результат к pd.Series длины n
        def _as_series(x):
            import numpy as np
            import pandas as pd
            if x is None:
                return pd.Series([np.nan] * n, dtype="float64")
            if isinstance(x, pd.Series):
                s = x
            elif isinstance(x, (list, tuple, np.ndarray)):
                s = pd.Series(x, dtype="float64")
            else:
                return pd.Series([np.nan] * n, dtype="float64")
            # выровнять длину
            if len(s) != n:
                s = s.reindex(range(n))
            return s.astype("float64")

        # === SMA / EMA ===
        sma20 = sma50 = ema20 = ema50 = None
        if hasattr(ti, "sma"):
            sma20 = _call(ti.sma, close, 20, fallback_args=(20,))
            sma50 = _call(ti.sma, close, 50, fallback_args=(50,))
        if hasattr(ti, "ema"):
            ema20 = _call(ti.ema, close, 20, fallback_args=(20,))
            ema50 = _call(ti.ema, close, 50, fallback_args=(50,))
        sma20, sma50, ema20, ema50 = map(_as_series, (sma20, sma50, ema20, ema50))

        # === RSI ===
        rsi14 = None
        if hasattr(ti, "rsi"):
            # возможны сигнатуры: rsi(series, period) или rsi(period)
            rsi14 = _call(ti.rsi, close, 14, fallback_args=(14,))
        rsi14 = _as_series(rsi14)

        # === MACD ===  (ожидаем tuple(line, signal, hist))
        macd_line = macd_signal = macd_hist = None
        if hasattr(ti, "macd"):
            m = _call(ti.macd, close, 12, 26, 9, fallback_args=(12, 26, 9))
            if isinstance(m, (list, tuple)) and len(m) == 3:
                macd_line, macd_signal, macd_hist = m
        macd_line, macd_signal, macd_hist = map(_as_series, (macd_line, macd_signal, macd_hist))

        # === Bollinger Bands ===
        bb_upper = bb_middle = bb_lower = None
        if hasattr(ti, "bbands"):
            bb = _call(ti.bbands, close, 20, 2.0, fallback_args=(20, 2.0))
            if isinstance(bb, (list, tuple)) and len(bb) == 3:
                bb_upper, bb_middle, bb_lower = bb
        elif hasattr(ti, "bollinger_bands"):
            bb = _call(ti.bollinger_bands, close, 20, 2.0, fallback_args=(20, 2.0))
            if isinstance(bb, (list, tuple)) and len(bb) == 3:
                bb_upper, bb_middle, bb_lower = bb
        bb_upper, bb_middle, bb_lower = map(_as_series, (bb_upper, bb_middle, bb_lower))

        # === ATR ===
        atr14 = None
        if hasattr(ti, "atr"):
            # большинство реализаций atr(period) используют self.data внутри
            atr14 = _call(ti.atr, 14, fallback_args=(14,))
        atr14 = _as_series(atr14)

        # === Stochastic ===
        stoch_k = stoch_d = None
        if hasattr(ti, "stoch"):
            sd = _call(ti.stoch, 14, 3, fallback_args=(14, 3))
            if isinstance(sd, (list, tuple)) and len(sd) == 2:
                stoch_k, stoch_d = sd
        elif hasattr(ti, "stochastic"):
            sd = _call(ti.stochastic, 14, 3, fallback_args=(14, 3))
            if isinstance(sd, (list, tuple)) and len(sd) == 2:
                stoch_k, stoch_d = sd
        stoch_k, stoch_d = map(_as_series, (stoch_k, stoch_d))

        def _last(s):
            s = s[~s.isna()]
            return float(s.iloc[-1]) if len(s) else None

        series = {
            "sma20": sma20.tolist(), "sma50": sma50.tolist(),
            "ema20": ema20.tolist(), "ema50": ema50.tolist(),
            "rsi14": rsi14.tolist(),
            "macd_line": macd_line.tolist(), "macd_signal": macd_signal.tolist(), "macd_hist": macd_hist.tolist(),
            "bb_upper": bb_upper.tolist(), "bb_middle": bb_middle.tolist(), "bb_lower": bb_lower.tolist(),
            "atr14": atr14.tolist(),
            "stoch_k": stoch_k.tolist(), "stoch_d": stoch_d.tolist(),
        }
        last = {
            "close": float(df["close"].iloc[-1]),
            "sma20": _last(sma20), "sma50": _last(sma50),
            "ema20": _last(ema20), "ema50": _last(ema50),
            "rsi14": _last(rsi14),
            "macd_line": _last(macd_line), "macd_signal": _last(macd_signal), "macd_hist": _last(macd_hist),
            "bb_upper": _last(bb_upper), "bb_middle": _last(bb_middle), "bb_lower": _last(bb_lower),
            "atr14": _last(atr14),
            "stoch_k": _last(stoch_k), "stoch_d": _last(stoch_d),
        }

        return {
            "ok": True,
            "rows": int(n),
            "series": series,
            "last": last,
            "meta": {
                "sma_periods": [20, 50],
                "ema_periods": [20, 50],
                "rsi_period": 14,
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bb": {"period": 20, "n_std": 2.0},
                "atr_period": 14,
                "stoch": {"k": 14, "d": 3},
            },
        }

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

    # внутри класса UltraAnalysisEngine
    async def detect_patterns(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Детектит свечные паттерны и простые события рынка.
        Возвращает список dict: {
            'name': <str>,            # название паттерна
            'i': <int>,               # индекс свечи в df
            'ts': <str>,              # ISO-время если есть колонка timestamp
            'direction': 'bull'|'bear'|'neutral',
            'confidence': <float 0..1>,
            'extras': {...}           # доп. поля (уровни, значения индикаторов)
        }
        """
        import numpy as np
        import pandas as pd

        if market_data is None or len(market_data) < 20:
            return []

        # 1) нормализуем столбцы
        df = market_data.copy()
        df.columns = [str(c).lower() for c in df.columns]
        need = {'open', 'high', 'low', 'close'}
        if not need.issubset(df.columns):
            return []
        has_ts = 'timestamp' in df.columns
        if has_ts:
            # гарантируем строковый ISO (без таймзоны ок)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            except Exception:
                pass

        # 2) числовые типы
        for c in ['open', 'high', 'low', 'close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
        n = len(df)
        if n < 20:
            return []

        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values

        # 3) базовые метрики свечи
        body = np.abs(c - o)
        rng = (h - l).astype(float)
        rng[rng == 0] = np.nan  # чтобы не делить на 0
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        body_pct = body / rng  # доля тела от диапазона
        upper_pct = upper_shadow / rng
        lower_pct = lower_shadow / rng

        # тренд по SMA (для контекстных паттернов)
        def _sma(arr, p):
            s = pd.Series(arr, dtype=float)
            return s.rolling(p, min_periods=p).mean().to_numpy()

        sma20 = _sma(c, 20)
        sma50 = _sma(c, 50)
        slope20 = np.r_[np.nan, np.diff(sma20)]
        slope50 = np.r_[np.nan, np.diff(sma50)]

        def ts_at(i):
            if not has_ts:
                return None
            ts = df['timestamp'].iloc[i]
            try:
                return ts.isoformat()
            except Exception:
                return str(ts)

        events: List[Dict[str, Any]] = []

        def add(name, i, direction='neutral', confidence=0.5, **extras):
            events.append({
                'name': name,
                'i': int(i),
                'ts': ts_at(i),
                'direction': direction,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'extras': extras or {}
            })

        # 4) правила паттернов
        # Doji: маленькое тело, относительно широкий диапазон
        doji_mask = (body_pct <= 0.1) & np.isfinite(body_pct)
        for i in np.where(doji_mask)[0]:
            # «неопределённость», чуть больше уверенность при большой волатильности
            conf = float(min(1.0, 0.3 + (rng[i] / np.nanmedian(rng[-50:]) if np.isfinite(rng[i]) else 0) * 0.1))
            add('Doji', i, 'neutral', conf, body_pct=float(body_pct[i]))

        # Hammer (после снижения): длинная нижняя тень, короткая верхняя, маленькое тело
        downtrend_mask = (slope20 < 0) & (slope50 <= 0)
        hammer_mask = (lower_pct >= 0.6) & (upper_pct <= 0.2) & (body_pct <= 0.35)
        for i in np.where(hammer_mask)[0]:
            ctx = bool(i > 0 and downtrend_mask[i - 1])
            conf = 0.55 + (0.2 if ctx else 0.0) + float(min(0.15, lower_pct[i] * 0.15))
            add('Hammer', i, 'bull', conf, lower_pct=float(lower_pct[i]), body_pct=float(body_pct[i]), ctx=ctx)

        # Shooting Star (после роста): длинная верхняя тень, короткая нижняя, маленькое тело
        uptrend_mask = (slope20 > 0) & (slope50 >= 0)
        star_mask = (upper_pct >= 0.6) & (lower_pct <= 0.2) & (body_pct <= 0.35)
        for i in np.where(star_mask)[0]:
            ctx = bool(i > 0 and uptrend_mask[i - 1])
            conf = 0.55 + (0.2 if ctx else 0.0) + float(min(0.15, upper_pct[i] * 0.15))
            add('ShootingStar', i, 'bear', conf, upper_pct=float(upper_pct[i]), body_pct=float(body_pct[i]), ctx=ctx)

        # Bullish Engulfing: пред. медвежья, текущая бычья, тело поглощает
        for i in range(1, n):
            prev_bear = c[i - 1] < o[i - 1]
            curr_bull = c[i] > o[i]
            engulf = (c[i] >= o[i - 1]) and (o[i] <= c[i - 1]) and (body[i] > body[i - 1] * 1.02)
            if prev_bear and curr_bull and engulf:
                ctx = bool(i > 2 and (c[i - 1] < c[i - 2] < c[i - 3]))  # небольшая «лестница» вниз до паттерна
                conf = 0.6 + (0.1 if ctx else 0.0) + float(
                    min(0.1, (body[i] / (rng[i] if np.isfinite(rng[i]) else 1)) * 0.2))
                add('BullishEngulfing', i, 'bull', conf,
                    body_ratio=float((body[i] / rng[i]) if np.isfinite(rng[i]) else np.nan), ctx=ctx)

        # Bearish Engulfing: пред. бычья, текущая медвежья, тело поглощает
        for i in range(1, n):
            prev_bull = c[i - 1] > o[i - 1]
            curr_bear = c[i] < o[i]
            engulf = (c[i] <= o[i - 1]) and (o[i] >= c[i - 1]) and (body[i] > body[i - 1] * 1.02)
            if prev_bull and curr_bear and engulf:
                ctx = bool(i > 2 and (c[i - 1] > c[i - 2] > c[i - 3]))
                conf = 0.6 + (0.1 if ctx else 0.0) + float(
                    min(0.1, (body[i] / (rng[i] if np.isfinite(rng[i]) else 1)) * 0.2))
                add('BearishEngulfing', i, 'bear', conf,
                    body_ratio=float((body[i] / rng[i]) if np.isfinite(rng[i]) else np.nan), ctx=ctx)

        # Локальные развороты (swing highs / lows) по 2-свечной схеме
        for i in range(2, n - 2):
            # swing high: максимум выше соседей
            if h[i] > h[i - 1] and h[i] > h[i + 1] and c[i] < o[i]:  # медвежий пин на вершине
                add('SwingHigh', i, 'bear', 0.55, level=float(h[i]))
            # swing low: минимум ниже соседей
            if l[i] < l[i - 1] and l[i] < l[i + 1] and c[i] > o[i]:  # бычий пин на донышке
                add('SwingLow', i, 'bull', 0.55, level=float(l[i]))

        # Пробой диапазона (breakout) последних 20 свечей
        lookback = 20
        if n > lookback + 1:
            rolling_max = pd.Series(h).rolling(lookback, min_periods=lookback).max().to_numpy()
            rolling_min = pd.Series(l).rolling(lookback, min_periods=lookback).min().to_numpy()
            for i in range(lookback, n):
                if np.isfinite(rolling_max[i - 1]) and c[i] > rolling_max[i - 1]:
                    # бычий пробой
                    thrust = float((c[i] - rolling_max[i - 1]) / (
                        rng[i] if np.isfinite(rng[i]) and rng[i] > 0 else max(1e-9, abs(c[i] * 0.001))))
                    conf = 0.6 + min(0.2, thrust * 0.2) + (0.1 if slope20[i] > 0 else 0.0)
                    add('BreakoutHigh', i, 'bull', conf, level=float(rolling_max[i - 1]), close=float(c[i]))
                if np.isfinite(rolling_min[i - 1]) and c[i] < rolling_min[i - 1]:
                    # медвежий пробой
                    thrust = float((rolling_min[i - 1] - c[i]) / (
                        rng[i] if np.isfinite(rng[i]) and rng[i] > 0 else max(1e-9, abs(c[i] * 0.001))))
                    conf = 0.6 + min(0.2, thrust * 0.2) + (0.1 if slope20[i] < 0 else 0.0)
                    add('BreakoutLow', i, 'bear', conf, level=float(rolling_min[i - 1]), close=float(c[i]))

        # 5) вернём только последние ~100 событий (чтобы не захламлять ответ)
        if len(events) > 100:
            events = events[-100:]

        return events

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