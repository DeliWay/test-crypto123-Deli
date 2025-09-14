import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import aiohttp
import json
from functools import lru_cache
import concurrent.futures
from numba import jit
import talib
import requests
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Конфигурация
TRADINGVIEW_API_URL = "https://scanner.tradingview.com/global/scan"
CACHE_TTL = 300  # 5 минут
MAX_WORKERS = 4


class AnalysisEngine:
    """Высокопроизводительный движок технического анализа"""

    def __init__(self):
        self.session = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.cache = {}
        self.cache_time = {}

    async def init(self):
        """Инициализация асинхронной сессии"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Закрытие ресурсов"""
        if self.session:
            await self.session.close()
        self.executor.shutdown()

    @lru_cache(maxsize=1000)
    def _cached_indicators(self, symbol: str, interval: str, limit: int) -> Optional[Dict]:
        """Кэширование результатов расчета индикаторов"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = datetime.now()

        if cache_key in self.cache:
            if current_time - self.cache_time[cache_key] < timedelta(seconds=CACHE_TTL):
                return self.cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Сохранение в кэш"""
        self.cache[cache_key] = data
        self.cache_time[cache_key] = datetime.now()

    @jit(nopython=True, fastmath=True)
    def _fast_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Оптимизированный расчет EMA"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    @jit(nopython=True, fastmath=True)
    def _fast_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Оптимизированный расчет RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period + 1]

        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - 100 / (1 + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]

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

    async def get_tradingview_data(self, symbol: str, interval: str) -> Optional[Dict]:
        """Получение precomputed данных из TradingView"""
        try:
            if not self.session:
                await self.init()

            payload = {
                "symbols": {"tickers": [f"BYBIT:{symbol}"], "query": {"types": []}},
                "columns": [
                    "Recommend.All|1", "RSI|1", "Stoch.K|1", "Stoch.D|1",
                    "MACD.macd|1", "MACD.signal|1", "BB.upper|1", "BB.lower|1",
                    "EMA50|1", "EMA200|1", "Volatility.D|1"
                ]
            }

            async with self.session.post(TRADINGVIEW_API_URL, json=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data.get('data'):
                        return data['data'][0]['d']
            return None

        except Exception as e:
            logger.warning(f"TradingView API error: {e}")
            return None

    async def calculate_indicators(self, closes: List[float], use_tradingview: bool = True) -> Dict[str, Any]:
        """Асинхронный расчет всех индикаторов"""
        prices = np.array(closes, dtype=np.float64)

        # Параллельный расчет индикаторов
        loop = asyncio.get_event_loop()

        # Используем TradingView для precomputed данных если доступно
        tv_data = None
        if use_tradingview and len(closes) > 50:
            try:
                tv_data = await self.get_tradingview_data("BTCUSDT", "1h")  # Пример
            except:
                tv_data = None

        # Параллельные вычисления
        tasks = [
            loop.run_in_executor(self.executor, self._calculate_ema, prices),
            loop.run_in_executor(self.executor, self._calculate_macd, prices),
            loop.run_in_executor(self.executor, self._calculate_rsi, prices),
            loop.run_in_executor(self.executor, self._calculate_bollinger, prices),
            loop.run_in_executor(self.executor, self._calculate_stochastic, prices, prices, prices),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка результатов
        indicators = {}
        if not any(isinstance(r, Exception) for r in results):
            ema9, ema21 = results[0]
            macd, macd_signal, macd_hist = results[1]
            rsi = results[2]
            bb_upper, bb_mid, bb_lower = results[3]
            stoch_k, stoch_d = results[4]

            indicators = {
                'ema9': ema9.tolist() if hasattr(ema9, 'tolist') else ema9,
                'ema21': ema21.tolist() if hasattr(ema21, 'tolist') else ema21,
                'macd': macd.tolist() if hasattr(macd, 'tolist') else macd,
                'macd_signal': macd_signal.tolist() if hasattr(macd_signal, 'tolist') else macd_signal,
                'macd_hist': macd_hist.tolist() if hasattr(macd_hist, 'tolist') else macd_hist,
                'rsi': rsi.tolist() if hasattr(rsi, 'tolist') else rsi,
                'bb_upper': bb_upper.tolist() if hasattr(bb_upper, 'tolist') else bb_upper,
                'bb_mid': bb_mid.tolist() if hasattr(bb_mid, 'tolist') else bb_mid,
                'bb_lower': bb_lower.tolist() if hasattr(bb_lower, 'tolist') else bb_lower,
                'stoch_k': stoch_k.tolist() if hasattr(stoch_k, 'tolist') else stoch_k,
                'stoch_d': stoch_d.tolist() if hasattr(stoch_d, 'tolist') else stoch_d,
            }

        # Объединяем с TradingView данными если есть
        if tv_data:
            indicators.update(self._parse_tradingview_data(tv_data))

        return indicators

    def _calculate_ema(self, prices: np.ndarray) -> tuple:
        """Расчет EMA"""
        try:
            ema9 = talib.EMA(prices, timeperiod=9)
            ema21 = talib.EMA(prices, timeperiod=21)
            return ema9, ema21
        except:
            return self._fast_ema(prices, 9), self._fast_ema(prices, 21)

    def _calculate_macd(self, prices: np.ndarray) -> tuple:
        """Расчет MACD"""
        macd, macd_signal, macd_hist = talib.MACD(prices)
        return macd, macd_signal, macd_hist

    def _calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """Расчет RSI"""
        try:
            return talib.RSI(prices)
        except:
            return self._fast_rsi(prices)

    def _calculate_bollinger(self, prices: np.ndarray) -> tuple:
        """Расчет Bollinger Bands"""
        bb_upper, bb_mid, bb_lower = talib.BBANDS(prices)
        return bb_upper, bb_mid, bb_lower

    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple:
        """Расчет Stochastic"""
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        return stoch_k, stoch_d

    def _parse_tradingview_data(self, tv_data: List) -> Dict:
        """Парсинг данных TradingView"""
        return {
            'rsi_tv': tv_data[1] if len(tv_data) > 1 else None,
            'stoch_k_tv': tv_data[2] if len(tv_data) > 2 else None,
            'stoch_d_tv': tv_data[3] if len(tv_data) > 3 else None,
            'macd_tv': tv_data[4] if len(tv_data) > 4 else None,
            'macd_signal_tv': tv_data[5] if len(tv_data) > 5 else None,
            'bb_upper_tv': tv_data[6] if len(tv_data) > 6 else None,
            'bb_lower_tv': tv_data[7] if len(tv_data) > 7 else None,
        }

    async def detect_signals(self, closes: List[float], strategy: str = 'classic') -> Dict[str, Any]:
        """Обнаружение торговых сигналов с улучшенной точностью"""
        indicators = await self.calculate_indicators(closes)

        signals = {
            'alerts': [],
            'strength': 0,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }

        # Добавляем индикаторы в результаты
        signals.update(indicators)

        # Расширенный анализ сигналов
        alerts = self._analyze_signals(indicators, closes, strategy)
        signals['alerts'] = alerts

        # Расчет уверенности сигнала
        signals['confidence'] = self._calculate_confidence(alerts, indicators)
        signals['strength'] = int(signals['confidence'] * 100)

        return signals

    def _analyze_signals(self, indicators: Dict, closes: List[float], strategy: str) -> List[Dict]:
        """Расширенный анализ сигналов с улучшенной точностью"""
        alerts = []
        current_price = closes[-1] if closes else 0

        # Множественные подтверждения сигналов
        bullish_confirmations = 0
        bearish_confirmations = 0

        # 1. RSI анализ
        if indicators.get('rsi') and len(indicators['rsi']) > 0:
            rsi = indicators['rsi'][-1]
            if rsi < 30:
                alerts.append({'type': 'oversold', 'indicator': 'RSI', 'value': rsi})
                bullish_confirmations += 1
            elif rsi > 70:
                alerts.append({'type': 'overbought', 'indicator': 'RSI', 'value': rsi})
                bearish_confirmations += 1

        # 2. MACD анализ
        if (indicators.get('macd') and indicators.get('macd_signal') and
                len(indicators['macd']) > 0 and len(indicators['macd_signal']) > 0):

            macd = indicators['macd'][-1]
            signal = indicators['macd_signal'][-1]
            hist = indicators.get('macd_hist', [0])[-1] if indicators.get('macd_hist') else 0

            if macd > signal and hist > 0:
                alerts.append({'type': 'bullish_crossover', 'indicator': 'MACD', 'value': hist})
                bullish_confirmations += 2  # Более сильное подтверждение
            elif macd < signal and hist < 0:
                alerts.append({'type': 'bearish_crossover', 'indicator': 'MACD', 'value': hist})
                bearish_confirmations += 2

        # 3. moving averages
        if (indicators.get('ema9') and indicators.get('ema21') and
                len(indicators['ema9']) > 0 and len(indicators['ema21']) > 0):

            ema9 = indicators['ema9'][-1]
            ema21 = indicators['ema21'][-1]

            if ema9 > ema21 and current_price > ema9:
                alerts.append({'type': 'bullish_trend', 'indicator': 'EMA', 'value': current_price})
                bullish_confirmations += 1
            elif ema9 < ema21 and current_price < ema9:
                alerts.append({'type': 'bearish_trend', 'indicator': 'EMA', 'value': current_price})
                bearish_confirmations += 1

        # 4. Bollinger Bands
        if (indicators.get('bb_upper') and indicators.get('bb_lower') and
                len(indicators['bb_upper']) > 0 and len(indicators['bb_lower']) > 0):

            bb_upper = indicators['bb_upper'][-1]
            bb_lower = indicators['bb_lower'][-1]

            if current_price < bb_lower:
                alerts.append({'type': 'oversold_bb', 'indicator': 'BB', 'value': current_price})
                bullish_confirmations += 1
            elif current_price > bb_upper:
                alerts.append({'type': 'overbought_bb', 'indicator': 'BB', 'value': current_price})
                bearish_confirmations += 1

        # 5. Stochastic
        if (indicators.get('stoch_k') and indicators.get('stoch_d') and
                len(indicators['stoch_k']) > 0 and len(indicators['stoch_d']) > 0):

            stoch_k = indicators['stoch_k'][-1]
            stoch_d = indicators['stoch_d'][-1]

            if stoch_k < 20 and stoch_d < 20:
                alerts.append({'type': 'oversold_stoch', 'indicator': 'Stochastic', 'value': stoch_k})
                bullish_confirmations += 1
            elif stoch_k > 80 and stoch_d > 80:
                alerts.append({'type': 'overbought_stoch', 'indicator': 'Stochastic', 'value': stoch_k})
                bearish_confirmations += 1

        # Определение общего направления
        if bullish_confirmations > bearish_confirmations and bullish_confirmations >= 2:
            alerts.append({'type': 'strong_bullish', 'indicator': 'Multiple', 'value': bullish_confirmations})
        elif bearish_confirmations > bullish_confirmations and bearish_confirmations >= 2:
            alerts.append({'type': 'strong_bearish', 'indicator': 'Multiple', 'value': bearish_confirmations})

        return alerts

    def _calculate_confidence(self, alerts: List[Dict], indicators: Dict) -> float:
        """Расчет уверенности в сигналах"""
        if not alerts:
            return 0.0

        # Веса разных типов сигналов
        weights = {
            'strong_bullish': 0.9, 'strong_bearish': 0.9,
            'bullish_crossover': 0.7, 'bearish_crossover': 0.7,
            'bullish_trend': 0.6, 'bearish_trend': 0.6,
            'oversold': 0.5, 'overbought': 0.5,
            'oversold_bb': 0.4, 'overbought_bb': 0.4,
            'oversold_stoch': 0.4, 'overbought_stoch': 0.4
        }

        total_confidence = 0.0
        count = 0

        for alert in alerts:
            weight = weights.get(alert['type'], 0.3)
            total_confidence += weight
            count += 1

        return min(total_confidence / max(count, 1), 1.0)


# Глобальный экземпляр
analysis_engine = AnalysisEngine()


# Функции для обратной совместимости
async def detect_signals(closes: List[float], strategy: str = 'classic') -> Dict[str, Any]:
    """Обнаружение сигналов (асинхронная версия)"""
    return await analysis_engine.detect_signals(closes, strategy)


def detect_signals_sync(closes: List[float], strategy: str = 'classic') -> Dict[str, Any]:
    """Обнаружение сигналов (синхронная версия)"""
    return asyncio.run(analysis_engine.detect_signals(closes, strategy))


async def calculate_indicators(closes: List[float]) -> Dict[str, Any]:
    """Расчет индикаторов (асинхронная версия)"""
    return await analysis_engine.calculate_indicators(closes)


def calculate_indicators_sync(closes: List[float]) -> Dict[str, Any]:
    """Расчет индикаторов (синхронная версия)"""
    return asyncio.run(analysis_engine.calculate_indicators(closes))