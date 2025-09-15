# backend/bybit_client.py
import os
import time
import asyncio
import aiohttp
import async_timeout
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import json
from functools import wraps
import redis.asyncio as redis
from circuitbreaker import circuit
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Конфигурация
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:5000')
BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
REQUEST_TIMEOUT = 15
MAX_CONCURRENT_REQUESTS = 100
CACHE_TTL = 300  # 5 минут


@dataclass
class RateLimiter:
    max_requests: int = 5
    period: float = 1.0
    calls: List[float] = None

    def __post_init__(self):
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Удаляем старые вызовы
            self.calls = [call for call in self.calls if now - call < self.period]

            if len(self.calls) >= self.max_requests:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time.time()

            self.calls.append(now)


class BybitClient:
    """Высокопроизводительный асинхронный клиент для Bybit API"""

    _instance = None
    _session = None
    _redis = None
    _ws_connections = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init(self):
        """Инициализация клиента"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            )

        if self._redis is None:
            try:
                self._redis = await redis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self._redis = None

        self.rate_limiter = RateLimiter(5, 1.0)
        self.symbols_cache = {}
        self.cache_timeout = CACHE_TTL

    async def close(self):
        """Закрытие соединений"""
        if self._session and not self._session.closed:
            await self._session.close()

        if self._redis:
            await self._redis.close()

        for ws in self._ws_connections.values():
            await ws.close()
        self._ws_connections.clear()

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        if not self._redis:
            return self.symbols_cache.get(key)

        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return self.symbols_cache.get(key)

    async def _cache_set(self, key: str, data: Any, ttl: int = CACHE_TTL):
        """Сохранение данных в кэш"""
        self.symbols_cache[key] = data

        if self._redis:
            try:
                await self._redis.setex(key, ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

    @circuit(failure_threshold=5, recovery_timeout=60)
    async def _make_request(self, endpoint: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Выполнение HTTP запроса с retry логикой"""
        url = f"{BYBIT_BASE_URL}/v5/{endpoint}"

        for attempt in range(retries):
            try:
                await self.rate_limiter.acquire()

                async with async_timeout.timeout(REQUEST_TIMEOUT):
                    async with self._session.get(url, params=params) as response:
                        if response.status == 429:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue

                        response.raise_for_status()
                        data = await response.json()

                        if data.get('retCode') == 0:
                            return data
                        else:
                            logger.warning(f"API error: {data.get('retMsg')}")
                            return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

            if attempt < retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

        return None

    async def get_spot_symbols(self, quote_coin: str = 'USDT') -> List[str]:
        """Получение списка спот символов"""
        cache_key = f"symbols:{quote_coin}"
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        data = await self._make_request('market/instruments-info', {'category': 'spot'})
        if not data:
            return await self._get_default_symbols()

        symbols_list = data.get('result', {}).get('list', [])
        symbols = [
            symbol['symbol'] for symbol in symbols_list
            if (symbol.get('quoteCoin') == quote_coin and
                symbol.get('status') == 'Trading' and
                symbol.get('baseCoin') not in ['USDC', 'BUSD', 'TUSD'])
        ]

        symbols = sorted(symbols)
        await self._cache_set(cache_key, symbols)
        return symbols

    async def _get_default_symbols(self) -> List[str]:
        """Резервный список символов"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT'
        ]

    async def get_klines(self, symbol: str, interval: str = '60', limit: int = 500) -> Optional[pd.DataFrame]:
        """Получение OHLCV данных"""
        cache_key = f"klines:{symbol}:{interval}:{limit}"
        cached = await self._cache_get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        interval_map = {
            '1': '1', '5': '5', '15': '15', '30': '30',
            '60': '60', '240': '240', 'D': 'D', 'W': 'W', 'M': 'M'
        }

        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval_map.get(interval, '60'),
            'limit': min(limit, 1000)
        }

        data = await self._make_request('market/kline', params)
        if not data or not data.get('result', {}).get('list'):
            return None

        df = await self._process_klines_data(data['result']['list'])
        if df is not None:
            await self._cache_set(cache_key, df.to_dict('records'), ttl=60)

        return df

    async def _process_klines_data(self, candles: List) -> Optional[pd.DataFrame]:
        """Обработка данных свечей"""
        if not candles:
            return None

        try:
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()
            if df.empty:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Добавляем расчетные поля как в оригинальном коде
            df['price_change'] = df['close'].pct_change() * 100
            df['price_change_abs'] = df['close'].diff()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

            return df

        except Exception as e:
            logger.error(f"Error processing klines: {e}")
            return None

    async def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """Информация о тикере"""
        cache_key = f"ticker:{symbol}"
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        params = {'category': 'spot', 'symbol': symbol}
        data = await self._make_request('market/tickers', params)

        if not data or not data.get('result', {}).get('list'):
            return None

        ticker_data = data['result']['list'][0]
        enhanced = await self._enhance_ticker_data(ticker_data)

        if enhanced:
            await self._cache_set(cache_key, enhanced, ttl=10)

        return enhanced

    async def _enhance_ticker_data(self, ticker: Dict) -> Dict:
        """Обогащение данных тикера"""
        try:
            numeric_fields = [
                'lastPrice', 'highPrice24h', 'lowPrice24h', 'prevPrice24h',
                'price24hPcnt', 'volume24h', 'turnover24h'
            ]

            enhanced = {}
            for field in numeric_fields:
                value = ticker.get(field)
                enhanced[field] = float(value) if value else 0.0

            # Расчет дополнительных метрик как в оригинале
            enhanced['price_change_24h_abs'] = enhanced['lastPrice'] - enhanced['prevPrice24h']
            enhanced['price_change_24h_percent'] = enhanced['price24hPcnt'] * 100
            enhanced['average_price_24h'] = (
                enhanced['turnover24h'] / enhanced['volume24h']
                if enhanced['volume24h'] > 0 else enhanced['lastPrice']
            )

            # Сохраняем все оригинальные поля
            result = {**ticker, **enhanced}
            return result

        except Exception as e:
            logger.error(f"Error enhancing ticker: {e}")
            return ticker

    async def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Стакан цен"""
        cache_key = f"orderbook:{symbol}:{limit}"
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        params = {
            'category': 'spot',
            'symbol': symbol,
            'limit': min(limit, 50)
        }

        data = await self._make_request('market/orderbook', params)
        if not data:
            return None

        orderbook = data.get('result', {})
        processed = await self._process_orderbook_data(orderbook)

        if processed:
            await self._cache_set(cache_key, processed, ttl=5)

        return processed

    async def _process_orderbook_data(self, orderbook: Dict) -> Dict:
        """Обработка стакана цен с полным функционалом"""
        try:
            bids = pd.DataFrame(orderbook.get('b', []), columns=['price', 'quantity'])
            asks = pd.DataFrame(orderbook.get('a', []), columns=['price', 'quantity'])

            for df in [bids, asks]:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                df.dropna(inplace=True)

            # Расчет метрик как в оригинальном коде
            bid_volume = bids['quantity'].sum() if not bids.empty else 0
            ask_volume = asks['quantity'].sum() if not asks.empty else 0
            total_volume = bid_volume + ask_volume

            spread = 0
            spread_percent = 0
            if not bids.empty and not asks.empty:
                spread = asks['price'].iloc[0] - bids['price'].iloc[0]
                if bids['price'].iloc[0] > 0:
                    spread_percent = (spread / bids['price'].iloc[0]) * 100

            orderbook_imbalance = 0
            if total_volume > 0:
                orderbook_imbalance = ((bid_volume - ask_volume) / total_volume) * 100

            return {
                'bids': bids.to_dict('records'),
                'asks': asks.to_dict('records'),
                'metrics': {
                    'bid_volume': float(bid_volume),
                    'ask_volume': float(ask_volume),
                    'total_volume': float(total_volume),
                    'spread': float(spread),
                    'spread_percent': float(spread_percent),
                    'orderbook_imbalance': float(orderbook_imbalance)
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            return None

    async def get_multiple_timeframes(self, symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Данные для нескольких таймфреймов"""
        if timeframes is None:
            timeframes = ['60', '240', '1440']  # 1h, 4h, 1d

        results = {}
        tasks = []

        for tf in timeframes:
            task = self.get_klines(symbol, tf, 200)
            tasks.append((tf, task))

        for tf, task in tasks:
            try:
                data = await task
                if data is not None:
                    results[tf] = data
                    logger.info(f"Retrieved {tf} data for {symbol}: {len(data)} candles")
                else:
                    logger.warning(f"Failed to get {tf} data for {symbol}")
            except Exception as e:
                logger.error(f"Error getting {tf} data for {symbol}: {e}")
                continue

        return results

    async def get_batch_symbols_data(self, symbols: List[str], interval: str = '60', limit: int = 100) -> Dict[
        str, pd.DataFrame]:
        """Пакетное получение данных"""
        results = {}
        tasks = []

        for symbol in symbols:
            task = self.get_klines(symbol, interval, limit)
            tasks.append((symbol, task))

        for symbol, task in tasks:
            try:
                data = await task
                if data is not None:
                    results[symbol] = data
                # Уважаем rate limits
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue

        return results

    async def get_server_time(self) -> Optional[Dict]:
        """Время сервера"""
        try:
            data = await self._make_request('market/time')
            if data and 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Фандинг рейт для перпетуальных контрактов"""
        try:
            params = {'category': 'linear', 'symbol': symbol}
            data = await self._make_request('market/funding/history', params)
            if data and 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {e}")
            return None

    async def get_market_data_binance(self, symbol: str, interval: str = '1h', limit: int = 500) -> Optional[
        pd.DataFrame]:
        """Поддержка Binance API для обратной совместимости"""
        # Реализация Binance API может быть добавлена при необходимости
        logger.warning("Binance API not implemented, using Bybit as fallback")
        return await self.get_klines(symbol, interval, limit)

    async def get_ticker_info_binance(self, symbol: str) -> Optional[Dict]:
        """Поддержка Binance API для обратной совместимости"""
        logger.warning("Binance API not implemented, using Bybit as fallback")
        return await self.get_ticker_info(symbol)


# Глобальный экземпляр
_client = BybitClient()


async def init_bybit_client():
    """Инициализация клиента"""
    await _client.init()


async def close_bybit_client():
    """Закрытие клиента"""
    await _client.close()


# Функции для обратной совместимости с существующим кодом
async def get_bybit_symbols(quote_coin: str = 'USDT') -> List[str]:
    """Получение списка символов Bybit (асинхронная версия)"""
    return await _client.get_spot_symbols(quote_coin)


async def get_market_data(symbol: str, interval: str = '60', limit: int = 500, source: str = 'bybit') -> Optional[
    pd.DataFrame]:
    """Получение рыночных данных (асинхронная версия)"""
    if source == 'binance':
        return await _client.get_market_data_binance(symbol, interval, limit)
    return await _client.get_klines(symbol, interval, limit)


async def get_liquidity_data(symbol: str, limit: int = 25) -> Optional[Dict]:
    """Получение данных ликвидности (асинхронная версия)"""
    return await _client.get_orderbook(symbol, limit)


async def get_ticker_info(symbol: str, source: str = 'bybit') -> Optional[Dict]:
    """Получение информации о тикере (асинхронная версия)"""
    if source == 'binance':
        return await _client.get_ticker_info_binance(symbol)
    return await _client.get_ticker_info(symbol)


async def get_multiple_timeframes(symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Получение данных по нескольким таймфреймам (асинхронная версия)"""
    return await _client.get_multiple_timeframes(symbol, timeframes)


async def get_batch_symbols_data(symbols: List[str], interval: str = '60', limit: int = 100) -> Dict[str, pd.DataFrame]:
    """Пакетное получение данных (асинхронная версия)"""
    return await _client.get_batch_symbols_data(symbols, interval, limit)


async def get_available_symbols(quote_coin: str = 'USDT', source: str = 'bybit') -> List[str]:
    """Получение доступных символов (асинхронная версия)"""
    if source == 'binance':
        # Fallback для Binance
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    return await _client.get_spot_symbols(quote_coin)





# Синхронные обертки для существующего кода
# Добавь эти функции в конец файла backend/bybit_client.py

def get_bybit_symbols_sync(quote_coin: str = 'USDT') -> List[str]:
    """Синхронная версия получения символов Bybit"""
    return asyncio.run(_client.get_spot_symbols(quote_coin))


def get_ticker_info_sync(symbol: str) -> Optional[Dict]:
    """Синхронная версия получения информации о тикере"""
    return asyncio.run(_client.get_ticker_info(symbol))


def get_market_data_sync(symbol: str, interval: str = '60', limit: int = 500) -> Optional[pd.DataFrame]:
    """Синхронная версия получения рыночных данных"""
    return asyncio.run(_client.get_klines(symbol, interval, limit))


def get_liquidity_data_sync(symbol: str, limit: int = 25) -> Optional[Dict]:
    """Синхронная версия получения данных ликвидности"""
    return asyncio.run(_client.get_orderbook(symbol, limit))


# Старые синхронные функции для обратной совместимости с app.py
def klines(symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict[str, Any]]]:
    """
    Старая синхронная функция для обратной совместимости
    Возвращает данные в формате: [{"open_time": timestamp, "open": price, "high": price, "low": price, "close": price, "volume": volume}]
    """
    try:
        df = get_market_data_sync(symbol, interval, limit)
        if df is None:
            return None

        # Конвертируем DataFrame в старый формат
        result = []
        for _, row in df.iterrows():
            result.append({
                "open_time": int(row['timestamp'].timestamp() * 1000),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        return result
    except Exception as e:
        logger.error(f"Error in klines compatibility function: {e}")
        return None


def ticker(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Старая синхронная функция для обратной совместимости
    Возвращает данные в формате: {"symbol": "...", "lastPrice": "...", "price24hPcnt": "...", ...}
    """
    try:
        ticker_data = get_ticker_info_sync(symbol)
        if ticker_data is None:
            return None

        # Преобразуем в старый формат
        return {
            "symbol": symbol,
            "lastPrice": ticker_data.get('lastPrice'),
            "price24hPcnt": ticker_data.get('price24hPcnt'),
            "highPrice24h": ticker_data.get('highPrice24h'),
            "lowPrice24h": ticker_data.get('lowPrice24h'),
            "ts": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error in ticker compatibility function: {e}")
        return None


# Глобальная переменная для обратной совместимости
bybit_client = type('CompatClient', (), {
    'klines': klines,
    'ticker': ticker,
    'get_spot_symbols': get_bybit_symbols_sync,
    'get_klines': get_market_data_sync,
    'get_ticker_info': get_ticker_info_sync,
    'get_orderbook': get_liquidity_data_sync
})()


def get_available_symbols_sync(quote_coin: str = 'USDT', source: str = 'bybit') -> List[str]:
    """Получение доступных символов (синхронная версия)"""
    if source == 'binance':
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    return asyncio.run(_client.get_spot_symbols(quote_coin))


def get_multiple_timeframes_sync(symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Получение данных по нескольким таймфреймам (синхронная версия)"""
    return asyncio.run(_client.get_multiple_timeframes(symbol, timeframes))


def get_batch_symbols_data_sync(symbols: List[str], interval: str = '60', limit: int = 100) -> Dict[str, pd.DataFrame]:
    """Пакетное получение данных (синхронная версия)"""
    return asyncio.run(_client.get_batch_symbols_data(symbols, interval, limit))


# Дополнительные функции для полной совместимости
def get_server_time_sync() -> Optional[Dict]:
    """Время сервера (синхронная версия)"""
    return asyncio.run(_client.get_server_time())


def get_funding_rate_sync(symbol: str) -> Optional[Dict]:
    """Фандинг рейт (синхронная версия)"""
    return asyncio.run(_client.get_funding_rate(symbol))


