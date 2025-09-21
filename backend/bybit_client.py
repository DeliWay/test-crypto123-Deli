"""
backend.bybit_client.py
ULTRA-PERFORMANCE BYBIT CLIENT V2
"""

import os
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import ssl as _ssl
import logging
import json
from functools import wraps, lru_cache
import redis.asyncio as redis
from circuitbreaker import circuit
from dataclasses import dataclass, field
import hashlib
import concurrent.futures
from enum import Enum
import backoff

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bybit_client")

class ExchangeSource(Enum):
    """Источники данных"""
    BYBIT = "bybit"
    BINANCE = "binance"

class MarketDataType(Enum):
    """Типы рыночных данных"""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"

# Конфигурация
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
BYBIT_BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://api.binance.com"

REQUEST_TIMEOUT = 10.0
MAX_CONCURRENT_REQUESTS = 300
CACHE_TTL = 180
CONNECTION_POOL_SIZE = 50

@dataclass
class UltraRateLimiter:
    """Ультра-оптимизированный rate limiter"""
    max_requests: int = 15
    period: float = 1.0
    calls: List[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self):
        """Получение доступа с учетом лимитов"""
        async with self.lock:
            now = time.monotonic()
            self.calls = [call for call in self.calls if now - call < self.period]

            if len(self.calls) >= self.max_requests:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time.monotonic()

            self.calls.append(now)

@dataclass
class ExchangeConfig:
    """Конфигурация биржи"""
    base_url: str
    ws_url: str
    rate_limiter: UltraRateLimiter
    weight: int = 1
    enabled: bool = True

class UltraBybitClient:
    """Ультра-оптимизированный клиент с поддержкой multiple exchanges"""

    _instance = None
    _sessions: Dict[ExchangeSource, aiohttp.ClientSession] = {}
    _redis = None
    _ws_connections = {}
    _exchange_configs: Dict[ExchangeSource, ExchangeConfig] = {}
    _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    _initialized = False
    _last_successful_symbols: Dict[str, List[str]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_exchange_configs()
        return cls._instance

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _init_exchange_configs(self):
        """Инициализация конфигураций бирж"""
        self._exchange_configs = {
            ExchangeSource.BYBIT: ExchangeConfig(
                base_url=BYBIT_BASE_URL,
                ws_url="wss://stream.bybit.com/v5/public/spot",
                rate_limiter=UltraRateLimiter(20, 0.5),
                weight=3
            ),
            ExchangeSource.BINANCE: ExchangeConfig(
                base_url=BINANCE_BASE_URL,
                ws_url="wss://stream.binance.com:9443/ws",
                rate_limiter=UltraRateLimiter(15, 0.5),
                weight=2
            )
        }

    async def init(self):
        """Инициализация клиента"""
        if self._initialized:
            return

        # Инициализация Redis
        try:
            self._redis = await redis.from_url(
                REDIS_URL,
                decode_responses=True,
                max_connections=5,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=False
            )
            await self._redis.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._redis = None

        # Проверка доступности бирж
        await self._check_exchange_availability()

        # Инициализация сессий
        for exchange, config in self._exchange_configs.items():
            if config.enabled:
                try:
                    connector = aiohttp.TCPConnector(
                        limit=CONNECTION_POOL_SIZE,
                        limit_per_host=30,
                        enable_cleanup_closed=True,
                        force_close=False,
                        ssl=False
                    )

                    self._sessions[exchange] = aiohttp.ClientSession(
                        connector=connector,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'application/json',
                            'Accept-Encoding': 'gzip, deflate',
                            'Connection': 'keep-alive'
                        },
                        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                        trust_env=True
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize session for {exchange.value}: {e}")
                    config.enabled = False

        self._initialized = True
        logger.info(f"Initialized exchanges: {[e.value for e, c in self._exchange_configs.items() if c.enabled]}")

    async def ensure_initialized(self):
        """Гарантирует, что клиент инициализирован"""
        if not self._initialized:
            await self.init()

    async def _check_exchange_availability(self):
        """Проверка доступности бирж"""
        checks = {
            ExchangeSource.BINANCE: "/api/v3/ping",
            ExchangeSource.BYBIT: "/v5/market/time",
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as temp_session:
            tasks = []
            for exchange, endpoint in checks.items():
                url = self._exchange_configs[exchange].base_url + endpoint
                tasks.append(self._probe_exchange(temp_session, exchange, url))

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _probe_exchange(self, session, exchange, url):
        """Проверка доступности конкретной биржи"""
        try:
            async with session.get(url) as response:
                self._exchange_configs[exchange].enabled = (response.status == 200)
                if not self._exchange_configs[exchange].enabled:
                    logger.warning(f"{exchange.value} disabled: HTTP {response.status}")
        except Exception as e:
            self._exchange_configs[exchange].enabled = False
            logger.warning(f"{exchange.value} disabled: {e}")

    async def close(self):
        """Закрытие всех соединений"""
        for exchange, session in self._sessions.items():
            if not session.closed:
                await session.close()

        if self._redis:
            await self._redis.close()

        for ws in self._ws_connections.values():
            await ws.close()

        self._sessions.clear()
        self._ws_connections.clear()

        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown()

        self._initialized = False
        logger.info("Client fully closed")

    def _get_available_exchanges(self) -> List[ExchangeSource]:
        """Получение списка доступных бирж"""
        return [exchange for exchange, config in self._exchange_configs.items()
                if config.enabled and exchange in self._sessions and not self._sessions[exchange].closed]

    def _select_best_exchange(self) -> Optional[ExchangeSource]:
        """Выбор оптимальной биржи"""
        available = self._get_available_exchanges()
        if not available:
            return None

        weights = [self._exchange_configs[exchange].weight for exchange in available]
        total_weight = sum(weights)

        if total_weight == 0:
            return available[0]

        rand = np.random.random() * total_weight
        cumulative = 0

        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return available[i]

        return available[-1]

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def _make_request(self, endpoint: str, params: Dict = None,
                            exchange: ExchangeSource = None, retries: int = 2) -> Optional[Dict]:
        """HTTP запрос с автоматическим выбором биржи"""
        await self.ensure_initialized()

        if exchange is None:
            exchange = self._select_best_exchange()
            if exchange is None:
                logger.error("No available exchanges")
                return None

        if exchange not in self._sessions or self._sessions[exchange].closed:
            logger.warning(f"Session for {exchange.value} not available")
            return None

        config = self._exchange_configs[exchange]
        url = self._build_url(exchange, endpoint)

        await config.rate_limiter.acquire()

        try:
            async with self._sessions[exchange].get(url, params=params) as response:
                if response.status == 429:
                    logger.warning(f"Rate limit exceeded on {exchange.value}")
                    other = [e for e in self._get_available_exchanges() if e != exchange]
                    if other and retries > 0:
                        return await self._make_request(endpoint, params, other[0], retries - 1)
                    await asyncio.sleep(0.3)
                    return await self._make_request(endpoint, params, exchange, retries - 1)

                response.raise_for_status()
                data = await response.json()
                return self._process_response(exchange, data)

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.warning(f"Request failed on {exchange.value}: {e}")
            if retries > 0:
                other_exchanges = [e for e in self._get_available_exchanges() if e != exchange]
                if other_exchanges:
                    return await self._make_request(endpoint, params, other_exchanges[0], retries - 1)
            return None
        except Exception as e:
            logger.error(f"Unexpected error on {exchange.value}: {e}")
            return None

    def _build_url(self, exchange: ExchangeSource, endpoint: str) -> str:
        """Построение URL для конкретной биржи"""
        base_url = self._exchange_configs[exchange].base_url
        return f"{base_url}/{endpoint.lstrip('/')}"

    def _process_response(self, exchange: ExchangeSource, data: Dict) -> Optional[Dict]:
        """Обработка ответа в зависимости от биржи"""
        try:
            if exchange == ExchangeSource.BYBIT:
                if data.get('retCode') == 0:
                    return data.get('result', {})
                else:
                    logger.warning(f"Bybit API error: {data.get('retMsg')}")
                    return None
            elif exchange == ExchangeSource.BINANCE:
                if 'code' in data and data['code'] != 200:
                    logger.warning(f"Binance API error: {data.get('msg')}")
                    return None
                return data
            return data
        except Exception as e:
            logger.error(f"Error processing response from {exchange.value}: {e}")
            return None

    async def get_spot_symbols(self, quote_coin: str = 'USDT') -> List[str]:
        """Получение списка спот символов"""
        await self.ensure_initialized()

        cache_key = f"symbols:spot:{quote_coin.upper()}"
        cached = await self._cache_get(cache_key)
        if cached:
            self._last_successful_symbols[quote_coin] = cached
            return cached

        if quote_coin in self._last_successful_symbols:
            return self._last_successful_symbols[quote_coin]

        symbols = set()
        tasks = []

        for exchange in self._get_available_exchanges():
            task = self._fetch_symbols_from_exchange(exchange, quote_coin)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                symbols.update(result)

        symbols_list = sorted(list(symbols))

        if symbols_list:
            await self._cache_set(cache_key, symbols_list, ttl=3600)
            self._last_successful_symbols[quote_coin] = symbols_list
            return symbols_list

        return self._last_successful_symbols.get(quote_coin, [])

    async def _fetch_symbols_from_exchange(self, exchange: ExchangeSource, quote_coin: str) -> List[str]:
        """Получение символов с конкретной биржи"""
        try:
            if exchange == ExchangeSource.BYBIT:
                data = await self._make_request('v5/market/instruments-info', {'category': 'spot'}, exchange)
                if data and 'list' in data:
                    return [
                        symbol['symbol'] for symbol in data['list']
                        if (symbol.get('quoteCoin') == quote_coin and symbol.get('status') == 'Trading')
                    ]
            elif exchange == ExchangeSource.BINANCE:
                data = await self._make_request('api/v3/exchangeInfo', None, exchange)
                if data and 'symbols' in data:
                    return [
                        symbol['symbol'] for symbol in data['symbols']
                        if (symbol.get('quoteAsset') == quote_coin and symbol.get('status') == 'TRADING')
                    ]
        except Exception as e:
            logger.warning(f"Failed to fetch symbols from {exchange.value}: {e}")
        return []

    async def get_klines(self, symbol: str, interval: str = '1h', limit: int = 500,
                         exchange: ExchangeSource = None) -> Optional[pd.DataFrame]:
        """Получение OHLCV данных"""
        await self.ensure_initialized()

        cache_key = f"klines:{symbol}:{interval}:{limit}:{exchange.value if exchange else 'auto'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        if exchange is None:
            exchange = self._select_best_exchange()

        data = await self._fetch_klines_from_exchange(exchange, symbol, interval, limit)
        if data is not None:
            df = self._process_klines_data(data, exchange)
            if df is not None:
                await self._cache_set(cache_key, df.to_dict('records'), ttl=60)
                return df

        return None

    async def _fetch_klines_from_exchange(self, exchange: ExchangeSource, symbol: str,
                                        interval: str, limit: int) -> Optional[List]:
        """Получение свечных данных с конкретной биржи"""
        try:
            params = self._build_klines_params(exchange, symbol, interval, limit)
            endpoint = self._get_klines_endpoint(exchange)
            data = await self._make_request(endpoint, params, exchange)
            return self._parse_klines_response(exchange, data) if data else None
        except Exception as e:
            logger.error(f"Failed to fetch klines from {exchange.value}: {e}")
            return None

    def _build_klines_params(self, exchange: ExchangeSource, symbol: str,
                           interval: str, limit: int) -> Dict:
        """Построение параметров запроса для свечных данных"""
        if exchange == ExchangeSource.BYBIT:
            interval_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
                           '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
                           '1d': 'D', '1w': 'W', '1M': 'M'}
            return {
                'category': 'spot',
                'symbol': symbol,
                'interval': interval_map.get(interval, '60'),
                'limit': min(limit, 1000)
            }
        elif exchange == ExchangeSource.BINANCE:
            return {
                'symbol': symbol,
                'interval': interval.lower(),
                'limit': min(limit, 1000)
            }
        return {}

    def _parse_klines_response(self, exchange: ExchangeSource, data: Any) -> Optional[List]:
        """Парсинг ответа со свечными данными"""
        try:
            if exchange == ExchangeSource.BYBIT:
                return data.get('list', [])
            elif exchange == ExchangeSource.BINANCE:
                return data
            return []
        except Exception as e:
            logger.error(f"Error parsing klines response from {exchange.value}: {e}")
            return []

    def _process_klines_data(self, candles: List, exchange: ExchangeSource) -> Optional[pd.DataFrame]:
        """Обработка свечных данных"""
        if not candles:
            return None

        try:
            unified_data = []
            for candle in candles:
                if exchange == ExchangeSource.BYBIT:
                    unified_data.append([
                        int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]),
                        float(candle[4]), float(candle[5]), float(candle[6])
                    ])
                elif exchange == ExchangeSource.BINANCE:
                    unified_data.append([
                        candle[0], float(candle[1]), float(candle[2]), float(candle[3]),
                        float(candle[4]), float(candle[5]), float(candle[7])
                    ])

            df = pd.DataFrame(unified_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()
            if df.empty:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Error processing klines data from {exchange.value}: {e}")
            return None

    async def get_ticker_info(self, symbol: str, exchange: ExchangeSource = None) -> Optional[Dict]:
        """Получение информации о тикере"""
        await self.ensure_initialized()

        cache_key = f"ticker:{symbol}:{exchange.value if exchange else 'auto'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        if exchange is None:
            exchange = self._select_best_exchange()

        data = await self._fetch_ticker_from_exchange(exchange, symbol)
        if data:
            await self._cache_set(cache_key, data, ttl=5)
            return data

        return None

    async def _fetch_ticker_from_exchange(self, exchange: ExchangeSource, symbol: str) -> Optional[Dict]:
        """Получение данных тикера с конкретной биржи"""
        try:
            if exchange == ExchangeSource.BYBIT:
                params = {'category': 'spot', 'symbol': symbol}
                data = await self._make_request('v5/market/tickers', params, exchange)
                if data and 'list' in data:
                    return data['list'][0] if data['list'] else None
            elif exchange == ExchangeSource.BINANCE:
                params = {'symbol': symbol}
                data = await self._make_request('api/v3/ticker/24hr', params, exchange)
                return data
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch ticker from {exchange.value}: {e}")
            return None

    async def get_orderbook(self, symbol: str, limit: int = 25,
                          exchange: ExchangeSource = None) -> Optional[Dict]:
        """Получение стакана заявок"""
        await self.ensure_initialized()

        cache_key = f"orderbook:{symbol}:{limit}:{exchange.value if exchange else 'auto'}"
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        if exchange is None:
            exchange = self._select_best_exchange()

        data = await self._fetch_orderbook_from_exchange(exchange, symbol, limit)
        if data:
            await self._cache_set(cache_key, data, ttl=3)
            return data

        return None

    async def _fetch_orderbook_from_exchange(self, exchange: ExchangeSource,
                                           symbol: str, limit: int) -> Optional[Dict]:
        """Получение стакана с конкретной биржи"""
        try:
            if exchange == ExchangeSource.BYBIT:
                params = {'category': 'spot', 'symbol': symbol, 'limit': min(limit, 50)}
                data = await self._make_request('v5/market/orderbook', params, exchange)
                return data
            elif exchange == ExchangeSource.BINANCE:
                params = {'symbol': symbol, 'limit': min(limit, 5000)}
                data = await self._make_request('api/v3/depth', params, exchange)
                return data
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook from {exchange.value}: {e}")
            return None

    async def get_batch_klines(self, symbols: List[str], interval: str = '1h',
                             limit: int = 100) -> Dict[str, Optional[pd.DataFrame]]:
        """Пакетное получение свечных данных"""
        await self.ensure_initialized()

        results = {}
        tasks = []

        for symbol in symbols:
            task = self.get_klines(symbol, interval, limit)
            tasks.append((symbol, task))

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def limited_task(symbol, task):
            async with semaphore:
                try:
                    data = await task
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    results[symbol] = None
                await asyncio.sleep(0.05)

        await asyncio.gather(*[limited_task(symbol, task) for symbol, task in tasks])
        return results

    async def get_multiple_timeframes(self, symbol: str,
                                    timeframes: List[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """Получение данных для multiple таймфреймов"""
        await self.ensure_initialized()

        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d', '1w']

        results = {}
        tasks = []

        for tf in timeframes:
            task = self.get_klines(symbol, tf, 200)
            tasks.append((tf, task))

        for tf, task in tasks:
            try:
                data = await task
                results[tf] = data
            except Exception as e:
                logger.error(f"Error getting {tf} data for {symbol}: {e}")
                results[tf] = None

        return results

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        if self._redis:
            try:
                data = await self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        return None

    async def _cache_set(self, key: str, data: Any, ttl: int = CACHE_TTL):
        """Сохранение данных в кэш"""
        if self._redis:
            try:
                if data and ((isinstance(data, (list, dict)) and len(data) > 0) or
                            (isinstance(data, pd.DataFrame) and not data.empty)):
                    actual_ttl = ttl * 2
                else:
                    actual_ttl = ttl // 3

                await self._redis.setex(key, actual_ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

    def _get_klines_endpoint(self, exchange: ExchangeSource) -> str:
        """Получение эндпоинта для свечных данных"""
        if exchange == ExchangeSource.BYBIT:
            return 'v5/market/kline'
        elif exchange == ExchangeSource.BINANCE:
            return 'api/v3/klines'
        return ''

    # Синхронные методы
    def get_spot_symbols_sync(self, quote_coin: str = 'USDT') -> List[str]:
        """Синхронная версия получения символов"""
        return _run_sync(self.get_spot_symbols(quote_coin))

    def get_klines_sync(self, symbol: str, interval: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Синхронная версия получения свечных данных"""
        return _run_sync(self.get_klines(symbol, interval, limit))

    def get_ticker_info_sync(self, symbol: str) -> Optional[Dict]:
        """Синхронная версия получения информации о тикере"""
        return _run_sync(self.get_ticker_info(symbol))

    def get_orderbook_sync(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Синхронная версия получения стакана"""
        return _run_sync(self.get_orderbook(symbol, limit))

    def get_batch_klines_sync(self, symbols: List[str], interval: str = '1h', limit: int = 100) -> Dict[
        str, Optional[pd.DataFrame]]:
        """Синхронная версия пакетного получения данных"""
        return _run_sync(self.get_batch_klines(symbols, interval, limit))

    def get_multiple_timeframes_sync(self, symbol: str, timeframes: List[str] = None) -> Dict[
        str, Optional[pd.DataFrame]]:
        """Синхронная версия получения multiple таймфреймов"""
        return _run_sync(self.get_multiple_timeframes(symbol, timeframes))


# Глобальные функции
_GLOBAL_CLIENT = None
_GLOBAL_LOCK = asyncio.Lock()

async def init_bybit_client():
    """Инициализация глобального клиента"""
    global _GLOBAL_CLIENT
    async with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT is None:
            _GLOBAL_CLIENT = UltraBybitClient()
        await _GLOBAL_CLIENT.init()
        logger.info("Global client initialized successfully")

async def close_bybit_client():
    """Закрытие глобального клиента"""
    global _GLOBAL_CLIENT
    async with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT and _GLOBAL_CLIENT._initialized:
            await _GLOBAL_CLIENT.close()
        _GLOBAL_CLIENT = None
        logger.info("Global client closed")

def get_client():
    """Получение экземпляра клиента"""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = UltraBybitClient()
    return _GLOBAL_CLIENT

def _run_sync(coroutine):
    """Запуск корутины в синхронном контексте"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result()
        else:
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        return asyncio.run(coroutine)

# Глобальные синхронные функции
def get_available_symbols(quote_coin: str = 'USDT') -> List[str]:
    """Глобальная функция получения символов"""
    client = get_client()
    return client.get_spot_symbols_sync(quote_coin)

def get_market_data(symbol: str, interval: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Глобальная функция получения свечных данных"""
    client = get_client()
    return client.get_klines_sync(symbol, interval, limit)

def get_ticker_info(symbol: str) -> Optional[Dict]:
    """Глобальная функция получения информации о тикере"""
    client = get_client()
    return client.get_ticker_info_sync(symbol)

def get_orderbook(symbol: str, limit: int = 25) -> Optional[Dict]:
    """Глобальная функция получения стакана"""
    client = get_client()
    return client.get_orderbook_sync(symbol, limit)

def get_multiple_timeframes(symbol: str, timeframes: List[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
    """Глобальная функция получения multiple таймфреймов"""
    client = get_client()
    return client.get_multiple_timeframes_sync(symbol, timeframes)

def get_batch_symbols_data(symbols: List[str], interval: str = '1h', limit: int = 100) -> Dict[str, Optional[pd.DataFrame]]:
    """Глобальная функция пакетного получения данных"""
    client = get_client()
    return client.get_batch_klines_sync(symbols, interval, limit)

# Алиасы для обратной совместимости
klines = get_market_data
get_bybit_symbols_sync = get_available_symbols
get_market_data_sync = get_market_data
get_ticker_info_sync = get_ticker_info
get_orderbook_sync = get_orderbook

# Автоматическая инициализация при импорте
async def _auto_init():
    try:
        await init_bybit_client()
    except Exception as e:
        logger.warning(f"Auto-init failed: {e}")

# Запускаем асинхронную инициализацию в фоне
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_auto_init())
    else:
        loop.run_until_complete(_auto_init())
except RuntimeError:
    pass

logger.info("UltraBybitClient loaded successfully")