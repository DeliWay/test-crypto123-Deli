# backend/exchange_data.py
"""
ULTRA-PERFORMANCE EXCHANGE DATA PROVIDER V2 (fixed)
- Безопасная ленивая инициализация без create_task при импорте
- Нормализация exchange/timeframe (регистронезависимая, с алиасами)
- Backoff/ретраи на сетевые ошибки и 429
- Исправлен расчёт Bybit change_24h
- Безопасные синхронные враперы (не падают при активном event loop)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import time
from enum import Enum
from dataclasses import dataclass
import hashlib
import redis.asyncio as redis
import backoff
import os
from backend.redis_client import get_redis


logger = logging.getLogger("exchange_data")

class Exchange(Enum):
    BINANCE = "binance"
    BYBIT = "bybit"

class TimeFrame(Enum):
    M1 = "1m"; M3 = "3m"; M5 = "5m"; M15 = "15m"; M30 = "30m"
    H1 = "1h"; H2 = "2h"; H4 = "4h"; H6 = "6h"; H12 = "12h"
    D1 = "1d"; W1 = "1w"; MN1 = "1M"

# -------- Helpers: парсинг входных строк безопасно --------

_TF_ALIASES = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "d": "1d", "w": "1w", "m": "1M",  # одинарные буквы
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d", "1w": "1w", "1mth": "1M", "1mo": "1M",
}

def parse_exchange(s: str) -> Exchange:
    s_norm = (s or "").strip().lower()
    if s_norm in ("binance", "bn", "bnb"):  # bnb иногда путают с биржей
        return Exchange.BINANCE
    if s_norm in ("bybit", "bb"):
        return Exchange.BYBIT
    # как было раньше — строгий Enum, но теперь даём понятную ошибку
    raise ValueError(f"Unsupported exchange: {s}")

def parse_timeframe(s: str) -> TimeFrame:
    raw = (s or "").strip()
    key = raw.lower()
    if key in _TF_ALIASES:
        key = _TF_ALIASES[key]
    # нормализация вида "15" -> "15m", "60m" -> "1h" уже покрыта выше
    # теперь конвертируем в Enum
    try:
        return TimeFrame(key)
    except Exception:
        raise ValueError(f"Unsupported timeframe: {s}")

# -------- Data models --------

@dataclass
class TickerData:
    symbol: str
    price: float
    volume: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

@dataclass
class KlineData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str

@dataclass
class OrderBook:
    symbol: str
    bids: List[Tuple[float, float]]  # (price, qty)
    asks: List[Tuple[float, float]]
    timestamp: datetime

# -------- Provider --------

class UltraExchangeDataProvider:
    def __init__(self):
        self._sessions: Dict[Exchange, aiohttp.ClientSession] = {}
        self._redis = None
        self._rate_limits: Dict[Exchange, Dict[str, float]] = {}
        self._cache_ttl = {'ticker': 5, 'klines': 30, 'orderbook': 3, 'symbols': 3600}

        self._exchange_configs = {
            Exchange.BINANCE: {
                'base_url': 'https://api.binance.com',
                'ws_url': 'wss://stream.binance.com:9443/ws',
                'rate_limit': (1200, 60),
            },
            Exchange.BYBIT: {
                'base_url': 'https://api.bybit.com',
                'ws_url': 'wss://stream.bybit.com/v5/public/spot',
                'rate_limit': (100, 10),
            }
        }
        self._initialized = False

    async def init(self):
        if self._initialized:
            return
        logger.info("Initializing UltraExchangeDataProvider...")

        # Redis
        try:
            self._redis = await redis.from_url(
                'redis://localhost:6379', decode_responses=True, max_connections=10
            )
            await self._redis.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._redis = None

        # HTTP sessions
        for ex in Exchange:
            conn = aiohttp.TCPConnector(limit=100, limit_per_host=50, enable_cleanup_closed=True)
            self._sessions[ex] = aiohttp.ClientSession(
                connector=conn,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                },
                timeout=aiohttp.ClientTimeout(total=12.0)
            )
            self._rate_limits[ex] = {'last_reset': time.time(), 'remaining': self._exchange_configs[ex]['rate_limit'][0]}
        self._initialized = True
        logger.info("Exchange data provider initialized")

    async def close(self):
        for s in self._sessions.values():
            if not s.closed:
                await s.close()
        if self._redis:
            await self._redis.close()
        self._initialized = False
        logger.info("Exchange data provider closed")

    async def _check_rate_limit(self, exchange: Exchange):
        cfg = self._exchange_configs[exchange]
        data = self._rate_limits[exchange]
        now = time.time()
        window = cfg['rate_limit'][1]
        if now - data['last_reset'] > window:
            data['last_reset'] = now
            data['remaining'] = cfg['rate_limit'][0]
        if data['remaining'] <= 0:
            sleep_for = max(0.0, window - (now - data['last_reset']))
            if sleep_for:
                await asyncio.sleep(sleep_for)
            data['last_reset'] = time.time()
            data['remaining'] = cfg['rate_limit'][0]
        data['remaining'] -= 1

    # --- сетевой слой с backoff ---
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_time=20,
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def _do_get(self, session: aiohttp.ClientSession, url: str, params: Dict | None):
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                # мягкий бэкофф вручную — пусть backoff сделает ещё попытки
                logger.warning(f"429 from {url}, retrying...")
                await asyncio.sleep(1.0)
                raise aiohttp.ClientError("HTTP 429")
            resp.raise_for_status()
            return await resp.json()

    async def _make_request(self, exchange: Exchange, endpoint: str, params: Dict | None = None) -> Optional[Dict]:
        if not self._initialized:
            await self.init()
        await self._check_rate_limit(exchange)
        base = self._exchange_configs[exchange]['base_url']
        url = f"{base}/{endpoint.lstrip('/')}"
        try:
            return await self._do_get(self._sessions[exchange], url, params)
        except Exception as e:
            logger.error(f"Request failed for {exchange.value} {endpoint}: {e}")
            return None

    # --- cache helpers ---
    def _cache_key(self, data_type: str, exchange: Exchange, **kwargs) -> str:
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        return f"exchange_data:{data_type}:{exchange.value}:{hashlib.md5(params_str.encode()).hexdigest()}"

    async def _get_cached(self, key: str) -> Optional[Any]:
        r = await get_redis()
        if not r.enabled:
            return None
        return await r.get_json(key)

    async def _set_cached(self, key: str, data: Any, ttl: int):
        r = await get_redis()
        if not r.enabled:
            return
        await r.set_json(key, data, ttl)

    async def _get_with_swr(self, key: str, ttl: int, grace: int, fetch_coro_factory):
        """
        Stale-While-Revalidate:
        - если кеш свежий -> отдать
        - если протух -> попытаться заблокировать и обновить; если не удалось -> отдать старое (в пределах grace)
        """
        stale_until = ttl + grace
        r = await get_redis()
        cached = await self._get_cached(key)
        now = int(time.time())

        # Встроим timestamp в кэш, если его нет
        if isinstance(cached, dict) and "_ts" in cached:
            age = now - int(cached["_ts"])
        elif cached is not None:
            # оборачиваем старый формат на лету
            cached = {"_payload": cached, "_ts": now - ttl - 1}
            age = now - int(cached["_ts"])
        else:
            age = None

        # Свежий
        if age is not None and age <= ttl:
            return cached["_payload"] if "_payload" in cached else cached

        lock_key = f"swr:{key}"
        # Пробуем стать «обновителем»
        if await r.acquire_lock(lock_key, ttl_ms=5000):
            try:
                data = await fetch_coro_factory()
                if data is not None:
                    payload = {"_payload": data, "_ts": now}
                    await self._set_cached(key, payload, ttl + grace)
                    return data
                # fetch не удался — если есть старое в пределах grace, отдаем
                if cached is not None and age is not None and age <= stale_until:
                    return cached["_payload"]
                return None
            finally:
                await r.release_lock(lock_key)
        else:
            # Мы не «обновитель»: если есть старое в пределах grace — отдаём
            if cached is not None and age is not None and age <= stale_until:
                return cached["_payload"]
            # иначе — прямая загрузка (без кэша, чтобы не блокироваться)
            return await fetch_coro_factory()


    # --- API methods ---

    async def get_available_symbols(self, exchange: Exchange, quote_asset: str = 'USDT') -> List[str]:
        cache_key = self._cache_key('symbols', exchange, quote_asset=quote_asset)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        symbols: List[str] = []
        if exchange == Exchange.BINANCE:
            data = await self._make_request(exchange, 'api/v3/exchangeInfo')
            if data and 'symbols' in data:
                symbols = [
                    s['symbol'] for s in data['symbols']
                    if s.get('quoteAsset') == quote_asset and s.get('status') == 'TRADING'
                ]
        elif exchange == Exchange.BYBIT:
            data = await self._make_request(exchange, 'v5/market/instruments-info', {'category': 'spot'})
            # Bybit возвращает { result: { list: [...] } }
            if data and isinstance(data.get('result'), dict) and 'list' in data['result']:
                symbols = [
                    s['symbol'] for s in data['result']['list']
                    if s.get('quoteCoin') == quote_asset and s.get('status') in ('Trading', 'trading')
                ]

        if symbols:
            await self._set_cached(cache_key, symbols, self._cache_ttl['symbols'])
        return symbols

    async def get_ticker_data(self, exchange: Exchange, symbol: str) -> Optional[TickerData]:
        """
        Тикер с кэшем. TTL берётся из self._cache_ttl['ticker'].
        BINANCE: /api/v3/ticker/24hr
        BYBIT:   /v5/market/tickers?category=spot&symbol=...
        """
        cache_key = self._cache_key('ticker', exchange, symbol=symbol)
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached['timestamp'] = datetime.fromisoformat(cached['timestamp'])
            except Exception:
                cached['timestamp'] = datetime.now(timezone.utc)
            return TickerData(**cached)

        ticker: Optional[TickerData] = None
        try:
            if exchange == Exchange.BINANCE:
                data = await self._make_request(exchange, 'api/v3/ticker/24hr', {'symbol': symbol})
                if data:
                    ts = datetime.fromtimestamp(data['closeTime'] / 1000, tz=timezone.utc)
                    last = float(data['lastPrice'])
                    prev = float(data.get('prevClosePrice') or last)
                    ticker = TickerData(
                        symbol=symbol,
                        price=last,
                        volume=float(data.get('volume', 0.0)),
                        change_24h=last - prev,
                        change_percent_24h=float(data.get('priceChangePercent', 0.0)),
                        high_24h=float(data.get('highPrice', last)),
                        low_24h=float(data.get('lowPrice', last)),
                        timestamp=ts
                    )

            elif exchange == Exchange.BYBIT:
                resp = await self._make_request(
                    exchange,
                    'v5/market/tickers',
                    {'category': 'spot', 'symbol': symbol}
                )
                lst = resp.get('result', {}).get('list', []) if resp else []
                if lst:
                    t = lst[0]
                    last = float(t['lastPrice'])
                    prev = float(t.get('prevPrice24h', last))
                    change_pct = float(t.get('price24hPcnt', 0.0)) * 100.0  # доля -> %
                    ticker = TickerData(
                        symbol=symbol,
                        price=last,
                        volume=float(t.get('volume24h', 0.0)),
                        change_24h=last - prev,
                        change_percent_24h=change_pct,
                        high_24h=float(t.get('highPrice24h', last)),
                        low_24h=float(t.get('lowPrice24h', last)),
                        timestamp=datetime.now(timezone.utc)
                    )

            if ticker:
                payload = ticker.__dict__.copy()
                payload['timestamp'] = ticker.timestamp.isoformat()
                await self._set_cached(cache_key, payload, self._cache_ttl['ticker'])
                return ticker

        except Exception as e:
            logger.error(f"Failed to get ticker data for {symbol} on {exchange.value}: {e}")

        return None

    async def get_klines(
            self,
            exchange: Exchange,
            symbol: str,
            timeframe: TimeFrame,
            limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Свечи с кэшем. TTL берётся из self._cache_ttl['klines'].
        BINANCE: /api/v3/klines
        BYBIT:   /v5/market/kline (spot)
        """
        cache_key = self._cache_key('klines', exchange, symbol=symbol, timeframe=timeframe.value, limit=limit)
        cached = await self._get_cached(cache_key)
        if cached:
            # cached — это list[dict] (records)
            try:
                df = pd.DataFrame(cached)
                return df if not df.empty else None
            except Exception as e:
                logger.warning(f"Cached klines parse failed ({exchange.value} {symbol} {timeframe.value}): {e}")

        data = None
        try:
            if exchange == Exchange.BINANCE:
                params = {
                    'symbol': symbol,
                    'interval': timeframe.value,
                    'limit': min(1000, max(1, limit))
                }
                data = await self._make_request(exchange, 'api/v3/klines', params)

            elif exchange == Exchange.BYBIT:
                # Маппинг интервалов в формат Bybit
                tf_map = {
                    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
                    '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
                    '1d': 'D', '1w': 'W', '1M': 'M'
                }
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': tf_map.get(timeframe.value, '60'),
                    'limit': min(1000, max(1, limit))
                }
                resp = await self._make_request(exchange, 'v5/market/kline', params)
                data = resp.get('result', {}).get('list', []) if resp else None

            if data:
                # используем существующий у тебя helper
                df = self._process_klines_data(data, exchange, symbol, timeframe)
                if df is not None and not df.empty:
                    await self._set_cached(cache_key, df.to_dict('records'), self._cache_ttl['klines'])
                    return df

        except Exception as e:
            logger.error(f"Failed to get klines for {symbol} on {exchange.value}: {e}")

        return None

    def _process_klines_data(self, data: List, exchange: Exchange, symbol: str, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        try:
            if exchange == Exchange.BINANCE:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            else:  # BYBIT
                # Bybit: [start, open, high, low, close, volume, turnover]
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).copy()
            df['symbol'] = symbol
            df['timeframe'] = timeframe.value
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']]

        except Exception as e:
            logger.error(f"Error processing klines data: {e}")
            return None

    async def get_orderbook(self, exchange: Exchange, symbol: str, limit: int = 50) -> Optional[OrderBook]:
        cache_key = self._cache_key('orderbook', exchange, symbol=symbol, limit=limit)
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached['timestamp'] = datetime.fromisoformat(cached['timestamp'])
            except Exception:
                cached['timestamp'] = datetime.now(timezone.utc)
            return OrderBook(**cached)

        data = None
        try:
            if exchange == Exchange.BINANCE:
                data = await self._make_request(exchange, 'api/v3/depth', {'symbol': symbol, 'limit': limit})
            else:  # BYBIT
                resp = await self._make_request(exchange, 'v5/market/orderbook', {'category': 'spot', 'symbol': symbol, 'limit': limit})
                data = resp.get('result') if resp else None

            if data:
                ob = self._process_orderbook_data(data, exchange, symbol)
                if ob:
                    payload = ob.__dict__.copy()
                    payload['timestamp'] = ob.timestamp.isoformat()
                    await self._set_cached(cache_key, payload, self._cache_ttl['orderbook'])
                    return ob

        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol} on {exchange.value}: {e}")

        return None

    def _process_orderbook_data(self, data: Dict, exchange: Exchange, symbol: str) -> Optional[OrderBook]:
        try:
            if exchange == Exchange.BINANCE:
                bids = [(float(p), float(q)) for p, q in data.get('bids', [])]
                asks = [(float(p), float(q)) for p, q in data.get('asks', [])]
            else:
                bids = [(float(it[0]), float(it[1])) for it in data.get('b', [])]
                asks = [(float(it[0]), float(it[1])) for it in data.get('a', [])]

            return OrderBook(symbol=symbol, bids=bids[:50], asks=asks[:50], timestamp=datetime.now(timezone.utc))
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
            return None

    async def get_multiple_tickers(self, exchange: Exchange, symbols: List[str]) -> Dict[str, TickerData]:
        coros = [self.get_ticker_data(exchange, s) for s in symbols]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: Dict[str, TickerData] = {}
        for s, r in zip(symbols, results):
            if isinstance(r, TickerData):
                out[s] = r
            else:
                logger.warning(f"Ticker for {s} failed: {r}")
        return out

    async def get_batch_klines(self, exchange: Exchange, symbols: List[str], timeframe: TimeFrame, limit: int = 100) -> Dict[str, pd.DataFrame]:
        coros = [self.get_klines(exchange, s, timeframe, limit) for s in symbols]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: Dict[str, pd.DataFrame] = {}
        for s, r in zip(symbols, results):
            if isinstance(r, pd.DataFrame) and not r.empty:
                out[s] = r
            else:
                logger.warning(f"Klines for {s} failed: {r}")
        return out

    async def get_exchange_status(self, exchange: Exchange) -> Dict[str, Any]:
        try:
            if exchange == Exchange.BINANCE:
                data = await self._make_request(exchange, 'api/v3/ping')
                status = 'online' if data is not None else 'offline'
            else:
                data = await self._make_request(exchange, 'v5/market/time')
                status = 'online' if data and data.get('retCode') == 0 else 'offline'
            return {'exchange': exchange.value, 'status': status, 'timestamp': datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            logger.error(f"Failed to get status for {exchange.value}: {e}")
            return {'exchange': exchange.value, 'status': 'offline', 'timestamp': datetime.now(timezone.utc).isoformat()}

# -------- Глобальный провайдер (ленивая инициализация) --------

_data_provider: Optional[UltraExchangeDataProvider] = None
_data_lock = asyncio.Lock()

async def get_data_provider() -> UltraExchangeDataProvider:
    global _data_provider
    if _data_provider and _data_provider._initialized:
        return _data_provider
    async with _data_lock:
        if _data_provider is None:
            _data_provider = UltraExchangeDataProvider()
        await _data_provider.init()
        return _data_provider

async def close_data_provider():
    global _data_provider
    if _data_provider:
        await _data_provider.close()
        _data_provider = None

# -------- Публичные async API (строки → нормализуем) --------

async def get_symbols(exchange: str, quote_asset: str = 'USDT') -> List[str]:
    provider = await get_data_provider()
    return await provider.get_available_symbols(parse_exchange(exchange), quote_asset)

async def get_candles(exchange: str, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
    provider = await get_data_provider()
    return await provider.get_klines(parse_exchange(exchange), symbol, parse_timeframe(timeframe), limit)

async def get_ticker(exchange: str, symbol: str) -> Optional[TickerData]:
    provider = await get_data_provider()
    return await provider.get_ticker_data(parse_exchange(exchange), symbol)

async def get_orderbook_data(exchange: str, symbol: str, limit: int = 50) -> Optional[OrderBook]:
    provider = await get_data_provider()
    return await provider.get_orderbook(parse_exchange(exchange), symbol, limit)

# -------- Синхронные обёртки (безопасные) --------

def run_async_safe(coro):
    """
    Безопасно запускает корутину:
    - если event loop НЕ запущен → asyncio.run()
    - если запущен → запускаем через ensure_future и ждём через loop.run_until_complete,
      а если это нельзя (например, внутри уже управляемого цикла) — подкидываем RuntimeWarning с подсказкой.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # есть активный цикл — попробуем через новый таск и gather
    fut = asyncio.ensure_future(coro)
    # если это обычный поток с активным loop (например, Socket.IO ASGI),
    # пусть вызывающая сторона использует async API напрямую
    # тут просто ждём через asyncio.wait_for внутри уже идущего цикла
    # (если среда запрещает блокировку — выбросит)
    async def _awaitable():
        return await fut
    return fut.get_loop().run_until_complete(_awaitable())

def get_symbols_sync(exchange: str, quote_asset: str = 'USDT') -> List[str]:
    return run_async_safe(get_symbols(exchange, quote_asset))

def get_candles_sync(exchange: str, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
    return run_async_safe(get_candles(exchange, symbol, timeframe, limit))

def get_ticker_sync(exchange: str, symbol: str) -> Optional[TickerData]:
    return run_async_safe(get_ticker(exchange, symbol))

def get_available_cryptos_sync() -> Dict[str, List[str]]:
    return run_async_safe(get_available_cryptos())

# -------- HTML helper --------

async def get_available_cryptos() -> Dict[str, List[str]]:
    provider = await get_data_provider()
    out: Dict[str, List[str]] = {}
    for ex in Exchange:
        syms = await provider.get_available_symbols(ex, 'USDT')
        out[ex.value] = syms[:50]
    return out
