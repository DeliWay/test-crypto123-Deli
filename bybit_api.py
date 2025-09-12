import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
import threading
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.calls = deque()
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

            self.calls.append(now)


class BybitAPIClient:
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.version = "v5"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1

        # Rate limiting
        self.rate_limiter = RateLimiter(5, 1)  # 5 requests per second
        self.daily_limiter = RateLimiter(1000, 86400)  # 1000 requests per day

        # Cache
        self.symbols_cache = None
        self.symbols_cache_time = None
        self.cache_timeout = 300  # 5 minutes

    def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Optional[Dict]:
        """Make HTTP request with retry logic and rate limiting"""
        url = f"{self.base_url}/{self.version}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                self.daily_limiter.wait()

                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=params, timeout=self.timeout)

                response.raise_for_status()
                data = response.json()

                if data.get('retCode') == 0:
                    return data
                else:
                    logger.warning(f"API error: {data.get('retMsg', 'Unknown error')}")
                    if attempt == self.max_retries - 1:
                        return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

        return None

    def get_spot_symbols(self, quote_coin: str = 'USDT') -> List[str]:
        """Get all available spot trading symbols"""
        try:
            # Check cache first
            current_time = time.time()
            if (self.symbols_cache and self.symbols_cache_time and
                    current_time - self.symbols_cache_time < self.cache_timeout):
                return self.symbols_cache

            data = self._make_request('market/instruments-info', {'category': 'spot'})
            if not data or 'result' not in data:
                return self._get_default_symbols()

            symbols_list = data['result'].get('list', [])
            symbols = [
                symbol['symbol'] for symbol in symbols_list
                if (symbol.get('quoteCoin') == quote_coin and
                    symbol.get('status') == 'Trading' and
                    symbol.get('baseCoin') not in ['USDC', 'BUSD', 'TUSD', 'USDC'])
            ]

            symbols = sorted(symbols)

            # Update cache
            self.symbols_cache = symbols
            self.symbols_cache_time = current_time

            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return self._get_default_symbols()

    def _get_default_symbols(self) -> List[str]:
        """Fallback symbols in case of API failure"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
            'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'XLMUSDT'
        ]

    def get_klines(self, symbol: str, interval: str = '60', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol"""
        try:
            interval_map = {
                '1': '1', '5': '5', '15': '15', '30': '30',
                '60': '60', '240': '240', 'D': 'D', 'W': 'W', 'M': 'M'
            }

            interval_str = interval_map.get(interval, '60')
            end_time = int(time.time() * 1000)
            start_time = end_time - (90 * 24 * 60 * 60 * 1000)  # 90 days

            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': interval_str,
                'start': start_time,
                'end': end_time,
                'limit': min(limit, 1000)
            }

            data = self._make_request('market/kline', params)
            if not data or 'result' not in data or not data['result'].get('list'):
                return None

            candles = data['result']['list']
            df = self._process_klines_data(candles)

            if df is None or df.empty:
                return None

            logger.info(f"Retrieved {len(df)} klines for {symbol} ({interval_str})")
            return df

        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return None

    def _process_klines_data(self, candles: List) -> Optional[pd.DataFrame]:
        """Process raw klines data into DataFrame"""
        try:
            if not candles:
                return None

            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values
            df = df.dropna()

            if df.empty:
                return None

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')

            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate additional metrics
            df['price_change'] = df['close'].pct_change() * 100
            df['price_change_abs'] = df['close'].diff()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

            return df

        except Exception as e:
            logger.error(f"Error processing klines data: {e}")
            return None

    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Get orderbook data for a symbol"""
        try:
            params = {
                'category': 'spot',
                'symbol': symbol,
                'limit': min(limit, 50)
            }

            data = self._make_request('market/orderbook', params)
            if not data or 'result' not in data:
                return None

            orderbook = data['result']
            return self._process_orderbook_data(orderbook)

        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return None

    def _process_orderbook_data(self, orderbook: Dict) -> Dict:
        """Process orderbook data"""
        try:
            bids = pd.DataFrame(orderbook.get('b', []), columns=['price', 'quantity'])
            asks = pd.DataFrame(orderbook.get('a', []), columns=['price', 'quantity'])

            for df in [bids, asks]:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                df.dropna(inplace=True)

            bid_volume = bids['quantity'].sum()
            ask_volume = asks['quantity'].sum()
            spread = asks['price'].iloc[0] - bids['price'].iloc[0] if not asks.empty and not bids.empty else 0

            return {
                'bids': bids.to_dict('records'),
                'asks': asks.to_dict('records'),
                'metrics': {
                    'bid_volume': float(bid_volume),
                    'ask_volume': float(ask_volume),
                    'total_volume': float(bid_volume + ask_volume),
                    'spread': float(spread),
                    'spread_percent': float(
                        (spread / bids['price'].iloc[0] * 100) if not bids.empty and bids['price'].iloc[0] > 0 else 0),
                    'orderbook_imbalance': float((bid_volume - ask_volume) / (bid_volume + ask_volume) * 100) if (
                                                                                                                             bid_volume + ask_volume) > 0 else 0
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            return None

    def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """Get ticker information for a symbol"""
        try:
            params = {
                'category': 'spot',
                'symbol': symbol
            }

            data = self._make_request('market/tickers', params)
            if not data or 'result' not in data or not data['result'].get('list'):
                return None

            ticker_data = data['result']['list'][0]
            return self._enhance_ticker_data(ticker_data)

        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None

    def _enhance_ticker_data(self, ticker: Dict) -> Dict:
        """Enhance ticker data with additional calculations"""
        try:
            numeric_fields = [
                'lastPrice', 'highPrice24h', 'lowPrice24h', 'prevPrice24h',
                'price24hPcnt', 'volume24h', 'turnover24h', 'usdIndexPrice'
            ]

            enhanced = {}
            for field in numeric_fields:
                if field in ticker and ticker[field]:
                    try:
                        enhanced[field] = float(ticker[field])
                    except (ValueError, TypeError):
                        enhanced[field] = 0.0
                else:
                    enhanced[field] = 0.0

            # Calculate additional metrics
            enhanced['price_change_24h_abs'] = enhanced['lastPrice'] - enhanced['prevPrice24h']
            enhanced['average_price_24h'] = enhanced['turnover24h'] / enhanced['volume24h'] if enhanced[
                                                                                                   'volume24h'] > 0 else \
            enhanced['lastPrice']
            enhanced['price_change_24h_percent'] = enhanced['price24hPcnt'] * 100

            # Add original data
            enhanced.update({k: v for k, v in ticker.items() if k not in numeric_fields})

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing ticker data: {e}")
            return ticker

    def get_multiple_timeframes(self, symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        if timeframes is None:
            timeframes = ['60', '240', '1440']  # 1h, 4h, 1d

        results = {}
        for tf in timeframes:
            try:
                data = self.get_klines(symbol, tf, 200)
                if data is not None:
                    results[tf] = data
                    logger.info(f"Retrieved {tf} data for {symbol}: {len(data)} candles")
                else:
                    logger.warning(f"Failed to get {tf} data for {symbol}")
            except Exception as e:
                logger.error(f"Error getting {tf} data for {symbol}: {e}")
                continue

        return results

    def get_batch_symbols_data(self, symbols: List[str], interval: str = '60', limit: int = 100) -> Dict[
        str, pd.DataFrame]:
        """Get data for multiple symbols in batch"""
        results = {}

        for symbol in symbols:
            try:
                data = self.get_klines(symbol, interval, limit)
                if data is not None:
                    results[symbol] = data
                # Respect rate limits
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                continue

        return results

    def get_server_time(self) -> Optional[Dict]:
        """Get server time for synchronization"""
        try:
            data = self._make_request('market/time')
            if data and 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return None

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Get funding rate for perpetual contracts"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol
            }
            data = self._make_request('market/funding/history', params)
            if data and 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {e}")
            return None


# Global instance for import
bybit_client = BybitAPIClient()


# Compatibility functions
def get_bybit_symbols(quote_coin: str = 'USDT') -> List[str]:
    return bybit_client.get_spot_symbols(quote_coin)


def get_market_data(symbol: str, interval: str = '60', limit: int = 500) -> Optional[pd.DataFrame]:
    return bybit_client.get_klines(symbol, interval, limit)


def get_liquidity_data(symbol: str, limit: int = 25) -> Optional[Dict]:
    return bybit_client.get_orderbook(symbol, limit)


def get_ticker_info(symbol: str) -> Optional[Dict]:
    return bybit_client.get_ticker_info(symbol)


def get_multiple_timeframes(symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    return bybit_client.get_multiple_timeframes(symbol, timeframes)


def get_batch_symbols_data(symbols: List[str], interval: str = '60', limit: int = 100) -> Dict[str, pd.DataFrame]:
    return bybit_client.get_batch_symbols_data(symbols, interval, limit)