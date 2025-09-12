import requests
import pandas as pd
import numpy as np
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class CryptoAPI:
    """Универсальный класс для работы с криптовалютными API"""

    def __init__(self):
        self.sources = {
            'bybit': 'https://api.bybit.com/v5',
            'binance': 'https://api.binance.com/api/v3'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.timeout = 10
        self.rate_limiter = threading.RLock()

    def get_ticker_info(self, symbol, source='bybit'):
        """Получение информации о тикере"""
        try:
            with self.rate_limiter:
                if source == 'bybit':
                    url = f"{self.sources['bybit']}/market/tickers"
                    params = {'category': 'spot', 'symbol': symbol}
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                        ticker = data['result']['list'][0]
                        return {
                            'lastPrice': float(ticker.get('lastPrice', 0)),
                            'price24hPcnt': float(ticker.get('price24hPcnt', 0)) * 100,
                            'volume24h': float(ticker.get('volume24h', 0)),
                            'highPrice24h': float(ticker.get('highPrice24h', 0)),
                            'lowPrice24h': float(ticker.get('lowPrice24h', 0))
                        }

                elif source == 'binance':
                    url = f"{self.sources['binance']}/ticker/24hr"
                    params = {'symbol': symbol}
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    return {
                        'lastPrice': float(data.get('lastPrice', 0)),
                        'price24hPcnt': (float(data.get('lastPrice', 0)) / float(data.get('openPrice', 1)) - 1) * 100,
                        'volume24h': float(data.get('volume', 0)),
                        'highPrice24h': float(data.get('highPrice', 0)),
                        'lowPrice24h': float(data.get('lowPrice', 0))
                    }

        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")

        return None

    def get_market_data(self, symbol, interval='1h', limit=500, source='bybit'):
        """Получение исторических данных"""
        try:
            with self.rate_limiter:
                interval_map = {
                    '1': '1', '5': '5', '15': '15', '30': '30',
                    '60': '60', '240': '240', 'D': 'D', 'W': 'W', 'M': 'M'
                }

                if source == 'bybit':
                    url = f"{self.sources['bybit']}/market/kline"
                    interval_minutes = interval_map.get(interval, '60')

                    params = {
                        'category': 'spot',
                        'symbol': symbol,
                        'interval': interval_minutes,
                        'limit': min(limit, 1000)
                    }

                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                        return self._process_bybit_data(data['result']['list'])

                elif source == 'binance':
                    binance_intervals = {
                        '1': '1m', '5': '5m', '15': '15m', '30': '30m',
                        '60': '1h', '240': '4h', 'D': '1d', 'W': '1w', 'M': '1M'
                    }

                    url = f"{self.sources['binance']}/klines"
                    params = {
                        'symbol': symbol,
                        'interval': binance_intervals.get(interval, '1h'),
                        'limit': min(limit, 1000)
                    }

                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    return self._process_binance_data(data)

        except Exception as e:
            logger.error(f"Ошибка получения исторических данных для {symbol}: {e}")

        return None

    def _process_bybit_data(self, candles):
        """Обработка данных от Bybit"""
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
        df = df.iloc[::-1].reset_index(drop=True)
        return df

    def _process_binance_data(self, candles):
        """Обработка данных от Binance"""
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_available_symbols(self, quote_coin='USDT', source='bybit'):
        """Получение списка доступных символов"""
        try:
            with self.rate_limiter:
                if source == 'bybit':
                    url = f"{self.sources['bybit']}/market/instruments-info"
                    params = {'category': 'spot'}

                    response = self.session.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    if data.get('retCode') == 0:
                        symbols = [
                            symbol['symbol'] for symbol in data['result']['list']
                            if (symbol['quoteCoin'] == quote_coin and
                                symbol['status'] == 'Trading' and
                                symbol['baseCoin'] not in ['USDC', 'BUSD', 'TUSD'])
                        ]
                        return sorted(symbols)

                elif source == 'binance':
                    url = f"{self.sources['binance']}/exchangeInfo"
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()

                    symbols = [
                        symbol['symbol'] for symbol in data.get('symbols', [])
                        if symbol['symbol'].endswith(quote_coin) and
                           symbol['status'] == 'TRADING'
                    ]
                    return sorted(symbols)

        except Exception as e:
            logger.error(f"Ошибка получения списка символов: {e}")

        # Fallback symbols
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT']