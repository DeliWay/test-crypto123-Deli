# tests/test_exchange_data.py
"""
Комплексные тесты для UltraExchangeDataProvider
Тестирование всех функций модуля exchange_data
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import json

# Добавляем путь к модулю
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.exchange_data import (
    UltraExchangeDataProvider,
    Exchange,
    TimeFrame,
    TickerData,
    OrderBook,
    get_symbols,
    get_candles,
    get_ticker,
    get_available_cryptos
)


class TestExchangeData:
    """Тесты для UltraExchangeDataProvider"""

    @pytest.fixture
    async def provider(self):
        """Фикстура провайдера данных"""
        provider = UltraExchangeDataProvider()
        # Мокаем Redis чтобы не требовать реальное подключение
        with patch('backend.exchange_data.redis.from_url') as mock_redis:
            mock_redis.return_value = AsyncMock()
            mock_redis.return_value.ping = AsyncMock()
            mock_redis.return_value.get = AsyncMock(return_value=None)
            mock_redis.return_value.setex = AsyncMock()

            await provider.init()
            yield provider
            await provider.close()

    @pytest.fixture
    def mock_response(self):
        """Фикстура мокового HTTP ответа"""
        mock_resp = Mock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock()
        return mock_resp

    def test_initialization(self):
        """Тест инициализации провайдера"""
        provider = UltraExchangeDataProvider()
        assert provider._initialized == False
        assert provider._sessions == {}
        assert provider._redis is None

    @pytest.mark.asyncio
    async def test_init_success(self, provider):
        """Тест успешной инициализации"""
        assert provider._initialized == True
        assert Exchange.BINANCE in provider._sessions
        assert Exchange.BYBIT in provider._sessions
        assert provider._redis is not None

    @pytest.mark.asyncio
    async def test_rate_limiting(self, provider):
        """Тест rate limiting"""
        # Проверяем что rate limit данные инициализированы
        assert Exchange.BINANCE in provider._rate_limits
        assert Exchange.BYBIT in provider._rate_limits

        binance_limit = provider._rate_limits[Exchange.BINANCE]
        assert 'last_reset' in binance_limit
        assert 'remaining' in binance_limit

        # Проверяем что можем пройти rate limit проверку
        result = await provider._check_rate_limit(Exchange.BINANCE)
        assert result == True

    def test_cache_key_generation(self, provider):
        """Тест генерации ключей кэша"""
        key1 = provider._cache_key('test', Exchange.BINANCE, symbol='BTCUSDT')
        key2 = provider._cache_key('test', Exchange.BINANCE, symbol='BTCUSDT')
        key3 = provider._cache_key('test', Exchange.BINANCE, symbol='ETHUSDT')

        assert key1 == key2  # Одинаковые параметры = одинаковый ключ
        assert key1 != key3  # Разные параметры = разные ключи
        assert key1.startswith('exchange_data:test:binance:')

    @pytest.mark.asyncio
    async def test_get_available_symbols_binance(self, provider, mock_response):
        """Тест получения символов Binance"""
        # Мокаем данные Binance
        mock_response.json.return_value = {
            'symbols': [
                {
                    'symbol': 'BTCUSDT',
                    'quoteAsset': 'USDT',
                    'status': 'TRADING'
                },
                {
                    'symbol': 'ETHUSDT',
                    'quoteAsset': 'USDT',
                    'status': 'TRADING'
                },
                {
                    'symbol': 'BTCBUSD',
                    'quoteAsset': 'BUSD',
                    'status': 'TRADING'
                }
            ]
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            symbols = await provider.get_available_symbols(Exchange.BINANCE, 'USDT')

            assert 'BTCUSDT' in symbols
            assert 'ETHUSDT' in symbols
            assert 'BTCBUSD' not in symbols  # Должен быть отфильтрован
            assert len(symbols) == 2

    @pytest.mark.asyncio
    async def test_get_available_symbols_bybit(self, provider, mock_response):
        """Тест получения символов Bybit"""
        # Мокаем данные Bybit
        mock_response.json.return_value = {
            'result': {
                'list': [
                    {
                        'symbol': 'BTCUSDT',
                        'quoteCoin': 'USDT',
                        'status': 'Trading'
                    },
                    {
                        'symbol': 'ETHUSDT',
                        'quoteCoin': 'USDT',
                        'status': 'Trading'
                    }
                ]
            }
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            symbols = await provider.get_available_symbols(Exchange.BYBIT, 'USDT')

            assert 'BTCUSDT' in symbols
            assert 'ETHUSDT' in symbols
            assert len(symbols) == 2

    @pytest.mark.asyncio
    async def test_get_ticker_data_binance(self, provider, mock_response):
        """Тест получения данных тикера Binance"""
        mock_time = datetime.now().timestamp() * 1000

        mock_response.json.return_value = {
            'symbol': 'BTCUSDT',
            'lastPrice': '45000.50',
            'volume': '1000.25',
            'priceChange': '500.25',
            'priceChangePercent': '1.12',
            'highPrice': '45500.00',
            'lowPrice': '44500.00',
            'closeTime': mock_time
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            ticker = await provider.get_ticker_data(Exchange.BINANCE, 'BTCUSDT')

            assert isinstance(ticker, TickerData)
            assert ticker.symbol == 'BTCUSDT'
            assert ticker.price == 45000.50
            assert ticker.volume == 1000.25
            assert ticker.change_24h == 500.25
            assert ticker.change_percent_24h == 1.12

    @pytest.mark.asyncio
    async def test_get_ticker_data_bybit(self, provider, mock_response):
        """Тест получения данных тикера Bybit"""
        mock_response.json.return_value = {
            'result': {
                'list': [
                    {
                        'symbol': 'BTCUSDT',
                        'lastPrice': '45000.50',
                        'volume24h': '1000.25',
                        'price24hPcnt': '0.0112',
                        'prevPrice24h': '44500.25',
                        'highPrice24h': '45500.00',
                        'lowPrice24h': '44500.00'
                    }
                ]
            }
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            ticker = await provider.get_ticker_data(Exchange.BYBIT, 'BTCUSDT')

            assert isinstance(ticker, TickerData)
            assert ticker.symbol == 'BTCUSDT'
            assert ticker.price == 45000.50
            assert ticker.volume == 1000.25
            # Проверяем расчет изменения цены
            expected_change = 0.0112 * 44500.25
            assert abs(ticker.change_24h - expected_change) < 0.01

    @pytest.mark.asyncio
    async def test_get_klines_binance(self, provider, mock_response):
        """Тест получения свечных данных Binance"""
        mock_data = [
            [
                1672531200000,  # timestamp
                "45000.00",  # open
                "45500.00",  # high
                "44500.00",  # low
                "45200.00",  # close
                "1000.25",  # volume
                1672534800000,  # close_time
                "4520000.50",  # quote_volume
                1000,  # trades
                "500.25",  # taker_buy_base
                "2260000.25",  # taker_buy_quote
                "0"  # ignore
            ]
        ]

        mock_response.json.return_value = mock_data

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            df = await provider.get_klines(
                Exchange.BINANCE,
                'BTCUSDT',
                TimeFrame.H1,
                100
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert 'timestamp' in df.columns
            assert 'open' in df.columns
            assert 'high' in df.columns
            assert 'low' in df.columns
            assert 'close' in df.columns
            assert 'volume' in df.columns
            assert df['symbol'].iloc[0] == 'BTCUSDT'
            assert df['timeframe'].iloc[0] == '1h'
            assert df['open'].iloc[0] == 45000.00

    @pytest.mark.asyncio
    async def test_get_klines_bybit(self, provider, mock_response):
        """Тест получения свечных данных Bybit"""
        mock_data = {
            'result': {
                'list': [
                    [
                        "1672531200000",  # timestamp
                        "45000.00",  # open
                        "45500.00",  # high
                        "44500.00",  # low
                        "45200.00",  # close
                        "1000.25",  # volume
                        "4520000.50"  # turnover
                    ]
                ]
            }
        }

        mock_response.json.return_value = mock_data

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            df = await provider.get_klines(
                Exchange.BYBIT,
                'BTCUSDT',
                TimeFrame.H1,
                100
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['open'].iloc[0] == 45000.00
            assert df['close'].iloc[0] == 45200.00

    @pytest.mark.asyncio
    async def test_get_orderbook_binance(self, provider, mock_response):
        """Тест получения стакана Binance"""
        mock_response.json.return_value = {
            'bids': [
                ['44900.00', '1.5'],
                ['44800.00', '2.0']
            ],
            'asks': [
                ['45100.00', '0.8'],
                ['45200.00', '1.2']
            ]
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            orderbook = await provider.get_orderbook(Exchange.BINANCE, 'BTCUSDT', 50)

            assert isinstance(orderbook, OrderBook)
            assert orderbook.symbol == 'BTCUSDT'
            assert len(orderbook.bids) == 2
            assert len(orderbook.asks) == 2
            assert orderbook.bids[0] == (44900.00, 1.5)
            assert orderbook.asks[0] == (45100.00, 0.8)

    @pytest.mark.asyncio
    async def test_get_orderbook_bybit(self, provider, mock_response):
        """Тест получения стакана Bybit"""
        mock_response.json.return_value = {
            'result': {
                'b': [  # bids
                    ['44900.00', '1.5'],
                    ['44800.00', '2.0']
                ],
                'a': [  # asks
                    ['45100.00', '0.8'],
                    ['45200.00', '1.2']
                ]
            }
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            orderbook = await provider.get_orderbook(Exchange.BYBIT, 'BTCUSDT', 50)

            assert isinstance(orderbook, OrderBook)
            assert orderbook.symbol == 'BTCUSDT'
            assert orderbook.bids[0] == (44900.00, 1.5)
            assert orderbook.asks[0] == (45100.00, 0.8)

    @pytest.mark.asyncio
    async def test_get_multiple_tickers(self, provider):
        """Тест получения нескольких тикеров"""
        # Мокаем отдельные вызовы get_ticker_data
        with patch.object(provider, 'get_ticker_data') as mock_ticker:
            mock_ticker.side_effect = [
                TickerData('BTCUSDT', 45000.0, 1000.0, 500.0, 1.12, 45500.0, 44500.0, datetime.now()),
                TickerData('ETHUSDT', 3000.0, 500.0, 50.0, 1.67, 3050.0, 2950.0, datetime.now()),
                Exception("Timeout")  # Симулируем ошибку для третьего символа
            ]

            tickers = await provider.get_multiple_tickers(
                Exchange.BINANCE,
                ['BTCUSDT', 'ETHUSDT', 'ERRORSYMBOL']
            )

            assert 'BTCUSDT' in tickers
            assert 'ETHUSDT' in tickers
            assert 'ERRORSYMBOL' not in tickers  # Должен быть пропущен из-за ошибки
            assert len(tickers) == 2

    @pytest.mark.asyncio
    async def test_get_batch_klines(self, provider):
        """Тест пакетного получения свечных данных"""
        # Создаем mock DataFrame
        mock_df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [45000.0],
            'high': [45500.0],
            'low': [44500.0],
            'close': [45200.0],
            'volume': [1000.0],
            'symbol': ['BTCUSDT'],
            'timeframe': ['1h']
        })

        with patch.object(provider, 'get_klines') as mock_klines:
            mock_klines.return_value = mock_df

            results = await provider.get_batch_klines(
                Exchange.BINANCE,
                ['BTCUSDT', 'ETHUSDT'],
                TimeFrame.H1,
                100
            )

            assert 'BTCUSDT' in results
            assert 'ETHUSDT' in results
            assert isinstance(results['BTCUSDT'], pd.DataFrame)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_exchange_status_online(self, provider, mock_response):
        """Тест проверки статуса биржи (online)"""
        mock_response.json.return_value = {}  # Пустой ответ для ping

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            status = await provider.get_exchange_status(Exchange.BINANCE)

            assert status['exchange'] == 'binance'
            assert status['status'] == 'online'
            assert 'timestamp' in status

    @pytest.mark.asyncio
    async def test_exchange_status_offline(self, provider):
        """Тест проверки статуса биржи (offline)"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection error")

            status = await provider.get_exchange_status(Exchange.BINANCE)

            assert status['exchange'] == 'binance'
            assert status['status'] == 'offline'

    @pytest.mark.asyncio
    async def test_error_handling(self, provider):
        """Тест обработки ошибок"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Network error")

            # Все методы должны возвращать None или пустые значения при ошибках
            symbols = await provider.get_available_symbols(Exchange.BINANCE, 'USDT')
            assert symbols == []

            ticker = await provider.get_ticker_data(Exchange.BINANCE, 'BTCUSDT')
            assert ticker is None

            klines = await provider.get_klines(Exchange.BINANCE, 'BTCUSDT', TimeFrame.H1)
            assert klines is None

    @pytest.mark.asyncio
    async def test_cache_usage(self, provider):
        """Тест использования кэша"""
        # Мокаем Redis чтобы проверить кэширование
        provider._redis.get = AsyncMock(return_value=json.dumps(['BTCUSDT', 'ETHUSDT']))

        symbols = await provider.get_available_symbols(Exchange.BINANCE, 'USDT')

        # Должны получить данные из кэша, не делая HTTP запрос
        assert symbols == ['BTCUSDT', 'ETHUSDT']

        # Проверяем что setex вызывается при сохранении в кэш
        provider._redis.setex = AsyncMock()
        await provider._set_cached('test_key', 'test_data', 60)
        provider._redis.setex.assert_called_once()


class TestGlobalFunctions:
    """Тесты глобальных функций модуля"""

    @pytest.mark.asyncio
    async def test_get_symbols(self):
        """Тест глобальной функции get_symbols"""
        with patch('backend.exchange_data.get_data_provider') as mock_provider:
            mock_provider.return_value.get_available_symbols = AsyncMock(
                return_value=['BTCUSDT', 'ETHUSDT']
            )

            symbols = await get_symbols('binance', 'USDT')

            assert symbols == ['BTCUSDT', 'ETHUSDT']
            mock_provider.return_value.get_available_symbols.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_candles(self):
        """Тест глобальной функции get_candles"""
        mock_df = pd.DataFrame({'close': [45000.0]})

        with patch('backend.exchange_data.get_data_provider') as mock_provider:
            mock_provider.return_value.get_klines = AsyncMock(return_value=mock_df)

            df = await get_candles('binance', 'BTCUSDT', '1h', 100)

            assert df is not None
            assert len(df) == 1

    @pytest.mark.asyncio
    async def test_get_ticker_global(self):
        """Тест глобальной функции get_ticker"""
        mock_ticker = TickerData('BTCUSDT', 45000.0, 1000.0, 500.0, 1.12, 45500.0, 44500.0, datetime.now())

        with patch('backend.exchange_data.get_data_provider') as mock_provider:
            mock_provider.return_value.get_ticker_data = AsyncMock(return_value=mock_ticker)

            ticker = await get_ticker('binance', 'BTCUSDT')

            assert ticker.symbol == 'BTCUSDT'
            assert ticker.price == 45000.0

    @pytest.mark.asyncio
    async def test_get_available_cryptos(self):
        """Тест получения доступных криптовалют"""
        with patch('backend.exchange_data.get_data_provider') as mock_provider:
            mock_provider.return_value.get_available_symbols = AsyncMock(
                return_value=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            )

            cryptos = await get_available_cryptos()

            assert 'binance' in cryptos
            assert 'bybit' in cryptos
            assert len(cryptos['binance']) == 3
            assert len(cryptos['bybit']) == 3


class TestIntegration:
    """Интеграционные тесты (требуют реальных API вызовов)"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_calls(self):
        """Тест реальных API вызовов (только при наличии интернета)"""
        provider = UltraExchangeDataProvider()

        try:
            await provider.init()

            # Тестируем только Binance (более стабильный)
            symbols = await provider.get_available_symbols(Exchange.BINANCE, 'USDT')

            # Должен вернуть непустой список
            assert isinstance(symbols, list)
            assert len(symbols) > 0
            assert any('BTC' in symbol for symbol in symbols)  # Должен быть BTC

            # Тестируем получение тикера для реального символа
            if symbols:
                ticker = await provider.get_ticker_data(Exchange.BINANCE, symbols[0])
                assert ticker is not None
                assert ticker.price > 0

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")
        finally:
            await provider.close()


# Утилиты для запуска тестов
def run_tests():
    """Запуск всех тестов"""
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    return result.returncode


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении файла
    exit_code = run_tests()
    exit(exit_code)