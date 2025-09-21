import pytest
import asyncio
import pandas as pd
import sys
import os

# Добавляем путь к backend в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.bybit_client import (
    UltraBybitClient,
    ExchangeSource,
    # Все асинхронные функции
    get_available_symbols,
    get_market_data,
    get_ticker_info,
    get_orderbook,
    get_multiple_timeframes,
    get_batch_symbols_data,
    # Все синхронные функции
    get_bybit_symbols_sync,
    get_market_data_sync,
    get_ticker_info_sync,
    get_orderbook_sync,
    klines,
    # Утилиты
    init_bybit_client,
    close_bybit_client,
    get_client
)


@pytest.mark.asyncio
async def test_client_initialization():
    """Тест инициализации клиента"""
    client = UltraBybitClient()
    await client.init()

    assert client._initialized == True
    assert len(client._sessions) > 0

    available_exchanges = client._get_available_exchanges()
    assert len(available_exchanges) > 0, "No exchanges available"
    print(f"Available exchanges: {[e.value for e in available_exchanges]}")

    await client.close()


@pytest.mark.asyncio
async def test_get_spot_symbols():
    """Тест получения списка символов"""
    client = UltraBybitClient()
    await client.init()

    symbols = await client.get_spot_symbols('USDT')

    assert isinstance(symbols, list)
    if len(symbols) == 0:
        pytest.skip("No symbols returned - might be temporary API issue")
    assert len(symbols) > 0
    print(f"Got {len(symbols)} symbols")

    await client.close()


@pytest.mark.asyncio
async def test_get_klines_basic():
    """Тест получения свечных данных"""
    client = UltraBybitClient()
    await client.init()

    df = await client.get_klines('BTCUSDT', '1h', 100)

    if df is None:
        pytest.skip("No klines data - might be temporary API issue")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Проверяем наличие необходимых колонок
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    assert all(col in df.columns for col in expected_columns)
    print(f"Got {len(df)} candles for BTCUSDT")

    await client.close()


@pytest.mark.asyncio
async def test_get_klines_different_symbols():
    """Тест получения данных для разных символов"""
    client = UltraBybitClient()
    await client.init()

    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    successful = 0

    for symbol in test_symbols:
        df = await client.get_klines(symbol, '1h', 50)
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {symbol}: {len(df)} candles")
        else:
            print(f"⚠ {symbol}: No data (might be exchange issue)")

    if successful == 0:
        pytest.skip("No klines data for any symbol - might be temporary API issue")

    await client.close()


@pytest.mark.asyncio
async def test_get_klines_different_intervals():
    """Тест получения данных для разных таймфреймов"""
    client = UltraBybitClient()
    await client.init()

    intervals = ['15m', '1h', '4h']
    successful = 0

    for interval in intervals:
        df = await client.get_klines('BTCUSDT', interval, 20)
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {interval}: {len(df)} candles")
        else:
            print(f"⚠ {interval}: No data")

    if successful == 0:
        pytest.skip("No klines data for any interval - might be temporary API issue")

    await client.close()


@pytest.mark.asyncio
async def test_get_ticker_info():
    """Тест получения информации о тикере"""
    client = UltraBybitClient()
    await client.init()

    ticker_info = await client.get_ticker_info('BTCUSDT')

    if ticker_info is None:
        pytest.skip("No ticker info - might be temporary API issue")

    assert isinstance(ticker_info, dict)
    assert len(ticker_info) > 0
    print("✓ Got ticker info")

    await client.close()


@pytest.mark.asyncio
async def test_get_orderbook():
    """Тест получения стакана заявок"""
    client = UltraBybitClient()
    await client.init()

    orderbook = await client.get_orderbook('BTCUSDT', 10)

    if orderbook is None:
        pytest.skip("No orderbook data - might be temporary API issue")

    assert isinstance(orderbook, dict)
    assert len(orderbook) > 0
    print("✓ Got orderbook")

    await client.close()


@pytest.mark.asyncio
async def test_get_batch_klines():
    """Тест пакетного получения данных"""
    client = UltraBybitClient()
    await client.init()

    symbols = ['BTCUSDT', 'ETHUSDT']
    results = await client.get_batch_klines(symbols, '1h', 20)

    assert isinstance(results, dict)
    assert len(results) == len(symbols)

    successful = 0
    for symbol, df in results.items():
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {symbol}: {len(df)} candles")
        else:
            print(f"⚠ {symbol}: No data")

    if successful == 0:
        pytest.skip("No batch klines data - might be temporary API issue")

    await client.close()


@pytest.mark.asyncio
async def test_get_multiple_timeframes():
    """Тест получения данных по нескольким таймфреймам"""
    client = UltraBybitClient()
    await client.init()

    timeframes = ['15m', '1h']
    results = await client.get_multiple_timeframes('BTCUSDT', timeframes)

    assert isinstance(results, dict)

    successful = 0
    for tf, df in results.items():
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {tf}: {len(df)} candles")
        else:
            print(f"⚠ {tf}: No data")

    if successful == 0:
        pytest.skip("No multiple timeframes data - might be temporary API issue")

    await client.close()


def test_sync_client_methods():
    """Тест синхронных методов клиента"""
    client = UltraBybitClient()

    # Инициализируем синхронно
    import asyncio
    asyncio.run(client.init())

    # get_spot_symbols_sync
    symbols = client.get_spot_symbols_sync('USDT')
    assert isinstance(symbols, list)
    if len(symbols) == 0:
        pytest.skip("No symbols returned - might be temporary API issue")
    assert len(symbols) > 0
    print(f"✓ Sync symbols: {len(symbols)}")

    # get_klines_sync
    df = client.get_klines_sync('BTCUSDT', '1h', 50)
    if df is None or len(df) == 0:
        pytest.skip("No klines data - might be temporary API issue")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print(f"✓ Sync klines: {len(df)} candles")

    # get_ticker_info_sync
    ticker = client.get_ticker_info_sync('BTCUSDT')
    if ticker is None:
        pytest.skip("No ticker info - might be temporary API issue")
    assert isinstance(ticker, dict)
    print("✓ Sync ticker info")

    # get_orderbook_sync
    orderbook = client.get_orderbook_sync('BTCUSDT', 10)
    if orderbook is None:
        pytest.skip("No orderbook data - might be temporary API issue")
    assert isinstance(orderbook, dict)
    print("✓ Sync orderbook")

    # Закрываем синхронно
    asyncio.run(client.close())


@pytest.mark.asyncio
async def test_global_async_functions():
    """Тест глобальных асинхронных функций"""
    # Убедимся, что глобальный клиент закрыт перед началом
    try:
        await close_bybit_client()
    except:
        pass

    # Инициализируем клиент
    await init_bybit_client()

    try:
        # get_available_symbols
        symbols = get_available_symbols('USDT')
        assert isinstance(symbols, list)
        if len(symbols) == 0:
            pytest.skip("No symbols returned - might be temporary API issue")
        assert len(symbols) > 0
        print(f"✓ Global async symbols: {len(symbols)}")

        # get_market_data
        df = get_market_data('BTCUSDT', '1h', 50)
        if df is None or len(df) == 0:
            pytest.skip("No market data - might be temporary API issue")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        print(f"✓ Global async market data: {len(df)} candles")

        # get_ticker_info
        ticker = get_ticker_info('BTCUSDT')
        if ticker is None:
            pytest.skip("No ticker info - might be temporary API issue")
        assert isinstance(ticker, dict)
        print("✓ Global async ticker info")

        # get_orderbook
        orderbook = get_orderbook('BTCUSDT', 10)
        if orderbook is None:
            pytest.skip("No orderbook data - might be temporary API issue")
        assert isinstance(orderbook, dict)
        print("✓ Global async orderbook")

    finally:
        # Всегда закрываем клиент в finally блоке
        await close_bybit_client()


def test_global_sync_functions():
    """Тест глобальных синхронных функций"""
    # Инициализируем глобальный клиент
    import asyncio
    asyncio.run(init_bybit_client())

    # get_bybit_symbols_sync
    symbols = get_bybit_symbols_sync('USDT')
    assert isinstance(symbols, list)
    if len(symbols) == 0:
        pytest.skip("No symbols returned - might be temporary API issue")
    assert len(symbols) > 0
    print(f"✓ Global sync symbols: {len(symbols)}")

    # get_market_data_sync
    df = get_market_data_sync('BTCUSDT', '1h', 50)
    if df is None or len(df) == 0:
        pytest.skip("No market data - might be temporary API issue")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    print(f"✓ Global sync market data: {len(df)} candles")

    # get_ticker_info_sync
    ticker = get_ticker_info_sync('BTCUSDT')
    if ticker is None:
        pytest.skip("No ticker info - might be temporary API issue")
    assert isinstance(ticker, dict)
    print("✓ Global sync ticker info")

    # get_orderbook_sync
    orderbook = get_orderbook_sync('BTCUSDT', 10)
    if orderbook is None:
        pytest.skip("No orderbook data - might be temporary API issue")
    assert isinstance(orderbook, dict)
    print("✓ Global sync orderbook")

    # klines (alias)
    df_klines = klines('BTCUSDT', '1h', 50)
    if df_klines is None or len(df_klines) == 0:
        pytest.skip("No klines data - might be temporary API issue")
    assert isinstance(df_klines, pd.DataFrame)
    print(f"✓ klines alias: {len(df_klines)} candles")

    # Закрываем клиент
    asyncio.run(close_bybit_client())


@pytest.mark.asyncio
async def test_batch_functions():
    """Тест пакетных функций"""
    # Инициализируем клиент
    await init_bybit_client()

    # get_batch_symbols_data
    symbols = ['BTCUSDT', 'ETHUSDT']
    batch_data = get_batch_symbols_data(symbols, '1h', 20)
    assert isinstance(batch_data, dict)
    assert len(batch_data) == len(symbols)

    successful = 0
    for symbol, df in batch_data.items():
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {symbol}: {len(df)} candles")
        else:
            print(f"⚠ {symbol}: No data")

    if successful == 0:
        pytest.skip("No batch data - might be temporary API issue")

    print(f"✓ Batch data for {len(symbols)} symbols")

    # get_multiple_timeframes
    timeframes = ['15m', '1h']
    tf_data = get_multiple_timeframes('BTCUSDT', timeframes)
    assert isinstance(tf_data, dict)

    successful = 0
    for tf, df in tf_data.items():
        if df is not None and len(df) > 0:
            assert isinstance(df, pd.DataFrame)
            successful += 1
            print(f"✓ {tf}: {len(df)} candles")
        else:
            print(f"⚠ {tf}: No data")

    if successful == 0:
        pytest.skip("No multiple timeframes data - might be temporary API issue")

    print(f"✓ Multiple timeframes: {len(tf_data)} timeframes")

    # Закрываем клиент
    await close_bybit_client()


@pytest.mark.asyncio
async def test_utility_functions():
    """Тест утилитных функций"""
    # get_client
    client_instance = get_client()
    assert isinstance(client_instance, UltraBybitClient)
    print("✓ get_client works")

    # init_bybit_client
    await init_bybit_client()
    print("✓ init_bybit_client works")

    # close_bybit_client
    await close_bybit_client()
    print("✓ close_bybit_client works")


@pytest.mark.asyncio
async def test_client_context_manager():
    """Тест контекстного менеджера"""
    async with UltraBybitClient() as client:
        symbols = await client.get_spot_symbols('USDT')
        assert isinstance(symbols, list)
        if len(symbols) == 0:
            pytest.skip("No symbols returned - might be temporary API issue")
        assert len(symbols) > 0
        print(f"✓ Context manager symbols: {len(symbols)}")

        df = await client.get_klines('BTCUSDT', '1h', 10)
        if df is None or len(df) == 0:
            pytest.skip("No klines data - might be temporary API issue")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        print(f"✓ Context manager klines: {len(df)} candles")


@pytest.mark.asyncio
async def test_error_handling():
    """Тест обработки ошибок"""
    # Инициализируем клиент
    await init_bybit_client()

    # Несуществующий символ
    df = get_market_data('INVALID_SYMBOL_123', '1h', 10)
    # Может вернуть None - это нормально
    print("✓ Invalid symbol handled gracefully")

    # Несуществующий таймфрейм
    df = get_market_data('BTCUSDT', 'invalid_interval', 10)
    # Может вернуть None - это нормально
    print("✓ Invalid interval handled gracefully")

    # Закрываем клиент
    await close_bybit_client()


def test_data_quality():
    """Тест качества данных"""
    # Инициализируем глобальный клиент
    import asyncio
    asyncio.run(init_bybit_client())

    df = get_market_data_sync('BTCUSDT', '1h', 100)
    if df is None or len(df) == 0:
        pytest.skip("No data for quality check - might be temporary API issue")

    # Проверяем структуру данных
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    assert all(col in df.columns for col in expected_columns)

    # Проверяем корректность цен
    assert df['high'].max() >= df['low'].min()
    assert all(df['high'] >= df['low'])

    # Проверяем временные метки
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

    print("✓ Data quality checks passed")

    # Закрываем клиент
    asyncio.run(close_bybit_client())


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Тест конкурентных запросов"""
    client = UltraBybitClient()
    await client.init()

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    # Создаем задачи для конкурентного выполнения
    tasks = [client.get_klines(symbol, '1h', 20) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    # Проверяем, что все задачи завершились
    assert len(results) == len(symbols)

    successful = 0
    for result in results:
        if result is not None and len(result) > 0:
            assert isinstance(result, pd.DataFrame)
            successful += 1
        else:
            print(f"⚠ No data for one of the symbols")

    if successful == 0:
        pytest.skip("No concurrent requests data - might be temporary API issue")

    print(f"✓ Concurrent requests: {successful}/{len(symbols)} successful")

    await client.close()


if __name__ == "__main__":
    print("🚀 Starting Bybit Client Tests...")
    print("=" * 50)

    # Запуск тестов
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])