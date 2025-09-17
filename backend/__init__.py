"""
ULTRA-PERFORMANCE TRADING BACKEND
Инициализация модуля с оптимизированными импортами
"""

from .bybit_client import (
    UltraBybitClient,
    init_bybit_client,
    close_bybit_client,
    get_available_symbols,
    get_market_data,
    get_ticker_info,
    get_orderbook,
    get_multiple_timeframes,
    get_batch_symbols_data,
    get_bybit_symbols_sync,
    get_market_data_sync,
    get_ticker_info_sync,
    get_orderbook_sync,
    get_multiple_timeframes_sync,
    get_batch_symbols_data_sync,
    klines,
    ExchangeSource,
    MarketDataType
)

__all__ = [
    'UltraBybitClient',
    'init_bybit_client',
    'close_bybit_client',
    'get_available_symbols',
    'get_market_data',
    'get_ticker_info',
    'get_orderbook',
    'get_multiple_timeframes',
    'get_batch_symbols_data',
    'get_bybit_symbols_sync',
    'get_market_data_sync',
    'get_ticker_info_sync',
    'get_orderbook_sync',
    'get_multiple_timeframes_sync',
    'get_batch_symbols_data_sync',
    'klines',
    'ExchangeSource',
    'MarketDataType'
]

# Автоматическая инициализация при импорте
import asyncio

async def _auto_init():
    """Автоматическая инициализация клиента"""
    try:
        await init_bybit_client()
    except Exception as e:
        print(f"Auto-init warning: {e}")

# Запуск инициализации в фоне
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_auto_init())
    else:
        loop.run_until_complete(_auto_init())
except:
    pass