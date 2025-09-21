from backend.bybit_client import (
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
    'klines',
    'ExchangeSource',
    'MarketDataType'
]

# ---- ВАЖНО: Авто-инициализация только по флагу среды, без блокировки цикла ----
import os
import asyncio

async def _auto_init():
    try:
        await init_bybit_client()
        print("✅ Bybit client auto-initialized successfully")
    except Exception as e:
        print(f"⚠️  Bybit client auto-init warning: {e}")

def _maybe_autoinit():
    # По умолчанию ОТКЛЮЧЕНО (без флага — никакого старта в тестах/импортах)
    if os.getenv("BYBIT_CLIENT_AUTOINIT", "0") != "1":
        return
    # Если цикл НЕ запущен — используем безопасный asyncio.run (не блокируем pytest loop)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_auto_init())
        return
    # Если цикл уже идёт (например, внутри приложения) — просто создаём задачу
    loop.create_task(_auto_init())

# Вызов не навредит в тестах, т.к. по умолчанию флаг = 0
_maybe_autoinit()
