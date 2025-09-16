import asyncio
import logging
import os
import json
import time
import threading
import concurrent.futures
import hashlib
import re
import urllib.parse
import zlib
import gzip
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from functools import wraps

from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_caching import Cache
from flask_compress import Compress
from flask_socketio import SocketIO, emit
import aiohttp
import redis
import numpy as np
import pandas as pd
from werkzeug.middleware.proxy_fix import ProxyFix

# Импорты из backend.bybit_client
from backend.bybit_client import (
    bybit_client,           # основной синхронный клиент
    get_available_symbols,  # асинхронная функция получения символов
    get_market_data,        # асинхронная функция получения рыночных данных
    get_ticker_info,        # асинхронная функция получения информации о тикере
    get_orderbook,          # асинхронная функция получения стакана
    init_bybit_client,      # инициализация
    close_bybit_client      # закрытие
)

# Импорты из analysis.analysis_engine
from analysis.analysis_engine import (
    analysis_engine,        # готовый экземпляр UltraAnalysisEngine
    analyze_symbol,         # асинхронная функция анализа символа
    detect_signals,         # асинхронная функция обнаружения сигналов
    calculate_indicators,   # асинхронная функция расчета индикаторов
    calculate_indicators_sync,  # синхронная версия расчета индикаторов
    init_analysis_engine,   # инициализация движка анализа
    close_analysis_engine   # закрытие движка анализа
)

from ml_analyzer import CryptoAnalyzer

# ========== КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ ==========

# Настройка расширенного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Конфигурация приложения
app.config.update({
    'SECRET_KEY': 'crypto-trading-analysis-secret-key-2024-v2-master',
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'SEND_FILE_MAX_AGE_DEFAULT': 300,
    'TEMPLATES_AUTO_RELOAD': True,
    'EXPLAIN_TEMPLATE_LOADING': False
})

# Конфигурация кэширования
cache_config = {
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_KEY_PREFIX': 'crypto_app_',
}

try:
    cache = Cache(app, config=cache_config)
except Exception as e:
    logger.warning(f"Cache initialization failed: {e}")
    cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Сжатие ответов
Compress(app)

# WebSocket
async_mode = os.getenv('SOCKETIO_ASYNC_MODE', 'threading')
try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode=async_mode,
        logger=False,
        engineio_logger=False
    )
except Exception as e:
    logger.warning(f"SocketIO initialization failed with {async_mode}: {e}, falling back to threading")
    socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*",
                       logger=False, engineio_logger=False)

# Инициализация клиентов

ml_analyzer = CryptoAnalyzer()

# Глобальный кэш и состояние
memory_cache = {}
app_state = {
    'start_time': time.time(),
    'request_count': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'active_connections': 0,
    'last_cleanup': time.time(),
    'initialized': False
}

# Потоковый исполнитель для блокирующих операций
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def async_handler(f: Callable) -> Callable:
    """Декоратор для асинхронных обработчиков с обработкой ошибок"""

    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return asyncio.run(f(*args, **kwargs))
        except asyncio.TimeoutError:
            logger.error(f"Async operation timeout in {f.__name__}")
            return jsonify({'error': 'Request timeout'}), 504
        except Exception as e:
            logger.error(f"Async error in {f.__name__}: {e}", exc_info=True)
            return jsonify({'error': 'Internal server error'}), 500

    return wrapped


def cache_memory(key_prefix: str, expiry: int = 30) -> Callable:
    """In-memory кэширование с автоматической очисткой"""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{urllib.parse.quote_plus(str(args))}:{urllib.parse.quote_plus(str(kwargs))}"
            current_time = time.time()

            # Очистка устаревших записей
            if current_time - app_state['last_cleanup'] > 300:
                cleanup_memory_cache()
                app_state['last_cleanup'] = current_time

            # Проверка кэша
            if cache_key in memory_cache:
                data, timestamp, _ = memory_cache[cache_key]
                if current_time - timestamp < expiry:
                    app_state['cache_hits'] += 1
                    return data

            # Выполнение функции
            result = f(*args, **kwargs)
            memory_cache[cache_key] = (result, current_time, expiry)
            app_state['cache_misses'] += 1
            return result

        return wrapper

    return decorator


def cleanup_memory_cache() -> None:
    """Очистка устаревших записей в memory cache"""
    current_time = time.time()
    keys_to_delete = []

    for key, (_, timestamp, expiry) in memory_cache.items():
        if current_time - timestamp > expiry:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del memory_cache[key]

    if keys_to_delete:
        logger.info(f"Memory cache cleaned: {len(keys_to_delete)} entries removed")


def json_error(status: int, code: str, detail: str) -> Response:
    """Универсальный хелпер для JSON ошибок"""
    response = jsonify({'error': code, 'detail': detail})
    response.status_code = status
    return response

def generate_etag(data: Any) -> str:
    """Генерация ETag для кэширования"""
    if isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True)
    else:
        content = str(data)
    return hashlib.md5(content.encode()).hexdigest()


def validate_symbol(symbol: str) -> str:
    """Валидация символа"""
    if not re.match(r'^[A-Z0-9]{1,20}$', symbol):
        return 'BTCUSDT'
    return symbol.upper()


def validate_timeframe(timeframe: str) -> str:
    """Валидация таймфрейма"""
    valid_timeframes = ['1', '3', '5', '15', '30', '60', '120', '240', 'D', 'W', 'M']
    return timeframe if timeframe in valid_timeframes else '15'


# ========== MIDDLEWARE И ОБРАБОТЧИКИ ==========

@app.before_request
def before_request() -> None:
    """Обработка входящих запросов"""
    app_state['active_connections'] += 1
    app_state['request_count'] += 1

    if request.path not in ['/health', '/favicon.ico', '/static/']:
        logger.info(f"{request.method} {request.path} - IP: {request.remote_addr}")


@app.after_request
def after_request(response: Response) -> Response:
    """Обработка исходящих ответов"""
    app_state['active_connections'] = max(0, app_state['active_connections'] - 1)

    # Добавление заголовков производительности
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Кэширование
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        response.headers['Cache-Control'] = 'public, max-age=60'

    response.headers['X-Request-Count'] = str(app_state['request_count'])
    response.headers['X-Cache-Hits'] = str(app_state['cache_hits'])
    response.headers['X-Cache-Misses'] = str(app_state['cache_misses'])

    return response


# ========== ОСНОВНЫЕ ЭНДПОИНТЫ ==========

@app.route('/')
@cache.cached(timeout=60, query_string=True)
def index() -> str:
    """Главная страница с TradingView виджетами"""
    symbol = validate_symbol(request.args.get('symbol', 'BTCUSDT'))
    timeframe = validate_timeframe(request.args.get('timeframe', '15'))
    theme = request.args.get('theme', 'dark')

    return render_template('index.html',
                           symbol=symbol,
                           timeframe=timeframe,
                           theme=theme,
                           page_title=f"Crypto Analysis - {symbol}",
                           server_start_time=app_state['start_time'])


@app.route('/api/market-data')
@async_handler
async def market_data() -> Response:
    """Комплексный эндпоинт рыночных данных"""
    try:
        symbol = validate_symbol(request.args.get('symbol', 'BTCUSDT'))
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        limit = min(int(request.args.get('limit', '200')), 1000)

        # Параллельная загрузка данных
        tasks = [
            get_market_data(symbol, timeframe, limit),
            get_ticker_info(symbol),
            get_orderbook(symbol),
            asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: analysis_engine.calculate_indicators_sync([1000, 2000, 3000])
            ),
            asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ml_analyzer.analyze_sync(symbol, timeframe)
            )
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка результатов
        market_data, ticker_info, orderbook, indicators, ml_analysis = results

        response_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'market_data': market_data if not isinstance(market_data, Exception) else {},
            'ticker_info': ticker_info if not isinstance(ticker_info, Exception) else {},
            'orderbook': orderbook if not isinstance(orderbook, Exception) else {},
            'technical_indicators': indicators if not isinstance(indicators, Exception) else {},
            'ml_analysis': ml_analysis if not isinstance(ml_analysis, Exception) else {},
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'server_time': int(time.time() * 1000)
        }

        # Генерация ETag
        etag = generate_etag(response_data)
        response = jsonify(response_data)
        response.headers['ETag'] = etag

        return response

    except Exception as e:
        logger.error(f"Market data error: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to fetch market data',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.route('/api/technical-analysis/<symbol>')
@async_handler
async def technical_analysis(symbol: str) -> Response:
    """Расширенный технический анализ"""
    try:
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        limit = min(int(request.args.get('limit', '500')), 1000)

        # Получение рыночных данных
        market_df = await get_market_data(symbol, timeframe, limit)
        if market_df is None or len(market_df) < 20:
            return jsonify({'error': 'Insufficient market data'}), 400

        # Анализ символа
        analysis = await analyze_symbol(symbol, market_df)

        response = jsonify(analysis)
        response.headers['ETag'] = generate_etag(analysis)

        return response

    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'Technical analysis failed',
            'details': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/ml-analysis/<symbol>')
@async_handler
async def ml_analysis(symbol: str) -> Response:
    """ML анализ с объяснимой AI"""
    try:
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        explain = request.args.get('explain', 'true').lower() == 'true'

        # Получение анализа
        analysis = await ml_analyzer.analyze(symbol, timeframe, explain)

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"ML analysis error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'ML analysis failed',
            'details': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/symbols')
@async_handler
async def get_symbols() -> Response:
    """Получение списка символов"""
    try:
        quote_coin = request.args.get('quote_coin', 'USDT').upper()
        source = request.args.get('source', 'bybit')

        symbols = await get_available_symbols(quote_coin, source)
        if not symbols:
            return json_error(502, 'symbols_fetch_failed', 'No symbols available from exchange')

        return jsonify({
            'symbols': symbols,
            'count': len(symbols),
            'quote_coin': quote_coin,
            'source': source,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'cached': False
        })

    except Exception as e:
        logger.error(f"Symbols fetch error: {e}", exc_info=True)
        return json_error(502, 'symbols_fetch_failed',
                         f'Failed to fetch symbols: {str(e)}')


@app.route('/api/orderbook/<symbol>')
@async_handler
async def get_orderbook_endpoint(symbol: str) -> Response:
    """Стакан заявок"""
    try:
        symbol = validate_symbol(symbol)
        depth = min(int(request.args.get('depth', '25')), 50)

        orderbook = await get_orderbook(symbol, depth)

        return jsonify(orderbook)

    except Exception as e:
        logger.error(f"Orderbook error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'Orderbook fetch failed',
            'details': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/historical-data/<symbol>')
@async_handler
async def historical_data(symbol: str) -> Response:
    """Исторические данные"""
    try:
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        limit = min(int(request.args.get('limit', '1000')), 2000)
        format_type = request.args.get('format', 'json')

        # Получение данных
        market_df = await get_market_data(symbol, timeframe, limit)
        if market_df is None:
            return jsonify({'error': 'No data available'}), 404

        if format_type == 'csv':
            # Конвертация в CSV
            output = BytesIO()
            market_df.to_csv(output, index=False)
            output.seek(0)

            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"{symbol}_{timeframe}_historical.csv"
            )
        else:
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'data': market_df.to_dict('records'),
                'count': len(market_df),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })

    except Exception as e:
        logger.error(f"Historical data error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'Historical data fetch failed',
            'details': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/indicators/<symbol>')
@async_handler
async def get_indicators(symbol: str) -> Response:
    """Получение технических индикаторов"""
    try:
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        limit = min(int(request.args.get('limit', '500')), 1000)

        # Получение рыночных данных
        market_df = await get_market_data(symbol, timeframe, limit)
        if market_df is None or len(market_df) < 20:
            return jsonify({'error': 'Insufficient data for indicators'}), 400

        closes = market_df['close'].tolist()

        # Расчет индикаторов
        indicators = await calculate_indicators(closes)

        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'indicators': indicators,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    except Exception as e:
        logger.error(f"Indicators error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'Indicators calculation failed',
            'details': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/signals/<symbol>')
@async_handler
async def get_signals(symbol: str) -> Response:
    """Получение торговых сигналов"""
    try:
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(request.args.get('timeframe', '15'))
        limit = min(int(request.args.get('limit', '500')), 1000)
        strategy = request.args.get('strategy', 'classic')

        # Получение рыночных данных
        market_df = await get_market_data(symbol, timeframe, limit)
        if market_df is None or len(market_df) < 20:
            return jsonify({'error': 'Insufficient data for signals'}), 400

        closes = market_df['close'].tolist()

        # Обнаружение сигналов
        signals = await detect_signals(closes, strategy)

        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'signals': signals,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    except Exception as e:
        logger.error(f"Signals error for {symbol}: {e}", exc_info=True)
        return jsonify({
            'error': 'Signals detection failed',
            'details': str(e),
            'symbol': symbol
        }), 500


# ========== СЛУЖЕБНЫЕ ЭНДПОИНТЫ ==========

@app.route('/health')
def health_check() -> Response:
    """Проверка здоровья системы"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'uptime_seconds': int(time.time() - app_state['start_time']),
        'services': {
            'bybit_api': 'unknown',
            'technical_analysis': 'unknown',
            'ml_service': 'unknown',
            'cache': 'healthy'
        },
        'metrics': {
            'requests_total': app_state['request_count'],
            'requests_active': app_state['active_connections'],
            'cache_hits': app_state['cache_hits'],
            'cache_misses': app_state['cache_misses'],
            'memory_cache_size': len(memory_cache)
        }
    }

    # Проверка сервисов
    try:
        # Проверка Bybit API
        test_symbols = asyncio.run(get_available_symbols('USDT', 'bybit'))
        if test_symbols:
            health_status['services']['bybit_api'] = 'healthy'
        else:
            health_status['services']['bybit_api'] = 'unhealthy'
            health_status['status'] = 'degraded'
    except:
        health_status['services']['bybit_api'] = 'unhealthy'
        health_status['status'] = 'degraded'

    # Проверка технического анализа
    try:
        test_data = analysis_engine.calculate_indicators_sync([1000, 2000, 3000])
        if test_data:
            health_status['services']['technical_analysis'] = 'healthy'
        else:
            health_status['services']['technical_analysis'] = 'unhealthy'
    except:
        health_status['services']['technical_analysis'] = 'unhealthy'

    # Проверка ML сервиса
    try:
        test_analysis = ml_analyzer.health_check()
        health_status['services']['ml_service'] = 'healthy' if test_analysis else 'unhealthy'
    except:
        health_status['services']['ml_service'] = 'unhealthy'

    return jsonify(health_status)


@app.route('/api/performance')
def performance_metrics() -> Response:
    """Метрики производительности"""
    metrics = {
        'memory': {
            'cache_size': len(memory_cache),
            'cache_hit_ratio': f"{(app_state['cache_hits'] / (app_state['cache_hits'] + app_state['cache_misses'] or 1)) * 100:.2f}%"
        },
        'system': {
            'active_threads': threading.active_count(),
            'memory_usage_mb': 'N/A',
        },
        'network': {
            'active_connections': app_state['active_connections'],
            'total_requests': app_state['request_count'],
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    return jsonify(metrics)


@app.route('/api/tradingview/config')
@async_handler
async def tradingview_config() -> Response:
    """Конфигурация TradingView"""
    try:
        config = {
            'supports_search': True,
            'supports_group_request': False,
            'supports_marks': False,
            'supports_timescale_marks': False,
            'supports_time': True,
            'exchanges': [
                {
                    'value': 'BYBIT',
                    'name': 'Bybit',
                    'desc': 'Bybit Exchange'
                }
            ],
            'symbols_types': [
                {'name': 'Crypto', 'value': 'crypto'}
            ],
            'supported_resolutions': ['1', '3', '5', '15', '30', '60', '120', '240', 'D', 'W', 'M'],
            'currency_codes': ['USD', 'USDT'],
            'supports_quoting': True,
            'supports_configuration': True
        }

        return jsonify(config)

    except Exception as e:
        logger.error(f"TradingView config error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/websocket-token')
@async_handler
async def get_websocket_token() -> Response:
    """Генерация токена для WebSocket"""
    try:
        # Генерация простого токена (в production использовать JWT)
        token = hashlib.sha256(f"{datetime.utcnow().isoformat()}{app.config['SECRET_KEY']}".encode()).hexdigest()
        return jsonify({
            'token': token,
            'expires_in': 3600,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        logger.error(f"WebSocket token error: {e}")
        return jsonify({'error': str(e)}), 500


# ========== СТАТИЧЕСКИЕ ФАЙЛЫ И SERVICE WORKER ==========

@app.route('/sw.js')
def service_worker() -> Response:
    """Service Worker для оффлайн работы"""
    return send_file('static/sw.js', mimetype='application/javascript')


@app.route('/manifest.json')
def webapp_manifest() -> Response:
    """Web App Manifest для PWA"""
    manifest = {
        "name": "Crypto Trading Analysis",
        "short_name": "CryptoAnalysis",
        "description": "Real-time cryptocurrency trading analysis with AI insights",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#131722",
        "theme_color": "#2962FF",
        "icons": [
            {
                "src": "/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    return jsonify(manifest)


@app.route('/static/<path:filename>')
def static_files(filename: str) -> Response:
    """Обслуживание статических файлов"""
    return send_file(f'static/{filename}')


# ========== ОБРАБОТЧИКИ ОШИБОК ==========

@app.errorhandler(404)
def not_found_error(error) -> Response:
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 404


@app.errorhandler(500)
def internal_error(error) -> Response:
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 500


@app.errorhandler(429)
def too_many_requests(error) -> Response:
    return jsonify({
        'error': 'Too many requests',
        'retry_after': 60,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 429


@app.errorhandler(400)
def bad_request(error) -> Response:
    return jsonify({
        'error': 'Bad request',
        'details': str(error),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 400


# ========== WebSocket ЭНДПОИНТЫ ==========

if socketio:
    @socketio.on('connect')
    def handle_connect() -> None:
        """Обработка подключения WebSocket"""
        logger.info(f"WebSocket client connected: {request.sid}")
        emit('connected', {'message': 'Connected to market data stream', 'sid': request.sid})


    @socketio.on('disconnect')
    def handle_disconnect() -> None:
        """Обработка отключения WebSocket"""
        logger.info(f"WebSocket client disconnected: {request.sid}")


    @socketio.on('subscribe')
    def handle_subscribe(data: Dict) -> None:
        """Подписка на рыночные данные"""
        try:
            symbol = validate_symbol(data.get('symbol', 'BTCUSDT'))
            timeframe = validate_timeframe(data.get('timeframe', '15'))

            logger.info(f"Client {request.sid} subscribed to {symbol}_{timeframe}")
            emit('subscribed', {
                'symbol': symbol,
                'timeframe': timeframe,
                'message': 'Subscription successful'
            })

        except Exception as e:
            logger.error(f"WebSocket subscribe error: {e}")
            emit('error', {'message': f'Subscription failed: {str(e)}'})


# ========== ИНИЦИАЛИЗАЦИЯ И ЗАПУСК ==========

def initialize_app() -> None:
    """Инициализация приложения"""
    logger.info("Starting application initialization...")

    # Предзагрузка в фоновом режиме
    def pre_warm_background():
        async def async_pre_warm():
            try:
                logger.info("Starting background pre-warming...")

                # Предзагрузка популярных символов
                popular_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

                for symbol in popular_symbols:
                    try:
                        data = await get_market_data(symbol, '15', 100)
                        if data is not None:
                            logger.debug(f"Pre-warmed data for {symbol}")
                        else:
                            logger.warning(f"Pre-warm failed for {symbol}: No data")
                        await asyncio.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Pre-warm failed for {symbol}: {str(e)}")
                logger.info("Pre-warming completed successfully")

            except Exception as e:
                logger.error(f"Pre-warming failed: {e}")

        asyncio.run(async_pre_warm())

    # Запуск в отдельном потоке
    threading.Thread(target=pre_warm_background, daemon=True).start()

    app_state['initialized'] = True
    logger.info("Application initialization complete")


# Инициализация при старте
initialize_app()

if __name__ == '__main__':
    # Конфигурация запуска
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )