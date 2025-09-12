from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import pandas as pd
import numpy as np
import talib
import json
import logging
import os
import time
from datetime import datetime, timedelta
import traceback
from backend.bybit_client import ticker as bybit_ticker, klines as bybit_klines
from analysis.signals import detect_signals
from functools import wraps
import threading

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-2025-crypto-analyst-pro')

# Настройка кэширования
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_THRESHOLD': 100
})
cache.init_app(app)

# Настройка rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# CSP политика
csp = {
    'default-src': "'self'",
    'script-src': [
        "'self'",
        "https://s3.tradingview.com",
        "https://www.tradingview.com",
        "https://unpkg.com",
        "'unsafe-inline'",
        "'unsafe-eval'"
    ],
    'style-src': [
        "'self'",
        "https://fonts.googleapis.com",
        "'unsafe-inline'"
    ],
    'img-src': [
        "'self'",
        "data:",
        "https:",
        "https://*.tradingview.com"
    ],
    'font-src': [
        "'self'",
        "https://cdnjs.cloudflare.com",
        "data:"
    ],
    'connect-src': [
        "'self'",
        "https://api.bybit.com",
        "https://*.tradingview.com"
    ],
    'frame-src': [
        "'self'",
        "https://s.tradingview.com"
    ],
    'object-src': "'none'",
    'base-uri': "'self'",
    'form-action': "'self'",
    'frame-ancestors': "'self'"
}

talisman = Talisman(
    app,
    content_security_policy=csp,
    force_https=False
)


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


# Инициализация API
crypto_api = CryptoAPI()


class TechnicalAnalyzer:
    """Класс для технического анализа данных"""

    def __init__(self):
        self.indicators_config = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'deviation': 2},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'atr': {'period': 14},
            'adx': {'period': 14}
        }

    def calculate_indicators(self, df):
        """Расчет технических индикаторов"""
        if df is None or len(df) < 50:
            return None

        try:
            df = df.copy()

            # Moving Averages
            df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ma_200'] = talib.SMA(df['close'], timeperiod=200)

            # Momentum Indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.indicators_config['rsi']['period'])

            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=self.indicators_config['macd']['fast'],
                slowperiod=self.indicators_config['macd']['slow'],
                signalperiod=self.indicators_config['macd']['signal']
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Volatility Indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'],
                                  timeperiod=self.indicators_config['atr']['period'])
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'],
                                  timeperiod=self.indicators_config['adx']['period'])

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'],
                timeperiod=self.indicators_config['bollinger']['period'],
                nbdevup=self.indicators_config['bollinger']['deviation'],
                nbdevdn=self.indicators_config['bollinger']['deviation']
            )
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower

            # Volume Analysis
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=self.indicators_config['stochastic']['k_period'],
                slowk_period=3,
                slowd_period=3
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            return df.dropna()

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return None

    def analyze_symbol(self, market_data):
        """Анализ символа на основе рыночных данных"""
        if market_data is None or len(market_data) < 50:
            return {"error": "Недостаточно данных для анализа"}

        df = self.calculate_indicators(market_data)
        if df is None:
            return {"error": "Ошибка расчета индикаторов"}

        latest = df.iloc[-1]

        # Анализ тренда
        trend = self._analyze_trend(latest)

        # Анализ RSI
        rsi_signal = self._analyze_rsi(latest)

        # Анализ MACD
        macd_signal = self._analyze_macd(latest)

        # Анализ объема
        volume_signal = self._analyze_volume(latest)

        # Генерация рекомендации
        recommendation, score, confidence = self._generate_recommendation(
            trend, rsi_signal, macd_signal, volume_signal
        )

        return {
            "trend": trend,
            "rsi": round(float(latest['rsi']), 2) if pd.notna(latest['rsi']) else 50,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "volume_signal": volume_signal,
            "score": score,
            "recommendation": recommendation,
            "confidence": confidence,
            "price": round(float(latest['close']), 8),
            "technical_indicators": {
                'macd': {
                    'value': round(float(latest['macd']), 6),
                    'signal': round(float(latest['macd_signal']), 6),
                    'histogram': round(float(latest['macd_hist']), 6)
                },
                'bollinger_bands': {
                    'position': self._get_bollinger_position(latest),
                    'width': round((latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle'] * 100, 2),
                    'upper': round(latest['bb_upper'], 8),
                    'middle': round(latest['bb_middle'], 8),
                    'lower': round(latest['bb_lower'], 8)
                },
                'stochastic': {
                    'k': round(float(latest['stoch_k']), 2),
                    'd': round(float(latest['stoch_d']), 2)
                }
            }
        }

    def _analyze_trend(self, data):
        """Анализ направления тренда"""
        if data['close'] > data['ma_50'] > data['ma_200']:
            return "Бычий"
        elif data['close'] < data['ma_50'] < data['ma_200']:
            return "Медвежий"
        else:
            return "Нейтральный"

    def _analyze_rsi(self, data):
        """Анализ RSI"""
        rsi = data['rsi']
        if rsi > 70:
            return "Перекупленность"
        elif rsi < 30:
            return "Перепроданность"
        else:
            return "Нейтральный"

    def _analyze_macd(self, data):
        """Анализ MACD"""
        if data['macd'] > data['macd_signal']:
            return "Бычий"
        elif data['macd'] < data['macd_signal']:
            return "Медвежий"
        else:
            return "Нейтральный"

    def _analyze_volume(self, data):
        """Анализ объема"""
        volume_ratio = data.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            return "Высокий объем"
        elif volume_ratio < 0.5:
            return "Низкий объем"
        else:
            return "Нормальный объем"

    def _get_bollinger_position(self, data):
        """Определение позиции относительно Bollinger Bands"""
        if data['close'] > data['bb_upper']:
            return "Выше верхней полосы"
        elif data['close'] < data['bb_lower']:
            return "Ниже нижней полосы"
        elif data['close'] > data['bb_middle']:
            return "Верхняя половина"
        else:
            return "Нижняя половина"

    def _generate_recommendation(self, trend, rsi, macd, volume):
        """Генерация торговой рекомендации"""
        score = 50

        # Весовые коэффициенты
        weights = {
            'trend': 0.3,
            'rsi': 0.25,
            'macd': 0.25,
            'volume': 0.2
        }

        # Учет тренда
        if trend == "Бычий":
            score += 20 * weights['trend']
        elif trend == "Медвежий":
            score -= 20 * weights['trend']

        # Учет RSI
        if rsi == "Перепроданность":
            score += 25 * weights['rsi']
        elif rsi == "Перекупленность":
            score -= 25 * weights['rsi']

        # Учет MACD
        if macd == "Бычий":
            score += 20 * weights['macd']
        elif macd == "Медвежий":
            score -= 20 * weights['macd']

        # Учет объема
        if volume == "Высокий объем":
            score += 15 * weights['volume']

        score = max(0, min(100, round(score)))

        if score >= 65:
            return "ПОКУПАТЬ", score, "Высокая"
        elif score <= 35:
            return "ПРОДАВАТЬ", score, "Высокая"
        else:
            return "НЕЙТРАЛЬНО", score, "Средняя"


# Инициализация анализатора
technical_analyzer = TechnicalAnalyzer()


def detect_patterns(market_data):
    """Обнаружение графических паттернов"""
    if market_data is None or len(market_data) < 100:
        return []

    patterns = []

    # Здесь можно добавить логику обнаружения реальных паттернов
    # Для демонстрации возвращаем тестовые данные

    patterns.append({
        "name": "Поддержка/Сопротивление",
        "type": "Разворотный",
        "confidence": "Средняя",
        "description": "Цена тестирует ключевые уровни",
        "easter_egg": "Обращайте внимание на объем при тестировании уровней"
    })

    return patterns


def calculate_profit_potential(market_data, patterns):
    """Расчет потенциальной прибыли"""
    if market_data is None:
        return {
            "potential_profit": "0%",
            "confidence": "Низкая",
            "risk_reward_ratio": "1:0"
        }

    try:
        # Анализ волатильности для оценки потенциала
        volatility = market_data['close'].pct_change().std() * np.sqrt(252) * 100
        potential_profit = min(25, max(5, volatility * 0.3))

        # Учет обнаруженных паттернов
        pattern_bonus = len(patterns) * 1.5
        potential_profit = min(30, potential_profit + pattern_bonus)

        confidence = "Высокая" if volatility > 40 and len(patterns) > 0 else "Средняя" if volatility > 20 else "Низкая"

        risk_reward = f"1:{min(3, max(1, potential_profit / 10)):.1f}"

        return {
            "potential_profit": f"{potential_profit:.1f}%",
            "confidence": confidence,
            "risk_reward_ratio": risk_reward,
            "stop_loss": f"{max(2, potential_profit / 3):.1f}%",
            "take_profit": f"{potential_profit:.1f}%",
            "volatility_based": f"{volatility:.1f}%"
        }

    except Exception as e:
        logger.error(f"Ошибка расчета прибыли: {e}")
        return {
            "potential_profit": "0%",
            "confidence": "Низкая",
            "risk_reward_ratio": "1:0"
        }


@app.context_processor
def utility_processor():
    return dict(
        min=min, max=max, len=len, range=range, zip=zip,
        enumerate=enumerate, round=round, str=str, int=int, float=float,
        datetime=datetime
    )


@app.route('/')
@limiter.limit("10 per minute")
@cache.cached(timeout=60)
def index():
    """Главная страница"""
    try:
        symbols = crypto_api.get_available_symbols()
        return render_template('index.html', symbols=symbols)
    except Exception as e:
        logger.error(f"Ошибка загрузки главной страницы: {e}")
        return render_template('error.html',
                               message="Ошибка загрузки данных",
                               error=str(e)), 500


@app.route('/analyze', methods=['GET'])
@limiter.limit("5 per minute")
def analyze():
    """Страница анализа криптовалюты"""
    try:
        symbol = request.args.get('symbol', '').upper()
        timeframe = request.args.get('timeframe', '60')

        if not symbol:
            return render_template('error.html',
                                   message="Не выбран символ для анализа"), 400

        # Получение рыночных данных
        market_data = crypto_api.get_market_data(symbol, timeframe, limit=300)
        if market_data is None or market_data.empty:
            return render_template('error.html',
                                   message=f"Не удалось получить данные для {symbol}"), 404

        if len(market_data) < 50:
            return render_template('error.html',
                                   message=f"Недостаточно данных для анализа ({len(market_data)} свечей)"), 400

        # Технический анализ
        analysis = technical_analyzer.analyze_symbol(market_data)

        # Обнаружение паттернов
        patterns = detect_patterns(market_data)

        # Расчет потенциала прибыли
        profit_potential = calculate_profit_potential(market_data, patterns)

        # Получение текущей цены
        ticker_info = crypto_api.get_ticker_info(symbol)
        current_price = ticker_info['lastPrice'] if ticker_info else analysis.get('price', 0)
        price_change = ticker_info['price24hPcnt'] if ticker_info else 0

        return render_template('analysis.html',
                               symbol=symbol,
                               timeframe=timeframe,
                               analysis=analysis,
                               patterns=patterns,
                               profit_potential=profit_potential,
                               current_price=current_price,
                               price_change=price_change,  # Добавьте это
                               now=datetime.now())

    except Exception as e:
        logger.error(f"Ошибка анализа: {e}\n{traceback.format_exc()}")
        return render_template('error.html',
                               message=f"Ошибка анализа: {str(e)}"), 500


@app.route('/patterns')
@limiter.limit("10 per minute")
@cache.cached(timeout=3600)
def patterns():
    """Страница с паттернами"""
    try:
        return render_template('patterns.html')
    except Exception as e:
        logger.error(f"Ошибка загрузки страницы паттернов: {e}")
        return render_template('error.html',
                               message="Ошибка загрузки страницы паттернов"), 500


@app.route('/api/top-cryptos')
@limiter.limit("30 per minute")
@cache.cached(timeout=15)
def api_top_cryptos():
    """API для получения топовых криптовалют"""
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                   'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT']

        cryptos_data = []

        for symbol in symbols:
            try:
                ticker_info = crypto_api.get_ticker_info(symbol)
                if ticker_info:
                    cryptos_data.append({
                        'symbol': symbol,
                        'name': symbol.replace('USDT', ''),
                        'price': float(ticker_info.get('lastPrice', 0)),
                        'price_change': float(ticker_info.get('price24hPcnt', 0)),
                        'volume': float(ticker_info.get('volume24h', 0)),
                        'high_24h': float(ticker_info.get('highPrice24h', 0)),
                        'low_24h': float(ticker_info.get('lowPrice24h', 0))
                    })
            except Exception as e:
                logger.warning(f"Ошибка обработки {symbol}: {e}")
                continue

        return jsonify({
            'success': True,
            'cryptos': cryptos_data,
            'timestamp': datetime.now().isoformat(),
            'count': len(cryptos_data)
        })

    except Exception as e:
        logger.error(f"Ошибка в api_top_cryptos: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'cryptos': [],
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/symbols')
@limiter.limit("20 per minute")
@cache.cached(timeout=300)
def api_symbols():
    """API для получения списка символов"""
    try:
        symbols = crypto_api.get_available_symbols()
        return jsonify({
            'success': True,
            'symbols': symbols,
            'count': len(symbols),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze/<symbol>')
@limiter.limit("10 per minute")
@cache.cached(timeout=60, query_string=True)
def api_analyze(symbol):
    """API для анализа символа"""
    try:
        timeframe = request.args.get('timeframe', '60')
        market_data = crypto_api.get_market_data(symbol, timeframe, limit=200)

        if market_data is None or market_data.empty:
            return jsonify({'success': False, 'error': 'No data available'}), 404

        analysis = technical_analyzer.analyze_symbol(market_data)
        patterns = detect_patterns(market_data)
        profit_potential = calculate_profit_potential(market_data, patterns)

        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis': analysis,
            'patterns': patterns,
            'profit_potential': profit_potential,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(market_data)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/patterns')
@limiter.limit("10 per minute")
@cache.cached(timeout=3600)
def api_patterns():
    """API для получения информации о паттернах"""
    try:
        patterns_data = [
            {
                "name": "Двойное дно",
                "type": "Разворотный",
                "confidence": "Высокая",
                "description": "Два минимума на одном уровне поддержки, указывающие на возможный разворот тренда вверх",
                "easter_egg": "Объем обычно увеличивается на втором дне"
            },
            {
                "name": "Двойная вершина",
                "type": "Разворотный",
                "confidence": "Высокая",
                "description": "Два максимума на одном уровне сопротивления, указывающие на возможный разворот тренда вниз",
                "easter_egg": "Пробитие линии шеи подтверждает разворот"
            },
            {
                "name": "Голова и плечи",
                "type": "Разворотный",
                "confidence": "Высокая",
                "description": "Три пика, где средний самый высокий, указывает на разворот восходящего тренда",
                "easter_egg": "Цель движения равна расстоянию от головы до линии шеи"
            },
            {
                "name": "Перевернутая голова и плечи",
                "type": "Разворотный",
                "confidence": "Высокая",
                "description": "Три впадины, где средняя самая глубокая, указывает на разворот нисходящего тренда",
                "easter_egg": "Требует подтверждения объемом на пробое"
            },
            {
                "name": "Треугольник",
                "type": "Продолжения",
                "confidence": "Средняя",
                "description": "Сужение диапазона цен между поддержкой и сопротивлением",
                "easter_egg": "Объем уменьшается во время формирования и увеличивается при пробое"
            },
            {
                "name": "Флаг",
                "type": "Продолжения",
                "confidence": "Высокая",
                "description": "Короткая консолидация против преобладающего тренда",
                "easter_egg": "Обычно возникает после резкого движения цены"
            },
            {
                "name": "Вымпел",
                "type": "Продолжения",
                "confidence": "Средняя",
                "description": "Треугольник с небольшим наклоном против тренда",
                "easter_egg": "Более надежный чем флаг, но встречается реже"
            },
            {
                "name": "Клин",
                "type": "Разворотный",
                "confidence": "Средняя",
                "description": "Сходящиеся линии поддержки и сопротивления с наклоном",
                "easter_egg": "Восходящий клин обычно медвежий, нисходящий - бычий"
            },
            {
                "name": "Поглощение",
                "type": "Свечной",
                "confidence": "Высокая",
                "description": "Свеча полностью поглощает предыдущую свечу противоположного цвета",
                "easter_egg": "Сильнее на дневных и недельных таймфреймах"
            },
            {
                "name": "Молот",
                "type": "Свечной",
                "confidence": "Средняя",
                "description": "Маленькое тело с длинной нижней тенью в конце нисходящего тренда",
                "easter_egg": "Требует подтверждения следующей свечой"
            },
            {
                "name": "Падающая звезда",
                "type": "Свечной",
                "confidence": "Средняя",
                "description": "Маленькое тело с длинной верхней тенью в конце восходящего тренда",
                "easter_egg": "Эффективнее на больших таймфреймах"
            },
            {
                "name": "Утренняя звезда",
                "type": "Свечной",
                "confidence": "Высокая",
                "description": "Трехсвечная модель разворота в конце нисходящего тренда",
                "easter_egg": "Средняя свеча должна быть дожи"
            },
            {
                "name": "Вечерняя звезда",
                "type": "Свечной",
                "confidence": "Высокая",
                "description": "Трехсвечная модель разворота в конце восходящего тренда",
                "easter_egg": "Сигнал усиливается при gap'ах между свечами"
            },
            {
                "name": "Бриша",
                "type": "Свечной",
                "confidence": "Средняя",
                "description": "Модель из пяти свеч с конкретными условиями закрытия",
                "easter_egg": "Редкая но очень надежная модель"
            }
        ]

        return jsonify({
            'success': True,
            'patterns': patterns_data,
            'count': len(patterns_data),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Ошибка загрузки паттернов: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'patterns': [],
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Проверка здоровья приложения"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'api_connection': 'ok',
            'data_processing': 'ok'
        }
    })


@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message="Страница не найдена"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Внутренняя ошибка сервера"), 500




# === New additive API v5 endpoints ===
@app.route('/api/ticker')
def api_ticker():
    symbol = request.args.get('symbol', 'BTCUSDT')
    data = bybit_ticker(symbol)
    if not data:
        return ('', 204)
    return jsonify(data)

@app.route('/api/klines')
def api_klines():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '60')
    limit = int(request.args.get('limit', '200'))
    data = bybit_klines(symbol, interval, limit)
    if not data:
        return ('', 204)
    return jsonify({"symbol": symbol, "interval": interval, "list": data})

@app.route('/api/analysis')
def api_analysis():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '60')
    strategy = request.args.get('strategy', 'classic')
    limit = int(request.args.get('limit', '200'))
    data = bybit_klines(symbol, interval, limit)
    if not data:
        return ('', 204)
    closes = [x["close"] for x in data]
    res = detect_signals(closes, strategy=strategy)
    N = 200
    result = {"symbol": symbol, "interval": interval, "strategy": strategy, "ind": {}, "alerts": res['alerts'][-N:]}
    # keep common keys safely if exist
    for k in ['ema9','ema21','macd','macd_signal','macd_hist','rsi','bb_mid','bb_upper','bb_lower','stoch_k','stoch_d']:
        if k in res:
            result["ind"][k] = res[k][-N:]
    return jsonify(result)

import io, csv
@app.route('/api/export/signals.csv')
def api_export_signals():
    # Export empty skeleton (alerts can be filled client-side in future iterations)
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["ts","type","idx"])
    return output.getvalue(), 200, {'Content-Type': 'text/csv; charset=utf-8'}
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'success': False, 'error': 'Rate limit exceeded'}), 429


if __name__ == '__main__':
    logger.info("Запуск CryptoAnalyst Pro...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', title="Dashboard")
