import pandas as pd
import numpy as np
import talib.abstract as ta
from datetime import datetime, timedelta
import logging
from backend.bybit_client import bybit_client, get_ticker_info
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoAnalyzer:
    def __init__(self):
        self.technical_indicators = [
            'RSI', 'MACD', 'BBANDS', 'STOCH', 'ADX',
            'ATR', 'OBV', 'CCI', 'MFI', 'WILLR'
        ]

    def analyze_symbol(self, symbol, timeframe='1h'):
        """Полный анализ символа"""
        try:
            # Получаем данные через bybit_client
            data = bybit_client.get_klines(symbol, timeframe, 500)
            if data is None or data.empty:
                logger.warning(f"No data for {symbol} on {timeframe}")
                return None

            # Вычисляем индикаторы
            indicators = self.calculate_indicators(data)

            # Анализируем паттерны
            patterns = self.analyze_patterns(data)

            # Генерируем сигналы
            signals = self.generate_signals(indicators, patterns)

            # Получаем текущую цену
            ticker = bybit_client.get_ticker_info(symbol)
            current_price = ticker['lastPrice'] if ticker and 'lastPrice' in ticker else data['close'].iloc[-1]

            # Создаем прогноз
            prediction = self.generate_prediction(data, indicators, signals, current_price)

            # Оцениваем качество анализа
            quality = self.assess_analysis_quality(data, indicators)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'indicators': indicators,
                'patterns': patterns,
                'signals': signals,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat(),
                'analysis_quality': quality
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def calculate_indicators(self, data):
        """Вычисление технических индикаторов"""
        indicators = {}

        try:
            # RSI
            indicators['rsi'] = ta.RSI(data, timeperiod=14).iloc[-1]

            # MACD
            macd = ta.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd['macd'].iloc[-1]
            indicators['macd_signal'] = macd['macdsignal'].iloc[-1]
            indicators['macd_hist'] = macd['macdhist'].iloc[-1]

            # Bollinger Bands
            bb = ta.BBANDS(data, timeperiod=20)
            indicators['bb_upper'] = bb['upperband'].iloc[-1]
            indicators['bb_middle'] = bb['middleband'].iloc[-1]
            indicators['bb_lower'] = bb['lowerband'].iloc[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']

            # Stochastic
            stoch = ta.STOCH(data, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = stoch['slowk'].iloc[-1]
            indicators['stoch_d'] = stoch['slowd'].iloc[-1]

            # Volume indicators
            indicators['volume_sma'] = ta.SMA(data, timeperiod=20, price='volume').iloc[-1]
            indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma'] if indicators[
                                                                                                   'volume_sma'] > 0 else 1

            # Trend indicators
            indicators['sma_20'] = ta.SMA(data, timeperiod=20).iloc[-1]
            indicators['sma_50'] = ta.SMA(data, timeperiod=50).iloc[-1]
            indicators['ema_12'] = ta.EMA(data, timeperiod=12).iloc[-1]
            indicators['ema_26'] = ta.EMA(data, timeperiod=26).iloc[-1]

            # Price ratios
            current_price = data['close'].iloc[-1]
            indicators['price_vs_sma20'] = current_price / indicators['sma_20'] - 1
            indicators['price_vs_sma50'] = current_price / indicators['sma_50'] - 1

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Заполняем значения по умолчанию при ошибке
            for key in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d']:
                indicators[key] = 50

        return indicators

    def analyze_patterns(self, data):
        """Анализ графических паттернов"""
        patterns = {}

        try:
            # Определяем тренд
            patterns['trend'] = self.detect_trend(data)

            # Ищем разворотные паттерны
            patterns['reversal'] = self.detect_reversal_patterns(data)

            # Ищем паттерны продолжения
            patterns['continuation'] = self.detect_continuation_patterns(data)

            # Анализ поддержки/сопротивления
            patterns['support_resistance'] = self.find_support_resistance(data)

        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            patterns['trend'] = 'neutral'
            patterns['reversal'] = []
            patterns['continuation'] = []
            patterns['support_resistance'] = {'support': [], 'resistance': []}

        return patterns

    def detect_trend(self, data):
        """Определение тренда"""
        try:
            prices = data['close'].tail(50)
            if len(prices) < 10:
                return 'neutral'

            # Простая линейная регрессия для определения тренда
            x = np.arange(len(prices))
            y = prices.values
            slope = np.polyfit(x, y, 1)[0]

            # Нормализуем наклон относительно цены
            normalized_slope = slope / prices.mean()

            if normalized_slope > 0.001:
                return 'bullish'
            elif normalized_slope < -0.001:
                return 'bearish'
            else:
                return 'neutral'

        except:
            return 'neutral'

    def detect_reversal_patterns(self, data):
        """Обнаружение разворотных паттернов"""
        patterns = []
        try:
            # Упрощенная проверка паттернов
            if self.is_double_bottom(data):
                patterns.append('double_bottom')
            if self.is_double_top(data):
                patterns.append('double_top')

        except:
            pass

        return patterns

    def is_double_bottom(self, data):
        """Проверка паттерна двойное дно"""
        try:
            lows = data['low'].tail(20)
            if len(lows) < 10:
                return False

            # Ищем два приблизительно равных минимума
            min1 = lows.iloc[-10]
            min2 = lows.iloc[-1]

            return abs(min1 - min2) / min1 < 0.02  # Разница менее 2%
        except:
            return False

    def is_double_top(self, data):
        """Проверка паттерна двойная вершина"""
        try:
            highs = data['high'].tail(20)
            if len(highs) < 10:
                return False

            # Ищем два приблизительно равных максимума
            max1 = highs.iloc[-10]
            max2 = highs.iloc[-1]

            return abs(max1 - max2) / max1 < 0.02  # Разница менее 2%
        except:
            return False

    def find_support_resistance(self, data):
        """Поиск уровней поддержки и сопротивления"""
        try:
            # Упрощенный алгоритм поиска уровней
            prices = data['close'].tail(100)
            support = prices.min() * 0.99
            resistance = prices.max() * 1.01

            return {
                'support': [support],
                'resistance': [resistance]
            }
        except:
            return {'support': [], 'resistance': []}

    def detect_continuation_patterns(self, data):
        """Обнаружение паттернов продолжения"""
        return []  # Упрощенная реализация

    def generate_signals(self, indicators, patterns):
        """Генерация торговых сигналов"""
        signals = []
        score = 0

        try:
            # Анализ RSI
            if indicators.get('rsi', 50) < 30:
                signals.append('rsi_oversold')
                score += 2
            elif indicators.get('rsi', 50) > 70:
                signals.append('rsi_overbought')
                score -= 2

            # Анализ MACD
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                signals.append('macd_bullish')
                score += 1
            else:
                signals.append('macd_bearish')
                score -= 1

            # Анализ цены относительно Bollinger Bands
            current_price = indicators.get('current_price', 0)
            bb_lower = indicators.get('bb_lower', 0)
            bb_upper = indicators.get('bb_upper', 0)

            if bb_lower > 0 and current_price < bb_lower * 1.02:
                signals.append('bb_oversold')
                score += 2
            elif bb_upper > 0 and current_price > bb_upper * 0.98:
                signals.append('bb_overbought')
                score -= 2

            # Анализ тренда
            trend = patterns.get('trend', 'neutral')
            if trend == 'bullish':
                signals.append('uptrend')
                score += 1
            elif trend == 'bearish':
                signals.append('downtrend')
                score -= 1

            # Определение общего сигнала
            if score >= 3:
                final_signal = 'STRONG_BUY'
            elif score >= 1:
                final_signal = 'BUY'
            elif score <= -3:
                final_signal = 'STRONG_SELL'
            elif score <= -1:
                final_signal = 'SELL'
            else:
                final_signal = 'NEUTRAL'

            return {
                'signals': signals,
                'score': score,
                'final_signal': final_signal,
                'confidence': min(abs(score) / 5.0, 1.0)  # Уверенность от 0 до 1
            }

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'signals': ['error'],
                'score': 0,
                'final_signal': 'NEUTRAL',
                'confidence': 0.0
            }

    def generate_prediction(self, data, indicators, signals, current_price):
        """Генерация прогноза цены"""
        try:
            # Рассчитываем волатильность
            volatility = data['close'].pct_change().std() * np.sqrt(365)
            if np.isnan(volatility) or volatility == 0:
                volatility = 0.02  # Значение по умолчанию 2%

            direction = 0
            if signals['final_signal'] in ['STRONG_BUY', 'BUY']:
                direction = 1
            elif signals['final_signal'] in ['STRONG_SELL', 'SELL']:
                direction = -1

            # Прогноз на 24 часа
            predicted_change = direction * volatility / np.sqrt(365) * 1.5
            target_price = current_price * (1 + predicted_change)

            # Определяем уровни стоп-лосс и тейк-профит
            if direction > 0:
                stop_loss = current_price * (1 - volatility * 0.5)
                take_profit = current_price * (1 + volatility * 2)
            elif direction < 0:
                stop_loss = current_price * (1 + volatility * 0.5)
                take_profit = current_price * (1 - volatility * 2)
            else:
                stop_loss = current_price * (1 - volatility * 0.3)
                take_profit = current_price * (1 + volatility * 0.3)

            return {
                'target_price': round(target_price, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'predicted_change_percent': round(predicted_change * 100, 2),
                'time_horizon': '24-48 hours',
                'risk_level': 'high' if abs(predicted_change) > 0.05 else 'medium'
            }

        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return {
                'target_price': round(current_price, 4),
                'stop_loss': round(current_price * 0.95, 4),
                'take_profit': round(current_price * 1.05, 4),
                'predicted_change_percent': 0.0,
                'time_horizon': '24-48 hours',
                'risk_level': 'medium'
            }

    def assess_analysis_quality(self, data, indicators):
        """Оценка качества анализа"""
        quality_score = 0

        try:
            # Проверяем достаточность данных
            if len(data) >= 100:
                quality_score += 2

            # Проверяем волатильность
            volatility = data['close'].pct_change().std()
            if not np.isnan(volatility) and volatility > 0.001:
                quality_score += 1

            # Проверяем объемы торгов
            if data['volume'].mean() > 1000:
                quality_score += 1

            # Проверяем качество индикаторов
            valid_indicators = sum(1 for value in indicators.values() if not np.isnan(value))
            if valid_indicators / len(indicators) > 0.8:
                quality_score += 1

            rating = 'excellent' if quality_score >= 5 else 'good' if quality_score >= 3 else 'fair'

            return {
                'score': quality_score,
                'rating': rating,
                'data_points': len(data),
                'volatility': volatility if not np.isnan(volatility) else 0.0
            }

        except:
            return {
                'score': 0,
                'rating': 'fair',
                'data_points': len(data) if data is not None else 0,
                'volatility': 0.0
            }


# Глобальный экземпляр анализатора
crypto_analyzer = CryptoAnalyzer()