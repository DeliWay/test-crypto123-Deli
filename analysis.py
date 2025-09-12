import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
            "price_change": self._calculate_price_change(df),
            "technical_indicators": {
                'rsi': round(float(latest['rsi']), 2),
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

    def _calculate_price_change(self, df):
        """Расчет изменения цены"""
        if len(df) < 2:
            return 0
        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2]
        return ((current_price - previous_price) / previous_price) * 100

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


class PatternDetector:
    """Класс для обнаружения графических паттернов"""

    def detect_patterns(self, df):
        """Обнаружение графических паттернов"""
        if df is None or len(df) < 100:
            return []

        patterns = []

        # Обнаружение паттернов на основе реальных данных
        patterns.extend(self._detect_support_resistance(df))
        patterns.extend(self._detect_trend_lines(df))
        patterns.extend(self._detect_chart_patterns(df))

        return patterns

    def _detect_support_resistance(self, df):
        """Обнаружение уровней поддержки/сопротивления"""
        patterns = []

        # Анализ последних 20 свечей для обнаружения уровней
        recent_data = df.tail(20)

        # Поиск локальных минимумов и максимумов
        for i in range(5, len(recent_data) - 5):
            if self._is_local_minimum(recent_data, i):
                patterns.append({
                    "name": "Уровень поддержки",
                    "type": "Разворотный",
                    "confidence": "Средняя",
                    "description": f"Обнаружен уровень поддержки вокруг ${recent_data['close'].iloc[i]:.2f}",
                    "easter_egg": "Обратите внимание на объем при тестировании уровня поддержки"
                })

            if self._is_local_maximum(recent_data, i):
                patterns.append({
                    "name": "Уровень сопротивления",
                    "type": "Разворотный",
                    "confidence": "Средняя",
                    "description": f"Обнаружен уровень сопротивления вокруг ${recent_data['close'].iloc[i]:.2f}",
                    "easter_egg": "Пробитие уровня сопротивления с объемом может сигнализировать о продолжении тренда"
                })

        return patterns

    def _is_local_minimum(self, df, index):
        """Проверка является ли точка локальным минимумом"""
        if index < 2 or index > len(df) - 3:
            return False

        return (df['low'].iloc[index] < df['low'].iloc[index - 1] and
                df['low'].iloc[index] < df['low'].iloc[index - 2] and
                df['low'].iloc[index] < df['low'].iloc[index + 1] and
                df['low'].iloc[index] < df['low'].iloc[index + 2])

    def _is_local_maximum(self, df, index):
        """Проверка является ли точка локальным максимумом"""
        if index < 2 or index > len(df) - 3:
            return False

        return (df['high'].iloc[index] > df['high'].iloc[index - 1] and
                df['high'].iloc[index] > df['high'].iloc[index - 2] and
                df['high'].iloc[index] > df['high'].iloc[index + 1] and
                df['high'].iloc[index] > df['high'].iloc[index + 2])

    def _detect_trend_lines(self, df):
        """Обнаружение линий тренда"""
        patterns = []

        # Простой анализ тренда
        if len(df) >= 50:
            short_ma = df['close'].tail(20).mean()
            long_ma = df['close'].tail(50).mean()

            if short_ma > long_ma * 1.02:
                patterns.append({
                    "name": "Восходящий тренд",
                    "type": "Продолжения",
                    "confidence": "Высокая",
                    "description": "Цена находится в устойчивом восходящем тренде",
                    "easter_egg": "Ищите возможности покупки на откатах к поддержке"
                })
            elif short_ma < long_ma * 0.98:
                patterns.append({
                    "name": "Нисходящий тренд",
                    "type": "Продолжения",
                    "confidence": "Высокая",
                    "description": "Цена находится в устойчивом нисходящем тренде",
                    "easter_egg": "Ищите возможности продажи на отскоках к сопротивлению"
                })

        return patterns

    def _detect_chart_patterns(self, df):
        """Обнаружение графических паттернов"""
        patterns = []

        # Обнаружение потенциальных паттернов
        if self._detect_double_bottom(df):
            patterns.append({
                "name": "Двойное дно",
                "type": "Разворотный",
                "confidence": "Средняя",
                "description": "Обнаружено формирование двойного дна",
                "easter_egg": "Подтверждение паттерна происходит при пробое линии шеи"
            })

        if self._detect_head_shoulders(df):
            patterns.append({
                "name": "Голова и плечи",
                "type": "Разворотный",
                "confidence": "Высокая",
                "description": "Обнаружено формирование паттерна Голова и плечи",
                "easter_egg": "Цель движения обычно равна расстоянию от головы до линии шеи"
            })

        return patterns

    def _detect_double_bottom(self, df):
        """Обнаружение двойного дна"""
        if len(df) < 30:
            return False

        # Упрощенная логика обнаружения
        recent = df.tail(30)
        lows = recent['low'].values

        # Поиск двух близких минимумов
        min1_idx = np.argmin(lows[:15])
        min2_idx = 15 + np.argmin(lows[15:])

        if abs(lows[min1_idx] - lows[min2_idx]) / lows[min1_idx] < 0.02:
            return True

        return False

    def _detect_head_shoulders(self, df):
        """Обнаружение головы и плеч"""
        if len(df) < 40:
            return False

        # Упрощенная логика обнаружения
        recent = df.tail(40)
        highs = recent['high'].values

        # Поиск характерной формации
        if len(highs) >= 10:
            # Простая проверка на наличие пика в середине
            mid = len(highs) // 2
            if (highs[mid] > highs[mid - 3] and highs[mid] > highs[mid + 3] and
                    highs[mid - 3] > highs[mid - 6] and highs[mid + 3] > highs[mid + 6]):
                return True

        return False


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