import pandas as pd
import numpy as np
import talib
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Детектор графических паттернов для технического анализа
    Реализует обнаружение и анализ основных ценовых паттернов
    """

    def __init__(self):
        # Конфигурация параметров для различных паттернов
        self.pattern_config = {
            'head_shoulders': {
                'min_peaks': 3,
                'peak_distance': 5,
                'height_variation': 0.05
            },
            'double_top_bottom': {
                'price_tolerance': 0.02,
                'time_tolerance': 15,
                'min_time_gap': 5
            },
            'triangle': {
                'convergence_threshold': 0.001,
                'min_bars': 10
            },
            'candlestick': {
                'shadow_ratio': 0.6,
                'body_ratio': 0.3
            }
        }

        # Описания паттернов для образовательных целей
        self.pattern_descriptions = {
            'Голова и плечи': {
                'type': 'Разворотный',
                'trend': 'Медвежий',
                'description': 'Состоит из трех вершин: левое плечо, голова (самая высокая), правое плечо. Линия шеи соединяет основания между вершинами.',
                'reliability': 'Высокая',
                'volume': 'Уменьшается на каждой последующей вершине',
                'target': 'Расстояние от головы до линии шеи',
                'confirmation': 'Пробой линии шеи с увеличением объема'
            },
            'Двойная вершина': {
                'type': 'Разворотный',
                'trend': 'Медвежий',
                'description': 'Две примерно равные вершины после восходящего тренда. Формируется на сопротивлении.',
                'reliability': 'Средняя',
                'volume': 'Обычно ниже на второй вершине',
                'target': 'Расстояние от вершин до уровня поддержки',
                'confirmation': 'Пробой поддержки между вершинами'
            },
            'Двойное дно': {
                'type': 'Разворотный',
                'trend': 'Бычий',
                'description': 'Две примерно равные впадины после нисходящего тренда. Формируется на поддержке.',
                'reliability': 'Средняя',
                'volume': 'Обычно выше на втором дне',
                'target': 'Расстояние от впадин до уровня сопротивления',
                'confirmation': 'Пробой сопротивления между впадинами'
            },
            'Треугольник': {
                'type': 'Продолжение',
                'trend': 'Зависит от направления пробоя',
                'description': 'Сходящиеся линии поддержки и сопротивления. Объем уменьшается при формировании.',
                'reliability': 'Высокая',
                'volume': 'Уменьшается внутри треугольника, увеличивается при пробое',
                'target': 'Высота треугольника в начале формирования',
                'confirmation': 'Пробой в направлении тренда с объемом'
            }
        }

    def detect_patterns(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        Основная функция обнаружения паттернов
        Возвращает список обнаруженных паттернов с метаданными
        """
        if market_data is None or len(market_data) < 50:
            return []

        patterns = []

        try:
            # Обнаружение различных типов паттернов
            patterns.extend(self._detect_reversal_patterns(market_data))
            patterns.extend(self._detect_continuation_patterns(market_data))
            patterns.extend(self._detect_candlestick_patterns(market_data))

            # Фильтрация дубликатов и слабых паттернов
            patterns = self._filter_patterns(patterns)

            logger.info(f"Обнаружено {len(patterns)} паттернов")
            return patterns

        except Exception as e:
            logger.error(f"Ошибка обнаружения паттернов: {e}")
            return []

    def _detect_reversal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Обнаружение разворотных паттернов"""
        patterns = []

        # Голова и плечи
        hs_pattern = self._detect_head_shoulders(df)
        if hs_pattern:
            patterns.append(hs_pattern)

        # Двойная вершина/дно
        dt_pattern = self._detect_double_top_bottom(df)
        if dt_pattern:
            patterns.append(dt_pattern)

        return patterns

    def _detect_continuation_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Обнаружение паттернов продолжения тренда"""
        patterns = []

        # Треугольники
        triangle_pattern = self._detect_triangle_patterns(df)
        if triangle_pattern:
            patterns.append(triangle_pattern)

        # Флаги и вымпелы
        flag_pattern = self._detect_flag_pennant(df)
        if flag_pattern:
            patterns.append(flag_pattern)

        return patterns

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Обнаружение свечных паттернов"""
        patterns = []

        # Молот и повешенный
        hammer_pattern = self._detect_hammer_hanging_man(df)
        if hammer_pattern:
            patterns.append(hammer_pattern)

        # Поглощение
        engulfing_pattern = self._detect_engulfing_patterns(df)
        if engulfing_pattern:
            patterns.append(engulfing_pattern)

        # Доджи
        doji_pattern = self._detect_doji_patterns(df)
        if doji_pattern:
            patterns.append(doji_pattern)

        return patterns

    def _detect_head_shoulders(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение паттерна 'Голова и плечи'
        🔍 Как работает: Ищем три вершины, где средняя (голова) выше двух других (плеч)
        """
        try:
            if len(df) < 30:
                return None

            # Используем сглаженные цены для лучшего обнаружения пиков
            close_prices = df['close'].values[-50:]
            high_prices = df['high'].values[-50:]

            # Находим все значимые пики
            peaks = []
            for i in range(5, len(high_prices) - 5):
                if (high_prices[i] > np.max(high_prices[i - 5:i]) and
                        high_prices[i] > np.max(high_prices[i + 1:i + 6])):
                    peaks.append((i, high_prices[i]))

            if len(peaks) < 3:
                return None

            # Сортируем пики по высоте
            peaks.sort(key=lambda x: x[1], reverse=True)

            # Голова - самая высокая вершина
            head_idx, head_price = peaks[0]

            # Ищем плечи - вершины примерно на одном уровне
            shoulders = []
            for idx, price in peaks[1:]:
                # Плечи должны быть примерно на одном уровне и симметрично расположены
                price_diff = abs(price - head_price) / head_price
                if price_diff > 0.15:  # Слишком большая разница с головой
                    continue

                shoulders.append((idx, price))

            if len(shoulders) >= 2:
                # Проверяем симметричность расположения плеч
                left_shoulder = min(shoulders, key=lambda x: x[0])
                right_shoulder = max(shoulders, key=lambda x: x[0])

                # Плечи должны быть по разные стороны от головы
                if left_shoulder[0] < head_idx < right_shoulder[0]:
                    # Проверяем примерное равенство высот плеч
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1])

                    if shoulder_diff < 0.03:  # Плечи примерно равны по высоте
                        return self._create_pattern_dict(
                            "Голова и плечи",
                            "Разворотный",
                            "Высокая",
                            "Классический разворотный паттерн после восходящего тренда",
                            "Ищите пробой линии шеи для подтверждения"
                        )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения головы и плеч: {e}")
            return None

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение двойной вершины или дна
        🔍 Двойная вершина: Две вершины на одном уровне после роста
        🔍 Двойное дно: Два минимума на одном уровне после падения
        """
        try:
            if len(df) < 20:
                return None

            high_prices = df['high'].values[-20:]
            low_prices = df['low'].values[-20:]

            # Поиск двойной вершины
            max1_idx = np.argmax(high_prices)
            max1_val = high_prices[max1_idx]

            # Ищем вторую вершину, исключая область вокруг первой
            mask = np.ones_like(high_prices, dtype=bool)
            start_idx = max(0, max1_idx - 3)
            end_idx = min(len(high_prices), max1_idx + 4)
            mask[start_idx:end_idx] = False

            if np.any(mask):
                max2_val = np.max(high_prices[mask])
                max2_idx = np.where(high_prices == max2_val)[0][0]

                price_diff = abs(max1_val - max2_val) / max1_val
                time_diff = abs(max1_idx - max2_idx)

                if (price_diff < 0.02 and 5 <= time_diff <= 15):
                    return self._create_pattern_dict(
                        "Двойная вершина",
                        "Разворотный",
                        "Средняя",
                        "Две вершины на одном уровне сопротивления",
                        "Объем обычно снижается на второй вершине"
                    )

            # Поиск двойного дна
            min1_idx = np.argmin(low_prices)
            min1_val = low_prices[min1_idx]

            mask = np.ones_like(low_prices, dtype=bool)
            start_idx = max(0, min1_idx - 3)
            end_idx = min(len(low_prices), min1_idx + 4)
            mask[start_idx:end_idx] = False

            if np.any(mask):
                min2_val = np.min(low_prices[mask])
                min2_idx = np.where(low_prices == min2_val)[0][0]

                price_diff = abs(min1_val - min2_val) / min1_val
                time_diff = abs(min1_idx - min2_idx)

                if (price_diff < 0.02 and 5 <= time_diff <= 15):
                    return self._create_pattern_dict(
                        "Двойное дно",
                        "Разворотный",
                        "Средняя",
                        "Два минимума на одном уровне поддержки",
                        "Объем обычно увеличивается на втором дне"
                    )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения двойной вершины/дна: {e}")
            return None

    def _detect_triangle_patterns(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение треугольных паттернов
        🔍 Восходящий: Горизонтальное сопротивление + rising support
        🔍 Нисходящий: Горизонтальная поддержка + falling resistance
        🔍 Симметричный: Сходящиеся поддержка и сопротивление
        """
        try:
            if len(df) < 20:
                return None

            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]

            # Анализ линий тренда
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]

            if high_slope < -0.001 and low_slope > 0.001:
                return self._create_pattern_dict(
                    "Треугольник (Симметричный)",
                    "Продолжение",
                    "Высокая",
                    "Сходящиеся линии поддержки и сопротивления",
                    "Пробивает в направлении предыдущего тренда"
                )
            elif abs(high_slope) < 0.001 and low_slope > 0.001:
                return self._create_pattern_dict(
                    "Треугольник (Восходящий)",
                    "Продолжение",
                    "Высокая",
                    "Горизонтальное сопротивление + восходящая поддержка",
                    "Бычий пробой более вероятен"
                )
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                return self._create_pattern_dict(
                    "Треугольник (Нисходящий)",
                    "Продолжение",
                    "Высокая",
                    "Горизонтальная поддержка + нисходящее сопротивление",
                    "Медвежий пробой более вероятен"
                )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения треугольника: {e}")
            return None

    def _detect_hammer_hanging_man(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение свечных паттернов Молот и Повешенный
        🔍 Молот: Длинная нижняя тень, маленькое тело (бычий сигнал внизу)
        🔍 Повешенный: То же, но вверху тренда (медвежий сигнал)
        """
        try:
            if len(df) < 3:
                return None

            latest = df.iloc[-1]
            prev_trend = self._get_short_term_trend(df)

            body_size = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']

            if total_range == 0:
                return None

            lower_shadow = min(latest['close'], latest['open']) - latest['low']
            upper_shadow = latest['high'] - max(latest['close'], latest['open'])

            shadow_ratio = lower_shadow / total_range
            body_ratio = body_size / total_range

            # Критерии паттерна
            if (shadow_ratio > 0.6 and
                    upper_shadow / total_range < 0.2 and
                    body_ratio < 0.3):
                pattern_name = "Молот" if prev_trend == 'down' else "Повешенный"
                pattern_type = "Бычий" if prev_trend == 'down' else "Медвежий"

                return self._create_pattern_dict(
                    pattern_name,
                    "Разворотный",
                    "Средняя",
                    f"{pattern_type} разворотный паттерн с длинной нижней тенью",
                    "Требует подтверждения следующей свечой"
                )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения молота/повешенного: {e}")
            return None

    def _detect_engulfing_patterns(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение паттерна Поглощение
        🔍 Бычье поглощение: Зеленая свеча полностью поглощает красную
        🔍 Медвежье поглощение: Красная свеча полностью поглощает зеленую
        """
        try:
            if len(df) < 3:
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Бычье поглощение
            if (latest['close'] > latest['open'] and  # Бычья свеча
                    prev['close'] < prev['open'] and  # Медвежья свеча
                    latest['open'] < prev['close'] and  # Открытие ниже закрытия предыдущей
                    latest['close'] > prev['open']):  # Закрытие выше открытия предыдущей

                return self._create_pattern_dict(
                    "Бычье поглощение",
                    "Разворотный",
                    "Высокая",
                    "Бычья свеча полностью поглощает медвежью",
                    "Сильный разворотный сигнал при объеме"
                )

            # Медвежье поглощение
            elif (latest['close'] < latest['open'] and  # Медвежья свеча
                  prev['close'] > prev['open'] and  # Бычья свеча
                  latest['open'] > prev['close'] and  # Открытие выше закрытия предыдущей
                  latest['close'] < prev['open']):  # Закрытие ниже открытия предыдущей

                return self._create_pattern_dict(
                    "Медвежье поглощение",
                    "Разворотный",
                    "Высокая",
                    "Медвежья свеча полностью поглощает бычью",
                    "Сильный разворотный сигнал при объеме"
                )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения поглощения: {e}")
            return None

    def _detect_doji_patterns(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение свечей Доджи
        🔍 Доджи: Очень маленькое тело, показывает нерешительность
        """
        try:
            latest = df.iloc[-1]

            body_size = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']

            if total_range == 0:
                return None

            body_ratio = body_size / total_range

            # Критерий доджи - очень маленькое тело
            if body_ratio < 0.1:
                return self._create_pattern_dict(
                    "Доджи",
                    "Разворотный",
                    "Низкая",
                    "Свеча с очень маленьким телом, показывает нерешительность",
                    "Требует подтверждения и анализа контекста"
                )

            return None

        except Exception as e:
            logger.error(f"Ошибка обнаружения доджи: {e}")
            return None

    def _detect_flag_pennant(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Обнаружение флагов и вымпелов
        🔍 Флаг: Параллельные линии после сильного движения
        🔍 Вымпел: Сходящиеся линии (маленький треугольник)
        """
        try:
            if len(df) < 25:
                return None

            prices = df['close'].values[-25:]

            # Проверяем наличие сильного движения (древко флага)
            initial_move = abs(prices[0] - prices[10]) / prices[0]
            if initial_move < 0.05:  # Слишком слабое движение
                return None

            # Анализируем консолидацию (полотнище флага)
            consolidation_prices = prices[10:20]
            consolidation_range = (np.max(consolidation_prices) - np.min(consolidation_prices)) / np.mean(
                consolidation_prices)

            if consolidation_range < 0.02:  # Слишком узкий диапазон
                return None

            # Определяем тип паттерна
            if consolidation_range > 0.08:  # Широкий диапазон - вероятно флаг
                return self._create_pattern_dict(
                    "Флаг",
                    "Продолжение",
                    "Высокая",
                    "Короткая консолидация после сильного движения",
                    "Пробивает в направлении исходного движения"
                )
            else:  # Узкий диапазон - вероятно вымпел
                return self._create_pattern_dict(
                    "Вымпел",
                    "Продолжение",
                    "Высокая",
                    "Маленький треугольник после сильного движения",
                    "Пробивает в направлении исходного движения"
                )

        except Exception as e:
            logger.error(f"Ошибка обнаружения флага/вымпела: {e}")
            return None

    def _get_short_term_trend(self, df: pd.DataFrame, period: int = 5) -> str:
        """Определение краткосрочного тренда"""
        if len(df) < period + 1:
            return 'neutral'

        recent_prices = df['close'].values[-(period + 1):]
        price_change = recent_prices[-1] - recent_prices[0]

        if price_change > 0:
            return 'up'
        elif price_change < 0:
            return 'down'
        else:
            return 'neutral'

    def _create_pattern_dict(self, name: str, pattern_type: str,
                             confidence: str, description: str, easter_egg: str) -> Dict:
        """Создание словаря с информацией о паттерне"""
        return {
            "name": name,
            "type": pattern_type,
            "confidence": confidence,
            "description": description,
            "easter_egg": easter_egg,
            "timestamp": datetime.now().isoformat()
        }

    def _filter_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Фильтрация и ранжирование паттернов"""
        if not patterns:
            return []

        # Удаляем дубликаты по имени
        unique_patterns = {}
        for pattern in patterns:
            if pattern['name'] not in unique_patterns:
                unique_patterns[pattern['name']] = pattern

        # Сортируем по уверенности (Высокая -> Средняя -> Низкая)
        confidence_order = {'Высокая': 3, 'Средняя': 2, 'Низкая': 1}
        sorted_patterns = sorted(
            unique_patterns.values(),
            key=lambda x: confidence_order.get(x['confidence'], 0),
            reverse=True
        )

        return sorted_patterns

    def analyze_patterns_advanced(self, market_data: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """
        Расширенный анализ обнаруженных паттернов
        Добавляет торговые рекомендации и метрики риска
        """
        if not patterns or market_data is None:
            return []

        advanced_analysis = []

        for pattern in patterns:
            try:
                analysis = {
                    'pattern_name': pattern['name'],
                    'pattern_type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'trading_recommendations': self._generate_trading_recommendations(pattern, market_data),
                    'risk_metrics': self._calculate_risk_metrics(pattern, market_data),
                    'timeframe_suitability': self._assess_timeframe_suitability(pattern),
                    'volume_confirmation': self._check_volume_confirmation(market_data, pattern),
                    'pattern_strength': self._calculate_pattern_strength(pattern, market_data)
                }
                advanced_analysis.append(analysis)

            except Exception as e:
                logger.error(f"Ошибка анализа паттерна {pattern.get('name')}: {e}")
                continue

        return advanced_analysis

    def _generate_trading_recommendations(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """Генерация торговых рекомендаций для паттерна"""
        latest_price = df['close'].iloc[-1]

        recommendations = {
            'entry_price': self._calculate_entry_price(pattern, latest_price),
            'stop_loss': self._calculate_stop_loss(pattern, latest_price),
            'take_profit_1': self._calculate_take_profit(pattern, latest_price, 1),
            'take_profit_2': self._calculate_take_profit(pattern, latest_price, 2),
            'position_size': '1-2% от депозита',
            'risk_reward_ratio': '1:2.5',
            'timeframe': '4H-1D для разворотных, 1H-4H для продолжения'
        }

        return recommendations

    def _calculate_entry_price(self, pattern: Dict, current_price: float) -> float:
        """Расчет рекомендуемой цены входа"""
        pattern_name = pattern['name']

        if any(x in pattern_name for x in ['Молот', 'Двойное дно', 'Бычье']):
            return round(current_price * 1.002, 4)  # На 0.2% выше
        elif any(x in pattern_name for x in ['Повешенный', 'Двойная вершина', 'Медвежье']):
            return round(current_price * 0.998, 4)  # На 0.2% ниже
        else:
            return round(current_price, 4)

    def _calculate_stop_loss(self, pattern: Dict, current_price: float) -> float:
        """Расчет стоп-лосса"""
        if 'Разворотный' in pattern['type']:
            return round(current_price * 0.97, 4)  # 3% стоп-лосс
        else:
            return round(current_price * 0.98, 4)  # 2% стоп-лосс

    def _calculate_take_profit(self, pattern: Dict, current_price: float, level: int = 1) -> float:
        """Расчет тейк-профита"""
        if level == 1:
            multiplier = 1.05 if any(x in pattern['name'] for x in ['Быч', 'Молот', 'Дно']) else 0.95
        else:
            multiplier = 1.08 if any(x in pattern['name'] for x in ['Быч', 'Молот', 'Дно']) else 0.92

        return round(current_price * multiplier, 4)

    def _calculate_risk_metrics(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """Расчет метрик риска"""
        current_price = df['close'].iloc[-1]
        entry = self._calculate_entry_price(pattern, current_price)
        stop_loss = self._calculate_stop_loss(pattern, current_price)

        risk = abs(entry - stop_loss)
        reward = abs(self._calculate_take_profit(pattern, current_price, 1) - entry)
        risk_reward = reward / risk if risk > 0 else 0

        # Вероятность успеха на основе уверенности
        success_prob = 0.68 if pattern['confidence'] == 'Высокая' else 0.55 if pattern[
                                                                                   'confidence'] == 'Средняя' else 0.45

        return {
            'risk_per_trade': round(risk, 4),
            'potential_reward': round(reward, 4),
            'risk_reward_ratio': f"1:{risk_reward:.1f}" if risk_reward > 0 else "1:0",
            'success_probability': f"{success_prob * 100:.0f}%"
        }

    def _assess_timeframe_suitability(self, pattern: Dict) -> Dict:
        """Оценка подходящих таймфреймов для паттерна"""
        if 'Разворотный' in pattern['type']:
            return {'1h': 'Хорошо', '4h': 'Отлично', '1d': 'Отлично'}
        else:
            return {'1h': 'Отлично', '4h': 'Хорошо', '1d': 'Удовлетворительно'}

    def _check_volume_confirmation(self, df: pd.DataFrame, pattern: Dict) -> str:
        """Проверка подтверждения объема"""
        if len(df) < 10:
            return 'Недостаточно данных'

        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-20:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio > 1.2:
            return 'Сильное подтверждение объемом'
        elif volume_ratio > 0.8:
            return 'Умеренное подтверждение'
        else:
            return 'Слабое подтверждение'

    def _calculate_pattern_strength(self, pattern: Dict, df: pd.DataFrame) -> float:
        """Расчет силы паттерна (0-100)"""
        strength = 50

        # Базовая уверенность
        if pattern['confidence'] == 'Высокая':
            strength += 20
        elif pattern['confidence'] == 'Средняя':
            strength += 10

        # Подтверждение объемом
        volume_confirmation = self._check_volume_confirmation(df, pattern)
        if volume_confirmation == 'Сильное подтверждение объемом':
            strength += 15
        elif volume_confirmation == 'Умеренное подтверждение':
            strength += 5

        return min(100, max(0, strength))

    def get_patterns_cheatsheet(self) -> List[Dict]:
        """Возвращает шпаргалку по паттернам с образовательной информацией"""
        cheatsheet = []

        for name, info in self.pattern_descriptions.items():
            cheatsheet.append({
                "name": name,
                "type": info['type'],
                "description": info['description'],
                "reliability": info['reliability'],
                "volume_characteristics": info['volume'],
                "price_target": info['target'],
                "confirmation": info['confirmation'],
                "trading_tips": [
                    "Дождитесь полного формирования паттерна",
                    "Ищите подтверждение объемом",
                    "Устанавливайте стоп-лосс ниже/выше ключевых уровней",
                    "Целевые уровни - минимальные цели, можно брать частичную прибыль"
                ],
                "common_mistakes": [
                    "Торговля до полного формирования паттерна",
                    "Игнорирование объема",
                    "Слишком агрессивные позиции",
                    "Отсутствие стоп-лосса"
                ]
            })

        return cheatsheet


# Глобальный экземпляр для импорта
pattern_detector = PatternDetector()


# Функции для обратной совместимости
def detect_patterns(market_data: pd.DataFrame) -> List[Dict]:
    """Основная функция обнаружения паттернов"""
    return pattern_detector.detect_patterns(market_data)


def analyze_patterns_advanced(market_data: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
    """Расширенный анализ паттернов"""
    return pattern_detector.analyze_patterns_advanced(market_data, patterns)


def get_patterns_cheatsheet() -> List[Dict]:
    """Получение шпаргалки по паттернам"""
    return pattern_detector.get_patterns_cheatsheet()