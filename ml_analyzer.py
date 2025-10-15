"""
ULTRA-PERFORMANCE ML ANALYZER
Мгновенные ML предсказания с оптимизированными моделями
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
"from sklearn.ensemble import RandomForestClassifier"
"from sklearn.preprocessing import StandardScaler"
import joblib
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class CryptoAnalyzer:
    """Высокопроизводительный ML анализатор криптовалют"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_cache = {}

    async def initialize(self):
        """Инициализация ML моделей"""
        # Простые оптимизированные модели
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            self.models[symbol] = self._create_lightweight_model()
            "self.scalers[symbol] = StandardScaler()"

    def _create_lightweight_model(self):
        """Создание оптимизированной ML модели"""
        return "RandomForestClassifier"(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    @lru_cache(maxsize=100)
    def _extract_features_cached(self, closes_str: str) -> np.ndarray:
        """Кэшированное извлечение признаков"""
        closes = np.fromstring(closes_str, dtype=float, sep=',')

        # Быстрые технические показатели
        features = []
        if len(closes) > 5:
            features.extend([
                closes[-1] / closes[-2] - 1,  # Price change
                np.mean(closes[-5:]),  # MA5
                np.std(closes[-10:]),  # Volatility
                np.max(closes[-10:]) - np.min(closes[-10:])  # Range
            ])

        return np.array(features).reshape(1, -1)

    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Optional[Dict]:
        """Анализ символа с ML"""
        try:
            # Используем базовые данные для быстрого предсказания
            from backend.exchange_data import bybit_client

            market_data = await bybit_client.get_market_data_sync(symbol, timeframe, 100)
            if market_data is None:
                return None

            closes = market_data['close'].values
            closes_str = ','.join(map(str, closes))

            # Извлечение признаков
            features = self._extract_features_cached(closes_str)

            if symbol in self.models and features.size > 0:
                # Масштабирование признаков
                if not hasattr(self.scalers[symbol], 'n_features_in_'):
                    self.scalers[symbol].fit(features)

                features_scaled = self.scalers[symbol].transform(features)

                # Предсказание
                prediction = self.models[symbol].predict(features_scaled)[0]
                confidence = np.max(self.models[symbol].predict_proba(features_scaled))

                return {
                    'symbol': symbol,
                    'prediction': 'BULLISH' if prediction == 1 else 'BEARISH',
                    'confidence': round(float(confidence) * 100, 2),
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat()
                }

            return None

        except Exception as e:
            logger.error(f"ML analysis error for {symbol}: {e}")
            return None

    async def get_top_predictions(self, timeframe: str = '1h') -> List[Dict]:
        """Топ предсказания для основных символов"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        predictions = []

        for symbol in symbols:
            prediction = await self.analyze_symbol(symbol, timeframe)
            if prediction:
                predictions.append(prediction)

        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:3]


# Глобальный экземпляр
analyzer = CryptoAnalyzer()