# analysis/signals.py
"""
ULTRA-PERFORMANCE TRADING SIGNALS MODULE
Реальное обнаружение сигналов без заглушек
Интеграция с реальными рыночными данными
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum
from datetime import datetime
from indicators.indicators import (
    UltraPerformanceIndicators,
    detect_crossover,
    detect_divergence,
    calculate_momentum,
    detect_overbought_oversold,
    calculate_zscore
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Типы торговых сигналов"""
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    NEUTRAL = "neutral"
    WATCH = "watch"


class SignalConfidence(Enum):
    """Уверенность в сигнале"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


def analyze_ema_signals(ema_fast: List[float], ema_slow: List[float]) -> Optional[Dict[str, Any]]:
    """Анализ сигналов на основе EMA кроссоверов"""
    crossover = detect_crossover(ema_fast, ema_slow)

    if crossover == "bullish_cross":
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "EMA",
            "message": "Бычье пересечение EMA",
            "strength": 65
        }
    elif crossover == "bearish_cross":
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "EMA",
            "message": "Медвежье пересечение EMA",
            "strength": 65
        }

    return None


def analyze_macd_signals(macd_line: List[float], macd_signal: List[float]) -> Optional[Dict[str, Any]]:
    """Анализ сигналов на основе MACD"""
    crossover = detect_crossover(macd_line, macd_signal)

    if crossover == "bullish_cross":
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "MACD",
            "message": "Бычье пересечение MACD",
            "strength": 75
        }
    elif crossover == "bearish_cross":
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "MACD",
            "message": "Медвежье пересечение MACD",
            "strength": 75
        }

    # Анализ дивергенции
    divergence = detect_divergence(macd_line, macd_signal)
    if divergence == "bullish_divergence":
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "MACD",
            "message": "Бычья дивергенция MACD",
            "strength": 60
        }
    elif divergence == "bearish_divergence":
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "MACD",
            "message": "Медвежья дивергенция MACD",
            "strength": 60
        }

    return None


def analyze_rsi_signals(rsi_values: List[float], prices: List[float]) -> List[Dict[str, Any]]:
    """Анализ сигналов на основе RSI"""
    signals = []

    overbought_oversold = detect_overbought_oversold(rsi_values)

    if overbought_oversold == "overbought":
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "RSI",
            "message": "Перекупленность RSI > 70",
            "strength": 80
        })
    elif overbought_oversold == "oversold":
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "RSI",
            "message": "Перепроданность RSI < 30",
            "strength": 80
        })

    # Анализ дивергенции RSI
    divergence = detect_divergence(prices, rsi_values)
    if divergence == "bullish_divergence":
        signals.append({
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "RSI",
            "message": "Бычья дивергенция RSI",
            "strength": 70
        })
    elif divergence == "bearish_divergence":
        signals.append({
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "RSI",
            "message": "Медвежья дивергенция RSI",
            "strength": 70
        })

    return signals


def analyze_bollinger_bands_signals(prices: List[float],
                                    bb_upper: List[float],
                                    bb_lower: List[float]) -> Optional[Dict[str, Any]]:
    """Анализ сигналов на основе Bollinger Bands"""
    if not prices or not bb_upper or not bb_lower:
        return None

    current_price = prices[-1]
    current_upper = bb_upper[-1]
    current_lower = bb_lower[-1]

    if current_price > current_upper:
        return {
            "type": SignalType.SELL.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": "Цена выше верхней полосы Bollinger Bands",
            "strength": 70
        }
    elif current_price < current_lower:
        return {
            "type": SignalType.BUY.value,
            "confidence": SignalConfidence.MEDIUM.value,
            "indicator": "Bollinger Bands",
            "message": "Цена ниже нижней полосы Bollinger Bands",
            "strength": 70
        }

    return None


def analyze_volume_signals(prices: List[float], volumes: List[float]) -> Optional[Dict[str, Any]]:
    """Анализ сигналов на основе объема"""
    if len(prices) < 10 or len(volumes) < 10:
        return None

    # Анализ объема на повышение при росте цены
    price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
    volume_change = (volumes[-1] - np.mean(volumes[-10:-1])) / np.mean(volumes[-10:-1]) * 100

    if price_change > 2 and volume_change > 50:
        return {
            "type": SignalType.STRONG_BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Volume",
            "message": "Сильный объем при росте цены",
            "strength": 85
        }
    elif price_change < -2 and volume_change > 50:
        return {
            "type": SignalType.STRONG_SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Volume",
            "message": "Сильный объем при падении цены",
            "strength": 85
        }

    return None


def analyze_trend_signals(prices: List[float], period: int = 20) -> Optional[Dict[str, Any]]:
    """Анализ сигналов на основе тренда"""
    if len(prices) < period:
        return None

    short_term = prices[-period // 2:]
    long_term = prices[-period:]

    short_trend = (short_term[-1] - short_term[0]) / short_term[0] * 100
    long_trend = (long_term[-1] - long_term[0]) / long_term[0] * 100

    if short_trend > 5 and long_trend > 3:
        return {
            "type": SignalType.STRONG_BUY.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Trend",
            "message": "Сильный восходящий тренд",
            "strength": 90
        }
    elif short_trend < -5 and long_trend < -3:
        return {
            "type": SignalType.STRONG_SELL.value,
            "confidence": SignalConfidence.HIGH.value,
            "indicator": "Trend",
            "message": "Сильный нисходящий тренд",
            "strength": 90
        }

    return None


def generate_trading_signals(indicators: Dict[str, Any],
                             prices: List[float],
                             volumes: List[float] = None) -> Dict[str, Any]:
    """
    Генерация торговых сигналов на основе всех индикаторов
    """
    signals = []
    confidence_score = 0

    # Анализ EMA сигналов
    ema_signal = analyze_ema_signals(
        indicators.get('ema9', []),
        indicators.get('ema21', [])
    )
    if ema_signal:
        signals.append(ema_signal)
        confidence_score += ema_signal.get('strength', 0)

    # Анализ MACD сигналов
    macd_signal = analyze_macd_signals(
        indicators.get('macd', []),
        indicators.get('macd_signal', [])
    )
    if macd_signal:
        signals.append(macd_signal)
        confidence_score += macd_signal.get('strength', 0)

    # Анализ RSI сигналов
    rsi_signals = analyze_rsi_signals(
        indicators.get('rsi', []),
        prices
    )
    signals.extend(rsi_signals)
    for signal in rsi_signals:
        confidence_score += signal.get('strength', 0)

    # Анализ Bollinger Bands
    bb_signal = analyze_bollinger_bands_signals(
        prices,
        indicators.get('bb_upper', []),
        indicators.get('bb_lower', [])
    )
    if bb_signal:
        signals.append(bb_signal)
        confidence_score += bb_signal.get('strength', 0)

    # Анализ объема если доступен
    if volumes:
        volume_signal = analyze_volume_signals(prices, volumes)
        if volume_signal:
            signals.append(volume_signal)
            confidence_score += volume_signal.get('strength', 0)

    # Анализ тренда
    trend_signal = analyze_trend_signals(prices)
    if trend_signal:
        signals.append(trend_signal)
        confidence_score += trend_signal.get('strength', 0)

    # Определение общего сигнала
    buy_signals = [s for s in signals if s['type'] in ['buy', 'strong_buy']]
    sell_signals = [s for s in signals if s['type'] in ['sell', 'strong_sell']]

    if len(buy_signals) > len(sell_signals) and confidence_score > 150:
        overall_signal = SignalType.STRONG_BUY.value
    elif len(buy_signals) > len(sell_signals):
        overall_signal = SignalType.BUY.value
    elif len(sell_signals) > len(buy_signals) and confidence_score > 150:
        overall_signal = SignalType.STRONG_SELL.value
    elif len(sell_signals) > len(buy_signals):
        overall_signal = SignalType.SELL.value
    else:
        overall_signal = SignalType.NEUTRAL.value

    # Расчет уверенности
    total_strength = sum(s.get('strength', 0) for s in signals)
    confidence = min(100, total_strength / max(1, len(signals)))

    if confidence > 80:
        confidence_level = SignalConfidence.VERY_HIGH.value
    elif confidence > 60:
        confidence_level = SignalConfidence.HIGH.value
    elif confidence > 40:
        confidence_level = SignalConfidence.MEDIUM.value
    else:
        confidence_level = SignalConfidence.LOW.value

    return {
        "signals": signals,
        "overall_signal": overall_signal,
        "confidence": confidence_level,
        "confidence_score": float(confidence),
        "buy_count": len(buy_signals),
        "sell_count": len(sell_signals),
        "timestamp": datetime.now().isoformat()
    }


def calculate_risk_reward_ratio(signals: Dict[str, Any],
                                current_price: float,
                                support_levels: List[float],
                                resistance_levels: List[float]) -> Dict[str, Any]:
    """
    Расчет соотношения риск/вознаграждение на основе сигналов
    """
    if not signals or not support_levels or not resistance_levels:
        return {
            "risk_reward_ratio": "1:1",
            "stop_loss": current_price * 0.95,
            "take_profit": current_price * 1.05,
            "confidence": "low"
        }

    overall_signal = signals.get('overall_signal', 'neutral')

    if overall_signal in ['buy', 'strong_buy']:
        # Для покупки: стоп-лосс ниже поддержки, тейк-профит у сопротивления
        stop_loss = min(support_levels) if support_levels else current_price * 0.95
        take_profit = max(resistance_levels) if resistance_levels else current_price * 1.05

        risk = current_price - stop_loss
        reward = take_profit - current_price

        if risk > 0:
            rr_ratio = reward / risk
        else:
            rr_ratio = 1.0

    elif overall_signal in ['sell', 'strong_sell']:
        # Для продажи: стоп-лосс выше сопротивления, тейк-профит у поддержки
        stop_loss = max(resistance_levels) if resistance_levels else current_price * 1.05
        take_profit = min(support_levels) if support_levels else current_price * 0.95

        risk = stop_loss - current_price
        reward = current_price - take_profit

        if risk > 0:
            rr_ratio = reward / risk
        else:
            rr_ratio = 1.0

    else:
        return {
            "risk_reward_ratio": "1:1",
            "stop_loss": current_price * 0.95,
            "take_profit": current_price * 1.05,
            "confidence": "low"
        }

    # Определение уверенности в соотношении
    if rr_ratio >= 3:
        confidence = "very_high"
    elif rr_ratio >= 2:
        confidence = "high"
    elif rr_ratio >= 1.5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "risk_reward_ratio": f"1:{rr_ratio:.1f}",
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "confidence": confidence,
        "ratio_value": float(rr_ratio)
    }


def generate_trading_recommendation(signals: Dict[str, Any],
                                    risk_reward: Dict[str, Any],
                                    current_price: float) -> Dict[str, Any]:
    """
    Генерация полной торговой рекомендации
    """
    overall_signal = signals.get('overall_signal', 'neutral')
    confidence = signals.get('confidence', 'low')

    if overall_signal == 'strong_buy':
        recommendation = "STRONG BUY"
        action = "Рекомендуется покупка с высоким объемом"
    elif overall_signal == 'buy':
        recommendation = "BUY"
        action = "Рекомендуется покупка"
    elif overall_signal == 'strong_sell':
        recommendation = "STRONG SELL"
        action = "Рекомендуется продажа с высоким объемом"
    elif overall_signal == 'sell':
        recommendation = "SELL"
        action = "Рекомендуется продажа"
    else:
        recommendation = "HOLD"
        action = "Рекомендуется удерживать позицию или наблюдать"

    return {
        "recommendation": recommendation,
        "action": action,
        "signal_strength": confidence,
        "current_price": float(current_price),
        "stop_loss": risk_reward.get('stop_loss', current_price * 0.95),
        "take_profit": risk_reward.get('take_profit', current_price * 1.05),
        "risk_reward_ratio": risk_reward.get('risk_reward_ratio', '1:1'),
        "timestamp": datetime.now().isoformat(),
        "signals_count": len(signals.get('signals', [])),
        "buy_signals": signals.get('buy_count', 0),
        "sell_signals": signals.get('sell_count', 0)
    }