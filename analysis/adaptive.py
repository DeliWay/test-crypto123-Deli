"""
АДАПТИВНЫЙ МОДУЛЬ ДЛЯ ТЕХНИЧЕСКОГО АНАЛИЗА V3
Исправлены edge-cases, улучшена стабильность метрик
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class AssetProfile:
    """Профиль характеристик торгового актива"""
    symbol: str
    volatility_atr: float  # Волатильность на основе ATR (% от цены)
    volatility_std: float  # Волатильность на основе стандартного отклонения (%)
    liquidity_score: float  # Оценка ликвидности (0-1)
    trend_strength: float  # Сила тренда (0-1)
    noise_level: float  # Уровень шума (0-1) на основе нормализованного ATR
    rsi_percentile_high: float  # 90-й перцентиль RSI за период
    rsi_percentile_low: float  # 10-й перцентиль RSI за период


def _calculate_rsi_corrected(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Корректный расчет RSI с обработкой edge-cases по стандарту:
    - RSI = 100, если avg_loss == 0 И есть рост (avg_gain > 0)
    - RSI = 0, если avg_gain == 0 И есть падение (avg_loss > 0)
    - RSI = 50, если оба нуля (плоская полка)
    """
    n = len(prices)
    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # Initial values
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Wilder's smoothing (EMA)
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # Обработка edge-cases
    rsi = np.full(n, 50.0)  # По умолчанию 50

    for i in range(period, n):
        if avg_loss[i] < 1e-10:  # Нет потерь
            if avg_gain[i] > 1e-10:  # Есть рост
                rsi[i] = 100.0
            else:  # Нет ни роста, ни потерь (плоская полка)
                rsi[i] = 50.0
        elif avg_gain[i] < 1e-10:  # Нет роста
            rsi[i] = 0.0  # Есть потери (avg_loss > 0)
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _calculate_atr_wilder(highs: np.ndarray, lows: np.ndarray,
                          closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Расчет ATR по методу Уайлдера (EMA сглаживание) с защитой от коротких рядов"""
    n = len(highs)
    if n < period:
        return np.zeros(n)  # Возвращаем нули вместо NaN

    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Wilder's EMA smoothing
    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
    else:
        atr[-1] = np.mean(tr[1:])  # Для очень коротких рядов

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def _normalize_liquidity(volumes: Optional[np.ndarray], lookback: int = 100) -> float:
    """
    Перцентильная нормализация ликвидности.
    Возвращает 0-1, где 1 = исторически высокая ликвидность
    """
    if volumes is None or len(volumes) < lookback:
        return 0.5  # Фолбэк значение

    try:
        recent_volumes = volumes[-lookback:]

        # Рассчитываем перцентили
        p5 = np.percentile(recent_volumes, 5)
        p50 = np.percentile(recent_volumes, 50)
        p95 = np.percentile(recent_volumes, 95)

        if p95 - p5 < 1e-10:  # Все объемы одинаковые
            return 0.5

        # Нормализуем текущий средний объем относительно исторического диапазона
        current_avg = np.mean(recent_volumes[-20:])  # Средний объем последних 20 баров
        liquidity_score = (current_avg - p5) / (p95 - p5)

        return float(np.clip(liquidity_score, 0.0, 1.0))
    except:
        return 0.5  # Фолбэк при любой ошибке


def calculate_asset_profile(symbol: str, highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray, volumes: Optional[np.ndarray],
                            lookback_period: int = 100) -> AssetProfile:
    """
    Улучшенный расчет характеристик актива с дополнительной защитой
    """
    n = len(closes)
    if n < 20:  # Уменьшил минимальный порог с 50 до 20
        return AssetProfile(
            symbol=symbol,
            volatility_atr=2.0,
            volatility_std=1.5,
            liquidity_score=0.5,
            trend_strength=0.5,
            noise_level=0.5,
            rsi_percentile_high=70.0,
            rsi_percentile_low=30.0
        )

    # Защита от NaN/Inf значений
    closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
    lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Волатильность на основе ATR (%) по Уайлдеру ---
    # Защита от коротких рядов для ATR
    if n < 14:
        atr_values = np.zeros(n)
        recent_atr = 0.0
    else:
        atr_values = _calculate_atr_wilder(highs, lows, closes, period=14)
        valid_atr = atr_values[~np.isnan(atr_values)]
        if len(valid_atr) >= 20:
            recent_atr = np.mean(valid_atr[-20:])
        elif len(valid_atr) > 0:
            recent_atr = valid_atr.mean()
        else:
            recent_atr = 0.0

    # Безопасный расчет волатильности ATR
    if closes[-1] > 1e-10 and np.isfinite(recent_atr):
        volatility_atr = (recent_atr / closes[-1]) * 100
    else:
        volatility_atr = 2.0

    # --- Волатильность на основе Std Dev доходностей (%) ---
    if n > 1:
        returns = np.diff(closes) / closes[:-1]
        # Используем nanstd для игнорирования NaN/Inf
        volatility_std = float(np.nanstd(returns[np.isfinite(returns)])) * 100
    else:
        volatility_std = 1.5

    # --- Ликвидность (перцентильная нормализация) ---
    # Защита от отсутствия данных по объему
    if volumes is None or len(volumes) == 0:
        liquidity_score = 0.5
    else:
        try:
            liquidity_score = _normalize_liquidity(volumes, lookback_period)
        except:
            liquidity_score = 0.5

    # --- Сила тренда ---
    trend_strength = 0.5  # Значение по умолчанию
    if n >= 20:
        lookback_trend = min(50, n)
        x = np.arange(lookback_trend)
        y = closes[-lookback_trend:]

        slope_strength = 0.0
        try:
            slope = np.polyfit(x, y, 1)[0]
            slope_strength = min(1.0, abs(slope) / (np.mean(y) * 0.01))
        except:
            slope_strength = 0.0

        # Расчет автокорреляции с защитой от NaN
        rets = np.diff(closes[-lookback_trend:]) / closes[-lookback_trend:-1]
        autocorr_strength = 0.0

        if len(rets) > 5:
            # Фильтруем нечисловые значения
            finite_rets = rets[np.isfinite(rets)]
            if len(finite_rets) > 5:
                try:
                    # Безопасный расчет корреляции
                    autocorr = np.corrcoef(finite_rets[:-1], finite_rets[1:])[0, 1]
                    if np.isfinite(autocorr):
                        autocorr_strength = abs(autocorr)
                except:
                    autocorr_strength = 0.0

        trend_strength = max(0.0, min(1.0, (slope_strength * 0.7 + autocorr_strength * 0.3)))

    # --- Уровень шума ---
    # Защита от нечислового recent_atr и нулевого диапазона
    recent_high = np.max(highs[-20:])
    recent_low = np.min(lows[-20:])
    price_range = recent_high - recent_low

    if price_range > 1e-10 and np.isfinite(recent_atr):
        noise_level = min(1.0, recent_atr / price_range)
    else:
        noise_level = 0.5

    # --- Исторические перцентили RSI (с корректным расчетом) ---
    rsi_values = _calculate_rsi_corrected(closes, period=14)
    valid_rsi = rsi_values[~np.isnan(rsi_values)]

    if len(valid_rsi) >= lookback_period:
        recent_rsi = valid_rsi[-lookback_period:]
        rsi_percentile_high = np.percentile(recent_rsi, 90)
        rsi_percentile_low = np.percentile(recent_rsi, 10)
    else:
        rsi_percentile_high = 70.0
        rsi_percentile_low = 30.0

    return AssetProfile(
        symbol=symbol,
        volatility_atr=volatility_atr,
        volatility_std=volatility_std,
        liquidity_score=liquidity_score,
        trend_strength=trend_strength,
        noise_level=noise_level,
        rsi_percentile_high=rsi_percentile_high,
        rsi_percentile_low=rsi_percentile_low
    )

def adaptive_rsi_levels(asset_profile: AssetProfile) -> Tuple[float, float]:
    """
    Адаптивные уровни перекупленности/перепроданности для RSI.
    """
    overbought_base = asset_profile.rsi_percentile_high
    oversold_base = asset_profile.rsi_percentile_low

    trend_adjustment = (asset_profile.trend_strength - 0.5) * 20
    overbought = overbought_base + trend_adjustment
    oversold = oversold_base - trend_adjustment

    overbought = min(85.0, max(60.0, overbought))
    oversold = max(15.0, min(40.0, oversold))

    return overbought, oversold


def adaptive_supertrend_multiplier(asset_profile: AssetProfile) -> float:
    """
    Адаптивный мультипликатор для SuperTrend.
    """
    base_multiplier = 2.5
    vola_factor = asset_profile.volatility_atr / 2.0
    vola_adjustment = (vola_factor - 1.0) * 0.5
    noise_adjustment = (asset_profile.noise_level - 0.5) * 1.0
    total_adjustment = vola_adjustment + noise_adjustment
    multiplier = base_multiplier + total_adjustment

    return max(1.5, min(5.0, multiplier))

def _adaptive_stop_multiplier(asset_profile: AssetProfile,
                              base: float = 1.8,
                              min_mult: float = 1.0,
                              max_mult: float = 4.0) -> float:
    """
    Адаптивный множитель для стоп-лосса.
    ↑ при высокой волатильности/шуме/неликвидности, ↓ при сильном тренде.

    Нормализации выбраны так, чтобы без тюнинга давать здравые значения на крипте.
    При желании подкрутить чувствительность — меняй веса и нормализацию ниже.
    """
    # --- Нормализации (все в диапазон 0..1) ---
    # ATR% обычно 0.5..5 на ликвидных парах → нормализуем к этому коридору
    vola_n = np.clip((asset_profile.volatility_atr - 0.5) / (5.0 - 0.5), 0.0, 1.0)

    # Шум уже 0..1 по определению (recent_ATR / price_range)
    noise_n = np.clip(asset_profile.noise_level, 0.0, 1.0)

    # Ликвидность 0..1, но риски растут при нИЗКОЙ ликвидности → инвертируем
    illiq_n = np.clip(1.0 - asset_profile.liquidity_score, 0.0, 1.0)

    # Сила тренда 0..1; чем сильнее тренд, тем чуть уже стоп:
    # используем только «положительную» часть (выше 0.5)
    trend_pos = max(0.0, asset_profile.trend_strength - 0.5) * 2.0  # 0..1

    # --- Веса влияния (подкручиваются) ---
    w_vola = 0.8
    w_noise = 0.6
    w_illiq = 0.6
    w_trend = 0.4  # уменьшает множитель

    delta = w_vola * vola_n + w_noise * noise_n + w_illiq * illiq_n - w_trend * trend_pos
    mult = base + delta
    return float(np.clip(mult, min_mult, max_mult))


def adaptive_stop_loss_take_profit(current_price: float,
                                  asset_profile: AssetProfile,
                                  trend_direction: Optional[int] = None,
                                  risk_reward_ratio: float = 2.0) -> Tuple[float, float]:
    """
    Адаптивные уровни SL/TP в абсолютных ценах.
    Теперь стоп-множитель тоже адаптивный.
    """
    # Безопасности на случай кривых данных
    if current_price <= 0 or not np.isfinite(current_price):
        return 0.0, 0.0

    atr_value = (asset_profile.volatility_atr / 100.0) * current_price

    # ← Новое: подбираем множитель стопа из профиля
    stop_loss_atr_multiplier = _adaptive_stop_multiplier(asset_profile)
    take_profit_atr_multiplier = stop_loss_atr_multiplier * risk_reward_ratio

    stop_loss_distance = atr_value * stop_loss_atr_multiplier
    take_profit_distance = atr_value * take_profit_atr_multiplier

    # Автоподбор направления, если не задан
    if trend_direction is None:
        trend_direction = 1 if asset_profile.trend_strength > 0.5 else -1

    if trend_direction > 0:
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + take_profit_distance
    else:
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - take_profit_distance

    return stop_loss, take_profit


class AdaptiveTechnicalAnalysis:
    """Класс для адаптивного технического анализа"""

    def __init__(self, symbol: str, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, volumes: np.ndarray):
        self.symbol = symbol
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes

        self.asset_profile = calculate_asset_profile(
            symbol, highs, lows, closes, volumes
        )

    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """Получение всех адаптивных параметров с улучшенным определением тренда"""
        rsi_overbought, rsi_oversold = adaptive_rsi_levels(self.asset_profile)
        supertrend_multiplier = adaptive_supertrend_multiplier(self.asset_profile)

        current_price = self.closes[-1] if len(self.closes) > 0 else 0

        # Улучшенное определение направления тренда
        trend_direction = self._calculate_trend_direction()

        suggested_stop, suggested_target = adaptive_stop_loss_take_profit(
            current_price, self.asset_profile, trend_direction=trend_direction
        )

        return {
            # Основные параметры
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'supertrend_multiplier': supertrend_multiplier,
            'trend_direction': trend_direction,  # Добавляем для отладки

            # Новые имена (рекомендуемые)
            'suggested_stop': suggested_stop,
            'suggested_target': suggested_target,

            # Полный профиль для аналитики
            'asset_profile': {
                'symbol': self.asset_profile.symbol,
                'volatility_atr': self.asset_profile.volatility_atr,
                'volatility_std': self.asset_profile.volatility_std,
                'liquidity_score': self.asset_profile.liquidity_score,
                'trend_strength': self.asset_profile.trend_strength,
                'noise_level': self.asset_profile.noise_level,
                'rsi_percentile_high': self.asset_profile.rsi_percentile_high,
                'rsi_percentile_low': self.asset_profile.rsi_percentile_low
            }
        }

    def _calculate_trend_direction(self) -> int:
        """Определение направления тренда на основе последних цен"""
        if len(self.closes) < 20:
            return 1  # По умолчанию восходящий

        # Анализ последних 20 свечей
        recent_prices = self.closes[-20:]
        x = np.arange(len(recent_prices))

        try:
            # Линейная регрессия для определения наклона
            slope, intercept = np.polyfit(x, recent_prices, 1)

            # Определяем направление по наклону
            price_change_percent = (slope * len(recent_prices)) / recent_prices[0] * 100

            if abs(price_change_percent) < 2:  # Боковой тренд
                # Используем силу тренда из профиля
                return 1 if self.asset_profile.trend_strength > 0.5 else -1
            elif price_change_percent > 0:
                return 1  # Восходящий
            else:
                return -1  # Нисходящий

        except:
            # Фолбэк на профиль актива
            return 1 if self.asset_profile.trend_strength > 0.5 else -1