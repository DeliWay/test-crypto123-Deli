# analysis/adaptive.py
"""
Адаптивные параметры для сигналов.
- Никаких внешних зависимостей, только numpy.
- Совместимо с analysis/signals.py (сигнатуры строго выдержаны).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================ utils ============================

def _as_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    return a


def _rsi_wilder(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI по Уайлдеру с корректной обработкой edge-cases.
    Возвращает массив той же длины (NaN до прогрева).
    """
    n = len(prices)
    if n < 2:
        return np.full(n, np.nan)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(n, np.nan)
    if n <= period:
        return rsi

    # стартовые средние
    avg_gain = np.empty(n)
    avg_loss = np.empty(n)
    avg_gain[:], avg_loss[:] = np.nan, np.nan

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # сглаживание Уайлдера
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    for i in range(period, n):
        g, l = avg_gain[i], avg_loss[i]
        if l < 1e-12:
            rsi[i] = 100.0 if g > 1e-12 else 50.0
        elif g < 1e-12:
            rsi[i] = 0.0
        else:
            rs = g / l
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_wilder(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR по Уайлдеру. Возвращает массив длины n, NaN до прогрева.
    """
    n = len(highs)
    tr = np.full(n, np.nan)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = np.nanmean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _pct_volatility(prices: np.ndarray, lookback: int = 60) -> float:
    """
    Волатильность в % как std дневных (барных) доходностей за lookback.
    """
    if len(prices) < 3:
        return 0.0
    p = prices[-lookback:] if len(prices) >= lookback else prices
    rets = np.diff(p) / np.where(p[:-1] != 0, p[:-1], np.nan)
    vol = float(np.nanstd(rets) * 100.0) if rets.size else 0.0
    return 0.0 if not np.isfinite(vol) else vol


def _liquidity_score(volumes: Optional[np.ndarray], lookback: int = 100) -> float:
    """
    0..1 — где 1 означает «исторически высокая ликвидность» на выбранном окне.
    """
    if volumes is None or len(volumes) < max(20, lookback//2):
        return 0.5
    v = volumes[-lookback:] if len(volumes) >= lookback else volumes
    p5, p95 = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    cur = np.nanmean(v[-20:]) if len(v) >= 20 else np.nanmean(v)
    if not np.isfinite(p5) or not np.isfinite(p95) or abs(p95 - p5) < 1e-12:
        return 0.5
    score = (cur - p5) / (p95 - p5)
    return float(np.clip(score, 0.0, 1.0))


def _trend_strength(prices: np.ndarray, lookback: int = 40) -> float:
    """
    Сила тренда 0..1: комбинация наклона линрег + направленной доли движений.
    """
    n = len(prices)
    if n < 10:
        return 0.5
    p = prices[-lookback:] if n >= lookback else prices
    x = np.arange(len(p))
    try:
        slope, _ = np.polyfit(x, p, 1)
        drift = (p[-1] - p[0]) / max(1e-12, abs(p[0]))
        up_moves = np.sum(np.diff(p) > 0)
        dir_ratio = up_moves / (len(p) - 1)
        # нормализация
        slope_n = np.tanh((slope / max(1e-12, np.nanstd(p))) * 0.5) * 0.5 + 0.5
        drift_n = np.tanh(drift * 2.0) * 0.5 + 0.5
        strength = 0.5 * slope_n + 0.3 * drift_n + 0.2 * dir_ratio
        return float(np.clip(strength, 0.0, 1.0))
    except Exception:
        return 0.5


def _noise_level(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 20) -> float:
    """
    Уровень «шума» 0..1: средний ATR на окно относительно среднего диапазона бара.
    """
    if len(closes) < 5:
        return 0.5
    atr = _atr_wilder(highs, lows, closes, period=14)
    atr_seg = atr[-lookback:] if len(atr) >= lookback else atr
    rng = (highs - lows)
    rng_seg = rng[-lookback:] if len(rng) >= lookback else rng
    mean_atr = float(np.nanmean(atr_seg))
    mean_rng = float(np.nanmean(rng_seg))
    if not np.isfinite(mean_atr) or not np.isfinite(mean_rng) or mean_rng < 1e-12:
        return 0.5
    ratio = np.clip(mean_atr / mean_rng, 0.0, 1.5)  # ограничим, чтобы не улетало
    return float(np.clip(ratio / 1.5, 0.0, 1.0))


def _rsi_percentiles(prices: np.ndarray, period: int = 14, lookback: int = 200) -> Tuple[float, float]:
    rsi = _rsi_wilder(prices, period=period)
    r = rsi[-lookback:] if len(rsi) >= lookback else rsi
    vals = r[~np.isnan(r)]
    if vals.size < 5:
        return 70.0, 30.0
    return float(np.nanpercentile(vals, 90)), float(np.nanpercentile(vals, 10))


# ============================ model ============================

@dataclass
class AssetProfile:
    """
    Профиль актива; значения подготовлены для быстрого принятия решений.
    Все величины — уже нормализованные/интерпретируемые метрики.
    """
    symbol: str
    atr: float                    # последний ATR в абсолютных единицах
    volatility: float             # std(returns) в %
    avg_volume: float             # средний объём за ~20 баров
    volatility_atr: float         # ATR как % от текущей цены
    volatility_std: float         # то же, что volatility (дублируем для обратной совместимости)
    liquidity_score: float        # 0..1 (выше = ликвиднее)
    trend_strength: float         # 0..1 (выше = тренд сильнее)
    noise_level: float            # 0..1 (выше = шума больше)
    rsi_percentile_high: float    # 90-й перцентиль RSI (исторический контекст)
    rsi_percentile_low: float     # 10-й перцентиль RSI

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "atr": self.atr,
            "volatility": self.volatility,
            "avg_volume": self.avg_volume,
            "volatility_atr": self.volatility_atr,
            "volatility_std": self.volatility_std,
            "liquidity_score": self.liquidity_score,
            "trend_strength": self.trend_strength,
            "noise_level": self.noise_level,
            "rsi_percentile_high": self.rsi_percentile_high,
            "rsi_percentile_low": self.rsi_percentile_low,
        }


# ============================ API expected by signals.py ============================

def calculate_asset_profile(
    symbol,
    highs,
    lows,
    prices=None,   # поддерживаем имя `prices`
    closes=None,   # и альтернативу `closes`
    volumes=None
) -> AssetProfile:
    """
    Унифицированный расчёт профиля. Безопасен к коротким рядам/NaN.
    """
    highs = _as_float_array(highs)
    lows  = _as_float_array(lows)
    if prices is None and closes is not None:
        prices = closes
    prices = _as_float_array(prices)
    volumes = _as_float_array(volumes) if volumes is not None else None

    # Защита от пустых/коротких рядов
    if highs is None or lows is None or prices is None or len(prices) < 5:
        return AssetProfile(
            symbol=str(symbol),
            atr=0.0, volatility=0.0, avg_volume=0.0,
            volatility_atr=0.0, volatility_std=0.0,
            liquidity_score=0.5, trend_strength=0.5, noise_level=0.5,
            rsi_percentile_high=70.0, rsi_percentile_low=30.0
        )

    n = len(prices)
    highs = highs[-n:]
    lows  = lows[-n:]

    # Базовые метрики
    atr_series = _atr_wilder(highs, lows, prices, period=14)
    atr_val = float(atr_series[~np.isnan(atr_series)][-1]) if np.any(~np.isnan(atr_series)) else 0.0
    last_price = float(prices[-1]) if np.isfinite(prices[-1]) else (float(np.nanmean(prices)) or 1.0)

    vola_pct = _pct_volatility(prices, lookback=60)
    vola_atr_pct = float((atr_val / last_price) * 100.0) if last_price > 0 else 0.0
    liq = _liquidity_score(volumes, lookback=100)
    trend = _trend_strength(prices, lookback=40)
    noise = _noise_level(highs, lows, prices, lookback=20)
    rsi_hi, rsi_lo = _rsi_percentiles(prices, period=14, lookback=200)

    avg_vol = float(np.nanmean(volumes[-20:])) if volumes is not None and len(volumes) >= 1 else 0.0

    return AssetProfile(
        symbol=str(symbol),
        atr=atr_val,
        volatility=vola_pct,
        avg_volume=avg_vol,
        volatility_atr=vola_atr_pct,
        volatility_std=vola_pct,
        liquidity_score=liq,
        trend_strength=trend,
        noise_level=noise,
        rsi_percentile_high=rsi_hi,
        rsi_percentile_low=rsi_lo,
    )


def adaptive_rsi_levels(asset_profile: AssetProfile) -> Tuple[float, float]:
    """
    Адаптивные RSI-уровни. База — исторические перцентили, корректируем силой тренда.
    """
    base_hi = asset_profile.rsi_percentile_high
    base_lo = asset_profile.rsi_percentile_low
    # тренд усиливает растяжку уровней
    adj = (asset_profile.trend_strength - 0.5) * 20.0  # -10..+10
    overbought = np.clip(base_hi + adj, 60.0, 85.0)
    oversold   = np.clip(base_lo - adj, 15.0, 40.0)
    return float(overbought), float(oversold)


def adaptive_supertrend_multiplier(asset_profile: AssetProfile) -> float:
    """
    Мультипликатор SuperTrend как функция ATR% (волы) и шума рынка.
    """
    base = 2.5
    vola_adj  = (asset_profile.volatility_atr / 2.0 - 1.0) * 0.5   # околонулевая при ATR% ≈ 2
    noise_adj = (asset_profile.noise_level - 0.5) * 1.0
    mult = base + vola_adj + noise_adj
    return float(np.clip(mult, 1.5, 5.0))


def _adaptive_stop_multiplier(asset_profile: AssetProfile,
                              base: float = 1.8,
                              min_mult: float = 1.0,
                              max_mult: float = 4.0) -> float:
    """
    Адаптивный множитель для стопа: ↑ при волатильности/шуме/неликвидности, ↓ при сильном тренде.
    """
    vola_n  = np.clip((asset_profile.volatility_atr - 0.5) / (5.0 - 0.5), 0.0, 1.0)  # ATR% ~0.5..5
    noise_n = np.clip(asset_profile.noise_level, 0.0, 1.0)
    illiq_n = np.clip(1.0 - asset_profile.liquidity_score, 0.0, 1.0)
    trend_pos = max(0.0, asset_profile.trend_strength - 0.5) * 2.0  # 0..1
    delta = 0.8 * vola_n + 0.6 * noise_n + 0.6 * illiq_n - 0.4 * trend_pos
    return float(np.clip(base + delta, min_mult, max_mult))


def adaptive_stop_loss_take_profit(asset_profile: AssetProfile,
                                   entry_price: float,
                                   position_type: str) -> Tuple[float, float, float]:
    """
    Совместимо с signals.py: (asset_profile, entry_price, position_type) -> (SL, TP, RR).
    RR выбираем 2.0, а сам стоп — адаптивный через профиль.
    """
    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0.0, 0.0, 1.0

    rr = 2.0
    # ATR в абсолюте из ATR% профиля
    atr_abs = (asset_profile.volatility_atr / 100.0) * entry_price
    stop_mult = _adaptive_stop_multiplier(asset_profile)
    sl_dist = atr_abs * stop_mult
    tp_dist = sl_dist * rr

    long_side = position_type.lower().startswith("buy") or position_type.lower().startswith("long")
    if long_side:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    return float(sl), float(tp), float(rr)
