import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

# импортируем целевой модуль
import analysis.signals as signals


signals.signal_cache._cache.clear()
signals.signal_cache._timestamps.clear()

def _mk_df(n=60, price=100.0, drift=0.05, vol=0.5):
    rng = np.random.default_rng(42)
    closes = price + np.cumsum(rng.normal(drift, vol, size=n))
    highs  = closes + rng.uniform(0, 1, size=n)
    lows   = closes - rng.uniform(0, 1, size=n)
    opens  = closes + rng.normal(0, 0.2, size=n)
    vols   = rng.integers(100, 1000, size=n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})


class DummyTIBase:
    """База для моков индикаторов — подстраиваем формат под тест."""
    def __init__(self, df): self.df = df
    def ema(self, period):  # простая EMA-заглушка: среднее + небольшой наклон
        arr = self.df["close"].to_numpy()
        w = np.exp(-np.linspace(0, 3, len(arr)))
        w = w / w.sum()
        ema = np.convolve(arr, w, mode="same")
        return ema
    def rsi(self, period: int = 14):
        # rsi ~ линейный тренд вокруг 50
        n = len(self.df)
        r = np.linspace(45, 55, n)
        return r
    def bollinger_bands(self, period: int = 20, mult: float = 2.0):
        c = self.df["close"].to_numpy()
        mid = pd.Series(c).rolling(period, min_periods=1).mean().to_numpy()
        std = pd.Series(c).rolling(period, min_periods=1).std(ddof=0).to_numpy()
        up = mid + mult * std
        lo = mid - mult * std
        return up, mid, lo
    def supertrend(self, period: int = 10, multiplier: float = 2.0):
        n = len(self.df)
        line = self.df["close"].to_numpy() * 0.99
        direction = np.ones(n, dtype=int)
        direction[: n // 3] = -1
        # гарантируем разворот на самом хвосте: ...,-1, +1
        if n >= 2:
                    direction[-2] = -1
        direction[-1] = 1
        return (line, direction)


@pytest.mark.asyncio
async def test_bb_and_supertrend_unpacking_tuple(monkeypatch):
    """Проверяем: BB и Supertrend как КОРТЕЖИ распаковываются и дают сигналы."""
    class DummyTI(DummyTIBase):
        def macd(self):
            # tuple-формат MACD: сделаем явный бычий кросс на хвосте
            n = len(self.df)
            line = np.zeros(n)
            sig = np.zeros(n)
            # на предпоследнем баре линия чуть ниже сигнала,
            # на последнем — выше (кросс + подтверждение)
            if n >= 2:
                line[-2], sig[-2] = -0.01, 0.00  # prev_diff <= 0
                line[-1], sig[-1] = 0.02, 0.01  # current_diff > 0 и линия выше
            hist = line - sig
            return (line, sig, hist)

    monkeypatch.setattr(signals, "TechnicalIndicators", DummyTI)

    df = _mk_df()
    result = await signals.generate_trading_signals(df, symbol="TEST-TUPLE")

    # Должны быть распознаны какие-то сигналы, включая Supertrend/Bollinger
    kinds = [s["indicator"] for s in result["signals"]]
    assert "Supertrend" in kinds
    assert "Bollinger Bands" in kinds
    # Никаких исключений, итоговая структура валидна
    assert result["overall_signal"] in {t.value for t in signals.SignalType}


@pytest.mark.asyncio
async def test_macd_dict_format(monkeypatch):
    """Проверяем: MACD в формате СЛОВАРЯ тоже распознаётся."""
    class DummyTIDict(DummyTIBase):
        def macd(self):
            n = len(self.df)
            macd = np.zeros(n)
            sig = np.zeros(n)

            if n >= 2:
                macd[-2], sig[-2] = -0.02, -0.01  # prev_diff <= 0
                macd[-1], sig[-1] = 0.03, 0.01  # current_diff > 0
            return {"macd": macd, "signal": sig, "hist": macd - sig}

    # берём базовый Dummy для BB/ST, но MACD подменим словарём
    monkeypatch.setattr(signals, "TechnicalIndicators", DummyTIDict)

    df = _mk_df()
    out = await signals.generate_trading_signals(df, symbol="TEST-MACD-DICT")
    kinds = [s["indicator"] for s in out["signals"]]
    assert "MACD" in kinds


@pytest.mark.asyncio
async def test_confidence_neutral_when_forces_equal(monkeypatch):
    """Искуственно делаем равную силу buy/sell → NEUTRAL и низкий confidence."""
    class DummyTIFlat(DummyTIBase):
        def macd(self):
            n = len(self.df)
            # сделаем линию почти равной сигналу -> слабый MACD
            base = np.zeros(n)
            return (base, base, base)
        def bollinger_bands(self, period: int = 20, mult: float = 2.0):
            # сузим полосы, чтобы сигналы BB были с обеих сторон по очереди
            c = self.df["close"].to_numpy()
            mid = np.full_like(c, c.mean())
            std = np.full_like(c, 0.0001)  # почти 0
            up = mid + mult * std
            lo = mid - mult * std
            return up, mid, lo
        def supertrend(self, period: int = 10, multiplier: float = 2.0):
            n = len(self.df)
            line = self.df["close"].to_numpy()
            # направление туда-сюда, чтобы баланс был близок
            direction = np.ones(n, dtype=int)
            direction[::2] = -1
            return (line, direction)

    # И подкрутим режим/веса, чтобы уравнять влияние
    monkeypatch.setattr(signals, "TechnicalIndicators", DummyTIFlat)
    monkeypatch.setattr(signals, "determine_market_regime", lambda prices, volumes=None, lookback=50: signals.MarketRegime.RANGING)
    monkeypatch.setattr(signals, "get_regime_weights", lambda regime, symbol=None: signals.SignalWeightConfig(
        ema_weight=1, macd_weight=1, rsi_weight=1, bollinger_weight=1, volume_weight=1, trend_weight=1, supertrend_weight=1
    ))

    df = _mk_df()
    out = await signals.generate_trading_signals(df, symbol="TEST-NEUTRAL")
    assert out["overall_signal"] in (signals.SignalType.NEUTRAL.value, signals.SignalType.WATCH.value)
    # confidence_score должен быть низким (около 0)
    assert out["confidence_score"] <= 0.5  # допустим мягкий порог


def test_signal_cache_key_handles_bytes():
    cache = signals.SignalCache()
    key = cache.get_key("unit", b"\x00\x01\x02", {"a": 1})
    assert isinstance(key, str) and len(key) > 0
