"""TA-Lib's complete OHLCV indicator and candlestick catalog, plus extensions."""

import numpy as np
import pandas as pd
import talib
from talib import abstract

from pfe.features._utils import divide, event_age, wma
from pfe.features.rolling import rolling_kernel

TA_GROUPS = (
    "Cycle Indicators",
    "Momentum Indicators",
    "Overlap Studies",
    "Price Transform",
    "Statistic Functions",
    "Volatility Indicators",
    "Volume Indicators",
)


def talib_features(bars: pd.DataFrame, windows: tuple[int, ...], *, patterns: bool = False) -> dict:
    out = {}
    inputs = {name: bars[name].to_numpy(dtype=float) for name in bars}
    groups = talib.get_function_groups()
    names = groups["Pattern Recognition"] if patterns else [n for g in TA_GROUPS for n in groups[g]]
    for name in names:
        original = abstract.Function(name)
        adjustable = any("period" in key for key in original.parameters)
        for w in windows if adjustable and not patterns else (None,):
            fn = abstract.Function(name)
            params = dict(fn.parameters)
            if w is not None:
                for key in params:
                    if "period" in key:
                        params[key] = w
                for key in ("fastperiod", "fastk_period"):
                    if key in params:
                        params[key] = max(2, w // 2)
                if "slowperiod" in params:
                    params["slowperiod"] = max(3, w)
                for key in ("signalperiod", "slowk_period", "slowd_period", "fastd_period"):
                    if key in params:
                        params[key] = max(2, w // 3)
                for key, value in (
                    ("timeperiod1", max(2, w // 2)),
                    ("timeperiod2", w),
                    ("timeperiod3", 2 * w),
                ):
                    if key in params:
                        params[key] = value
                if "minperiod" in params:
                    params["minperiod"] = 2
            fn.set_parameters(params)
            data = {**inputs, "periods": np.full(len(bars), w or 14, dtype=float)}
            result = fn(data)
            arrays = result if isinstance(result, (list, tuple)) else [result]
            for label, values in zip(fn.output_names, arrays, strict=True):
                family = "patterns" if patterns else "technical"
                suffix = "" if w is None else f"_w{w}"
                output = "" if len(arrays) == 1 else f"_{label}"
                key = f"{family}.{name.lower()}{output}{suffix}"
                series = pd.Series(np.asarray(values, dtype=float), index=bars.index)
                series.iloc[: fn.lookback] = np.nan
                out[key] = series
                if patterns:
                    event = series != 0
                    event = event.where(series.notna(), False)
                    out[f"{key}_age"] = event_age(event).where(series.notna())
                    for period in windows:
                        out[f"{key}_frequency_w{period}"] = (
                            event.astype(float).where(series.notna()).rolling(period).mean()
                        )
    return out


def adaptive_mean(series: pd.Series, alpha: pd.Series, window: int) -> pd.Series:
    values, rates = series.to_numpy(), alpha.to_numpy()
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return pd.Series(result, index=series.index)
    state = float(np.mean(values[:window]))
    result[window - 1] = state
    for i in range(window, len(values)):
        if np.isfinite(rates[i]):
            state += rates[i] * (values[i] - state)
            result[i] = state
    return pd.Series(result, index=series.index)


def extensions(bars: pd.DataFrame, windows: tuple[int, ...]) -> dict:
    o, h, low, c, v = (bars[x] for x in ("open", "high", "low", "close", "volume"))
    delta = c.diff()
    clv = divide(2 * c - h - low, h - low)
    out = {}
    out["technical.pvt"] = (c.pct_change(fill_method=None) * v).fillna(0).cumsum()
    for name, condition in (("nvi", v < v.shift()), ("pvi", v > v.shift())):
        out[f"technical.{name}"] = (
            1 + c.pct_change(fill_method=None).where(condition, 0).fillna(0)
        ).cumprod() * 1000
    mid = (h + low) / 2
    ease = divide(mid.diff() * (h - low), v)
    out["technical.market_facilitation_index"] = divide(h - low, v)
    for w in windows:

        def put(name, value, window=w):
            out[f"technical.{name}_w{window}"] = value

        alpha = pd.Series(1 / w, index=bars.index)
        rma = adaptive_mean(c, alpha, w)
        ema = c.ewm(span=w, adjust=False, min_periods=w).mean()
        alma_weights = np.exp(-((np.arange(w) - 0.85 * (w - 1)) ** 2) / (2 * (w / 6) ** 2))
        alma_weights /= alma_weights.sum()
        hma = wma(2 * wma(c, max(1, w // 2)) - wma(c, w), max(1, int(np.sqrt(w))))
        zlema = (2 * c - c.shift((w - 1) // 2)).ewm(span=w, adjust=False, min_periods=w).mean()
        cmo = divide(
            delta.clip(lower=0).rolling(w).sum() + delta.clip(upper=0).rolling(w).sum(),
            delta.abs().rolling(w).sum(),
        )
        vidya = adaptive_mean(c, (2 / (w + 1)) * cmo.abs().fillna(0), w)
        fw = 2 * max(1, (w + 1) // 2)
        half = fw // 2
        n1 = (h.rolling(half).max() - low.rolling(half).min()).shift(half) / half
        n2 = (h.rolling(half).max() - low.rolling(half).min()) / half
        n3 = (h.rolling(fw).max() - low.rolling(fw).min()) / fw
        dimension_ratio = divide(n1 + n2, n3)
        dimension = np.log2(dimension_ratio.where(dimension_ratio > 0)).clip(1, 2)
        frama_alpha = np.exp(-4.6 * (dimension - 1)).clip(0.01, 1).fillna(0.01)
        frama = adaptive_mean(c, frama_alpha, fw)
        for name, value in {
            "rma": rma,
            "alma": rolling_kernel(c, w, "dot", alma_weights),
            "hma": hma,
            "zlema": zlema,
            "vidya": vidya,
            "frama": frama,
        }.items():
            put(name, value)
            put(f"distance_{name}", c / value - 1)
        smoothed = (
            delta.ewm(span=w, min_periods=w, adjust=False)
            .mean()
            .ewm(span=max(2, w // 2), min_periods=max(2, w // 2), adjust=False)
            .mean()
        )
        magnitude = (
            delta.abs()
            .ewm(span=w, min_periods=w, adjust=False)
            .mean()
            .ewm(span=max(2, w // 2), min_periods=max(2, w // 2), adjust=False)
            .mean()
        )
        put("tsi", 100 * divide(smoothed, magnitude))
        tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(
            axis=1
        )
        put("vortex_plus", divide((h - low.shift()).abs().rolling(w).sum(), tr.rolling(w).sum()))
        put("vortex_minus", divide((low - h.shift()).abs().rolling(w).sum(), tr.rolling(w).sum()))
        normalized = (
            2 * divide(mid - mid.rolling(w).min(), mid.rolling(w).max() - mid.rolling(w).min()) - 1
        )
        normalized = normalized.clip(-0.999, 0.999)
        put("fisher", 0.5 * np.log((1 + normalized) / (1 - normalized)))
        osc_range = h.rolling(w).max() - low.rolling(w).min()
        osc_mid = (h.rolling(w).max() + low.rolling(w).min()) / 2
        smooth_numerator = (
            (c - osc_mid)
            .ewm(span=3, adjust=False, min_periods=3)
            .mean()
            .ewm(span=3, adjust=False, min_periods=3)
            .mean()
        )
        smooth_range = (
            osc_range.ewm(span=3, adjust=False, min_periods=3)
            .mean()
            .ewm(span=3, adjust=False, min_periods=3)
            .mean()
        )
        put("stochastic_momentum", 200 * divide(smooth_numerator, smooth_range))
        kline = 100 * divide(c - low.rolling(w).min(), osc_range)
        kline = kline.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
        dline = kline.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
        put("kdj_k", kline)
        put("kdj_d", dline)
        put("kdj_j", 3 * kline - 2 * dline)
        fast = c.ewm(span=max(2, w // 2), adjust=False, min_periods=max(2, w // 2)).mean()
        macd = fast - ema
        stc = 100 * divide(
            macd - macd.rolling(w).min(), macd.rolling(w).max() - macd.rolling(w).min()
        )
        stc = stc.ewm(alpha=0.5, adjust=False, min_periods=2).mean()
        stc = 100 * divide(stc - stc.rolling(w).min(), stc.rolling(w).max() - stc.rolling(w).min())
        put("schaff_trend_cycle", stc.ewm(alpha=0.5, adjust=False, min_periods=2).mean())
        range_ema = (h - low).ewm(span=w, adjust=False, min_periods=w).mean()
        put("chaikin_volatility", range_ema.pct_change(w, fill_method=None))
        mass_ema = (h - low).ewm(span=9, adjust=False, min_periods=9).mean()
        put(
            "mass_index",
            divide(mass_ema, mass_ema.ewm(span=9, adjust=False, min_periods=9).mean())
            .rolling(w)
            .sum(),
        )
        dispersion = c.rolling(w).std(ddof=0)
        upvol = (
            dispersion.where(delta >= 0, 0)
            .where(dispersion.notna())
            .ewm(alpha=1 / w, adjust=False, min_periods=w)
            .mean()
        )
        totalvol = dispersion.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
        put("relative_volatility_index", 100 * divide(upvol, totalvol))
        put("awesome_oscillator", mid.rolling(max(2, w // 2)).mean() - mid.rolling(w).mean())
        roc = c.pct_change(w, fill_method=None) + c.pct_change(
            max(2, int(w * 0.75)), fill_method=None
        )
        put("coppock", wma(100 * roc, max(2, w // 2)))
        put("force_index", (delta * v).ewm(span=w, min_periods=w, adjust=False).mean())
        put("ease_of_movement", ease.rolling(w).mean())
        put("cmf", divide((clv * v).rolling(w).sum(), v.rolling(w).sum()))
        put(
            "volume_oscillator",
            v.ewm(span=max(2, w // 2), min_periods=max(2, w // 2), adjust=False).mean()
            / v.ewm(span=w, min_periods=w, adjust=False).mean().replace(0, np.nan)
            - 1,
        )
        put("keltner_middle", ema)
        atr = pd.Series(
            talib.ATR(h.to_numpy(), low.to_numpy(), c.to_numpy(), timeperiod=w), index=bars.index
        )
        upper, lower = ema + 2 * atr, ema - 2 * atr
        put("keltner_upper", upper)
        put("keltner_lower", lower)
        put("keltner_position", divide(c - lower, upper - lower))
        put("keltner_width", divide(upper - lower, ema))
        donhigh, donlow = h.rolling(w).max(), low.rolling(w).min()
        put("donchian_upper", donhigh)
        put("donchian_lower", donlow)
        put("donchian_middle", (donhigh + donlow) / 2)
        put("donchian_width", divide(donhigh - donlow, c))
        std = c.rolling(w).std(ddof=0)
        put("bollinger_percent_b", divide(c - (c.rolling(w).mean() - 2 * std), 4 * std))
        put("bollinger_width", 4 * std / c.rolling(w).mean())
        put("squeeze", (2 * std < 2 * atr).astype(float).where(std.notna() & atr.notna()))
        put("envelope_upper", ema * 1.02)
        put("envelope_lower", ema * 0.98)
        put("dpo_causal", c - c.rolling(w).mean().shift(w // 2 + 1))
        # Connors RSI: streak RSI + close RSI + rolling return percentile.
        changes = np.sign(delta.fillna(0).to_numpy())
        streak = np.zeros(len(c))
        for i in range(1, len(c)):
            streak[i] = streak[i - 1] + changes[i] if changes[i] == changes[i - 1] else changes[i]
        rsi = talib.RSI(c.to_numpy(), timeperiod=max(2, w // 3))
        streak_rsi = talib.RSI(streak, timeperiod=max(2, w // 5))
        rank = c.pct_change(fill_method=None).rolling(w).rank(pct=True)
        put("connors_rsi", (rsi + streak_rsi + rank * 100) / 3)
        # KST uses four causal rate-of-change / smoothing pairs.
        kst = pd.Series(0.0, index=bars.index)
        for multiple in (1, 2, 3, 4):
            kst += multiple * c.pct_change(multiple * w, fill_method=None).rolling(w).mean()
        put("kst", 100 * kst)
        put("elder_bull_power", h - ema)
        put("elder_bear_power", low - ema)
        # Supertrend carries state forward only; no repainting past values.
        basic_upper, basic_lower = mid + 3 * atr, mid - 3 * atr
        fu, fl = basic_upper.to_numpy().copy(), basic_lower.to_numpy().copy()
        direction = np.full(len(c), np.nan)
        line = np.full(len(c), np.nan)
        closes = c.to_numpy()
        for i in range(len(c)):
            if not np.isfinite(fu[i]):
                continue
            if i == 0 or not np.isfinite(direction[i - 1]):
                direction[i] = 1
            else:
                if fu[i] >= fu[i - 1] and closes[i - 1] <= fu[i - 1]:
                    fu[i] = fu[i - 1]
                if fl[i] <= fl[i - 1] and closes[i - 1] >= fl[i - 1]:
                    fl[i] = fl[i - 1]
                direction[i] = (
                    (1 if closes[i] > fu[i - 1] else -1)
                    if direction[i - 1] < 0
                    else (-1 if closes[i] < fl[i - 1] else 1)
                )
            line[i] = fl[i] if direction[i] > 0 else fu[i]
        put("supertrend", line)
        put("supertrend_direction", direction)
        tenkan = (h.rolling(w).max() + low.rolling(w).min()) / 2
        kijun = (h.rolling(2 * w).max() + low.rolling(2 * w).min()) / 2
        put("ichimoku_tenkan", tenkan)
        put("ichimoku_kijun", kijun)
        put("ichimoku_span_a_visible", ((tenkan + kijun) / 2).shift(2 * w))
        put(
            "ichimoku_span_b_visible",
            ((h.rolling(4 * w).max() + low.rolling(4 * w).min()) / 2).shift(2 * w),
        )
    # Heikin-Ashi uses previous transformed open/close, not future bars.
    ha_close = (o + h + low + c) / 4
    ha_open = np.empty(len(c))
    if len(c):
        ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2
        for i in range(1, len(c)):
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2
    out.update(
        {
            "technical.heikin_ashi_open": ha_open,
            "technical.heikin_ashi_close": ha_close,
            "technical.heikin_ashi_high": np.maximum(h, np.maximum(ha_open, ha_close)),
            "technical.heikin_ashi_low": np.minimum(low, np.minimum(ha_open, ha_close)),
        }
    )
    return out
