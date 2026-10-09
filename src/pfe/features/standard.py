"""Vectorized OHLCV features; rolling windows count observations."""

import numpy as np
import pandas as pd

from pfe.features._utils import divide, event_age
from pfe.features.rolling import rolling_kernel


def standard_features(bars: pd.DataFrame, windows: tuple[int, ...], groups: set[str]) -> dict:
    o, h, low, c, v = (bars[name] for name in ("open", "high", "low", "close", "volume"))
    prev = c.shift()
    ret = c.pct_change(fill_method=None)
    logret = np.log(c).diff()
    body, spread = c - o, h - low
    typical = (h + low + c) / 3
    tr = pd.concat([spread, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out = {}
    if "price" in groups:
        out.update(
            {
                "price.median": (h + low) / 2,
                "price.typical": typical,
                "price.ohlc_mean": (o + h + low + c) / 4,
                "price.weighted_close": (h + low + 2 * c) / 4,
            }
        )
        for name in bars:
            out[f"price.log_{name}"] = np.log1p(v) if name == "volume" else np.log(bars[name])
    if "returns" in groups:
        out.update(
            {
                "returns.simple": ret,
                "returns.log": logret,
                "returns.absolute_change": c.diff(),
                "returns.open_close": c / o - 1,
                "returns.gap": o / prev - 1,
                "returns.acceleration": logret.diff(),
            }
        )
        out["returns.open_gap"] = out["returns.gap"]
    if "candles" in groups:
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - low
        overlap = (
            pd.concat([h, h.shift()], axis=1).min(axis=1)
            - pd.concat([low, low.shift()], axis=1).max(axis=1)
        ).clip(lower=0)
        out.update(
            {
                "candles.body": body,
                "candles.body_abs": body.abs(),
                "candles.range": spread,
                "candles.range_relative": spread / o,
                "candles.upper_wick": upper,
                "candles.lower_wick": lower,
                "candles.body_ratio": divide(body.abs(), spread),
                "candles.upper_wick_ratio": divide(upper, spread),
                "candles.lower_wick_ratio": divide(lower, spread),
                "candles.wick_asymmetry": divide(upper - lower, spread),
                "candles.close_position": divide(c - low, spread),
                "candles.open_position": divide(o - low, spread),
                "candles.direction": np.sign(body),
                "candles.overlap": overlap,
                "candles.range_expansion": divide(spread, spread.shift()),
                "candles.inside": ((h < h.shift()) & (low > low.shift())).astype(float),
                "candles.outside": ((h > h.shift()) & (low < low.shift())).astype(float),
            }
        )
    if "volatility" in groups:
        out["volatility.true_range"] = tr
    if "volume" in groups:
        out.update(
            {
                "volume.change": v.diff(),
                "volume.growth": divide(v, v.shift()) - 1,
                "volume.dollar_proxy": typical * v,
                "volume.zero": (v == 0).astype(float),
                "volume.signed_proxy": np.sign(ret) * v,
                "volume.amihud_proxy": divide(ret.abs(), typical * v),
            }
        )

    signals = {
        **{name: bars[name] for name in bars},
        "return": logret,
        "range": spread,
        "body": body,
        "upper_wick": h - np.maximum(o, c),
        "lower_wick": np.minimum(o, c) - low,
    }
    for w in windows:
        suffix = f"_w{w}"

        def put(family, name, value, suffix=suffix):
            out[f"{family}.{name}{suffix}"] = value

        if "returns" in groups:
            multi = np.log(c).diff(w)
            put("returns", "log", multi)
            put("returns", "simple", c.pct_change(w, fill_method=None))
            put("returns", "absolute", c.diff(w))
            put("returns", "vol_scaled", divide(multi, logret.rolling(w).std() * np.sqrt(w)))
        if "statistics" in groups:
            for name, signal in signals.items():
                roll = signal.rolling(w)
                values = {
                    "mean": roll.mean(),
                    "median": roll.median(),
                    "min": roll.min(),
                    "max": roll.max(),
                    "std": roll.std(),
                    "variance": roll.var(),
                    "skew": rolling_kernel(signal, w, "skew") if w >= 3 else signal * np.nan,
                    "kurtosis": rolling_kernel(signal, w, "kurtosis")
                    if w >= 4
                    else signal * np.nan,
                    "q05": roll.quantile(0.05),
                    "q25": roll.quantile(0.25),
                    "q75": roll.quantile(0.75),
                    "q95": roll.quantile(0.95),
                    "iqr": roll.quantile(0.75) - roll.quantile(0.25),
                    "rank": roll.rank(pct=True),
                    "mad": rolling_kernel(signal, w, "mad"),
                }
                values["zscore"] = divide(signal - values["mean"], values["std"])
                values["robust_zscore"] = divide(signal - values["median"], 1.4826 * values["mad"])
                for statistic, value in values.items():
                    put("statistics", f"{name}_{statistic}", value)
        if "volatility" in groups:
            roll = logret.rolling(w)
            variance = (logret**2).rolling(w).sum()
            up = logret.clip(lower=0).pow(2).rolling(w).sum()
            down = logret.clip(upper=0).pow(2).rolling(w).sum()
            rs = np.log(h / c) * np.log(h / o) + np.log(low / c) * np.log(low / o)
            parkinson = np.log(h / low).pow(2) / (4 * np.log(2))
            gk = 0.5 * np.log(h / low).pow(2) - (2 * np.log(2) - 1) * np.log(c / o).pow(2)
            overnight = np.log(o / prev)
            intraday = np.log(c / o)
            k = 0.34 / (1.34 + (w + 1) / (w - 1))
            estimators = {
                "parkinson": parkinson.rolling(w).mean(),
                "garman_klass": gk.rolling(w).mean(),
                "rogers_satchell": rs.rolling(w).mean(),
                "yang_zhang": overnight.rolling(w).var()
                + k * intraday.rolling(w).var()
                + (1 - k) * rs.rolling(w).mean(),
            }
            put("volatility", "close_std", roll.std())
            put("volatility", "realized_variance", variance)
            put("volatility", "realized", np.sqrt(variance))
            put("volatility", "up_semivariance", up)
            put("volatility", "down_semivariance", down)
            put("volatility", "semivariance_ratio", divide(up, down))
            put("volatility", "ewma", np.sqrt(logret.pow(2).ewm(span=w, min_periods=w).mean()))
            put("volatility", "atr_simple", tr.rolling(w).mean())
            put("volatility", "atr_relative", tr.rolling(w).mean() / c)
            put("volatility", "vol_of_vol", roll.std().rolling(w).std())
            put("volatility", "rank", roll.std().rolling(w).rank(pct=True))
            for name, value in estimators.items():
                put("volatility", name, np.sqrt(value.clip(lower=0)))
        if "trend" in groups:
            sma = c.rolling(w).mean()
            ema = c.ewm(span=w, adjust=False, min_periods=w).mean()
            path = c.diff().abs().rolling(w).sum()
            for name, ma in {"sma": sma, "ema": ema}.items():
                put("trend", name, ma)
                put("trend", f"distance_{name}", c / ma - 1)
                put("trend", f"{name}_slope", ma.diff())
                put("trend", f"{name}_curvature", ma.diff().diff())
                crossed = np.sign(c - ma).diff().abs() > 0
                put("trend", f"{name}_cross", crossed.astype(float).where(ma.notna()))
                put("trend", f"{name}_cross_age", event_age(crossed).where(ma.notna()))
            put("trend", "efficiency", divide(c.diff(w).abs(), path))
            put("trend", "vhf", divide(c.rolling(w).max() - c.rolling(w).min(), path))
            for name in ("slope", "intercept", "r_squared", "residual_std"):
                put(
                    "trend",
                    f"regression_{name}",
                    rolling_kernel(c, w, f"regression_{name}"),
                )
            put(
                "trend",
                "choppiness",
                100
                * np.log10(divide(tr.rolling(w).sum(), h.rolling(w).max() - low.rolling(w).min()))
                / np.log10(w),
            )
        if "levels" in groups:
            hi, lo = h.rolling(w).max(), low.rolling(w).min()
            oldhi, oldlo = hi.shift(), lo.shift()
            put("levels", "high", hi)
            put("levels", "low", lo)
            put("levels", "high_distance", c / hi - 1)
            put("levels", "low_distance", c / lo - 1)
            put("levels", "position", divide(c - lo, hi - lo))
            put("levels", "new_high", (h > oldhi).astype(float).where(oldhi.notna()))
            put("levels", "new_low", (low < oldlo).astype(float).where(oldlo.notna()))
            put("levels", "breakout_up", (c / oldhi - 1).clip(lower=0))
            put("levels", "breakout_down", (1 - c / oldlo).clip(lower=0))
            put(
                "levels",
                "high_age",
                rolling_kernel(h, w, "high_age"),
            )
            put(
                "levels",
                "low_age",
                rolling_kernel(low, w, "low_age"),
            )
            put("levels", "drawdown", c / c.rolling(w).max() - 1)
            put("levels", "drawup", c / c.rolling(w).min() - 1)
            put(
                "levels",
                "max_drawdown",
                rolling_kernel(c, w, "max_drawdown"),
            )
            put(
                "levels",
                "underwater_age",
                rolling_kernel(c, w, "high_age"),
            )
        if "volume" in groups:
            put("volume", "sum", v.rolling(w).sum())
            put("volume", "mean", v.rolling(w).mean())
            put("volume", "ema", v.ewm(span=w, min_periods=w, adjust=False).mean())
            put("volume", "std", v.rolling(w).std())
            put("volume", "relative", divide(v, v.rolling(w).mean()))
            put("volume", "rank", v.rolling(w).rank(pct=True))
            put("volume", "zscore", divide(v - v.rolling(w).mean(), v.rolling(w).std()))
            put("volume", "zero_fraction", (v == 0).astype(float).rolling(w).mean())
            put(
                "volume",
                "concentration",
                divide(v.pow(2).rolling(w).sum(), v.rolling(w).sum().pow(2)),
            )
            put("volume", "vwap_proxy", divide((typical * v).rolling(w).sum(), v.rolling(w).sum()))
            put("volume", "vwap_distance", c / out[f"volume.vwap_proxy{suffix}"] - 1)
            put("volume", "return_corr", ret.rolling(w).corr(v))
            put("volume", "absolute_return_corr", ret.abs().rolling(w).corr(v))
            put("volume", "growth_return_corr", ret.rolling(w).corr(v.diff()))
        if "sequences" in groups:
            sign = np.sign(logret)
            put(
                "sequences",
                "up_fraction",
                (logret > 0).astype(float).where(logret.notna()).rolling(w).mean(),
            )
            put(
                "sequences",
                "turn_count",
                rolling_kernel(sign, w, "turn_count"),
            )
            for name in ("up", "down"):
                put(
                    "sequences",
                    f"longest_{name}_run",
                    rolling_kernel(sign, w, f"longest_{name}_run"),
                )
            for lag in (1, 2, 3):
                put(
                    "sequences",
                    f"return_autocorr_lag{lag}",
                    logret.rolling(w).corr(logret.shift(lag)),
                )
                put(
                    "sequences",
                    f"absolute_autocorr_lag{lag}",
                    logret.abs().rolling(w).corr(logret.abs().shift(lag)),
                )
                put(
                    "sequences",
                    f"squared_autocorr_lag{lag}",
                    logret.pow(2).rolling(w).corr(logret.pow(2).shift(lag)),
                )
    if "volatility" in groups and len(windows) > 1:
        a, b = min(windows), max(windows)
        out[f"volatility.short_long_w{a}_w{b}"] = divide(
            logret.rolling(a).std(), logret.rolling(b).std()
        )
    if "trend" in groups and len(windows) > 1:
        a, b = min(windows), max(windows)
        out[f"trend.sma_spread_w{a}_w{b}"] = c.rolling(a).mean() / c.rolling(b).mean() - 1
    return out
