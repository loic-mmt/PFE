"""Trailing-window distribution, spectral and complexity descriptors."""

import numpy as np
import pandas as pd
import pywt
from scipy.spatial.distance import pdist

from pfe.features._utils import entropy, ratio, regression


def describe_window(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    centered = x - x.mean()
    std = float(x.std())
    power = np.abs(np.fft.rfft(centered)) ** 2
    power[0] = 0
    total = float(power.sum())
    out = {
        "spectral_energy": total / n**2,
        "spectral_entropy": entropy(power),
        "dominant_frequency": float(np.fft.rfftfreq(n)[np.argmax(power)]) if total else np.nan,
        "low_frequency_share": ratio(float(power[: max(2, len(power) // 3)].sum()), total),
        "sign_entropy": entropy(np.bincount((np.sign(x).astype(int) + 1), minlength=3)),
        "zero_crossings": float(np.count_nonzero(np.diff(np.sign(centered)))),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
        "expected_shortfall_05": float(x[x <= np.quantile(x, 0.05)].mean()),
        "tail_fraction": float(np.mean(np.abs(centered) > 2 * std)) if std else 0.0,
        "permutation_entropy": np.nan,
        "sample_entropy": np.nan,
        "hurst": np.nan,
        "ar1": np.nan,
        "pacf_lag2": np.nan,
        "petrosian_dimension": np.nan,
    }
    coefficients = np.fft.rfft(centered) / n
    for k in range(1, 4):
        out[f"fft_real_k{k}"] = float(coefficients[k].real) if k < len(coefficients) else np.nan
        out[f"fft_imag_k{k}"] = float(coefficients[k].imag) if k < len(coefficients) else np.nan
    if n >= 3:
        embeddings = np.lib.stride_tricks.sliding_window_view(x, 3)
        permutations = np.argsort(embeddings, axis=1, kind="stable")
        _, counts = np.unique(permutations, axis=0, return_counts=True)
        out["permutation_entropy"] = entropy(counts)
        changes = np.count_nonzero(np.diff(np.sign(np.diff(x))))
        out["petrosian_dimension"] = float(
            np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * changes)))
        )
        for lag, name in ((1, "ar1"), (2, "pacf_lag2")):
            if n > 2 * lag:
                design = np.column_stack(
                    [np.ones(n - lag), *(x[lag - k : n - k] for k in range(1, lag + 1))]
                )
                if np.linalg.matrix_rank(design) == lag + 1:
                    coef = np.linalg.lstsq(design, x[lag:], rcond=None)[0]
                    out[name] = float(coef[-1])
    if n >= 4:
        if std == 0:
            out["sample_entropy"] = 0.0
        else:
            # Same number of embeddings at orders 2 and 3; self-pairs excluded.
            count = n - 2
            a = np.lib.stride_tricks.sliding_window_view(x, 2)[:count]
            b = np.lib.stride_tricks.sliding_window_view(x, 3)
            matches_a = np.count_nonzero(pdist(a, metric="chebyshev") <= 0.2 * std)
            matches_b = np.count_nonzero(pdist(b, metric="chebyshev") <= 0.2 * std)
            if matches_a and matches_b:
                out["sample_entropy"] = float(-np.log(matches_b / matches_a))
    if n >= 8:
        lags = np.arange(2, min(20, n // 2) + 1)
        scale = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        valid = scale > 0
        if valid.sum() >= 2:
            out["hurst"] = float(np.polyfit(np.log(lags[valid]), np.log(scale[valid]), 1)[0])
    # Haar wavelet detail energies; no padding beyond the finite past window.
    approximation = x.copy()
    for level in range(1, 4):
        if len(approximation) < 2:
            out[f"wavelet_detail_energy_l{level}"] = np.nan
            out[f"wavelet_detail_first_l{level}"] = np.nan
        else:
            approximation, detail = pywt.dwt(approximation, "haar", mode="periodization")
            out[f"wavelet_detail_energy_l{level}"] = float(np.mean(detail**2))
            out[f"wavelet_detail_first_l{level}"] = float(detail[0])
    slope = regression(x)["slope"]
    out["trend_slope"] = slope
    return out


def complexity_features(bars: pd.DataFrame, windows: tuple[int, ...]) -> dict:
    out = {}
    signals = {
        "return": np.log(bars.close).diff(),
        "volume": np.log1p(bars.volume),
        "price": np.log(bars.close),
    }
    for signal_name, signal in signals.items():
        values = signal.to_numpy()
        for w in windows:
            records = []
            for end in range(w - 1, len(values)):
                x = values[end - w + 1 : end + 1]
                records.append(describe_window(x) if np.isfinite(x).all() else {})
            # Get a stable schema even when the input is shorter than the lookback.
            keys = describe_window(np.zeros(w)).keys()
            frame = pd.DataFrame(records, index=bars.index[w - 1 :], columns=list(keys)).reindex(
                bars.index
            )
            for name in frame:
                out[f"complexity.{signal_name}_{name}_w{w}"] = frame[name]
    return out


_complexity_features_python = complexity_features


def complexity_features(bars, windows):
    """Whole-array C++ windows for supported descriptors; native FFT/LAPACK retained."""
    from pfe.features.backend import using_cpp

    if not using_cpp():
        return _complexity_features_python(bars, windows)
    from pfe.features.selected import _complexity

    keys = describe_window(np.zeros(4))
    names = tuple(
        f"complexity.{signal}_{key}_w{w}"
        for signal in ("return", "volume", "price")
        for w in windows
        for key in keys
    )
    out = _complexity(bars, names)
    return {name: out[name] for name in names}
