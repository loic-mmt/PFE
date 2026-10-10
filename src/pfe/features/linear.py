"""Walk-forward OLS/Ridge outputs with explicit label-maturity delays."""

import re
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

from pfe.features.aggregation import fixed_duration
from pfe.features.selected import _names, _validate, compute_selected_features

LINEAR_FEATURE_GROUPS = ("prediction", "fit", "coefficients", "errors")
DEFAULT_LINEAR_PREDICTORS = ("returns.log", "candles.range_relative", "volume.relative_w20")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _predictor_frame(bars, predictors, step, gap_policy):
    if not isinstance(predictors, pd.DataFrame):
        return compute_selected_features(
            bars, predictors, source_frequency=step, gap_policy=gap_policy, validate=False
        )
    if not predictors.index.equals(bars.index):
        raise ValueError("Predictors must have exactly the bars' timezone-aware index")
    if not predictors.columns.is_unique:
        raise ValueError("Predictor columns must be unique")
    if "available_at" in predictors:
        available = predictors["available_at"]
        if not isinstance(available.dtype, pd.DatetimeTZDtype) or available.isna().any():
            raise ValueError("Predictor available_at must contain timezone-aware timestamps")
        if (available > bars.index + step).any():
            raise ValueError("Predictors are not yet available at the source bar close")
        predictors = predictors.drop(columns="available_at")
    elif predictors.attrs.get("source_frequency"):
        predictor_step = fixed_duration(predictors.attrs["source_frequency"], "source_frequency")
        if predictor_step > step:
            raise ValueError("Predictors are not yet available at the source bar close")
    if not len(predictors.columns) or any(
        not isinstance(name, str)
        or not name
        or not is_numeric_dtype(predictors[name])
        or is_complex_dtype(predictors[name])
        for name in predictors
    ):
        raise ValueError("Predictors must have non-empty string names and numeric columns")
    return predictors


def _catalog(columns, horizons):
    out = {}
    for horizon in horizons:
        for group, metrics in (
            ("prediction", ("prediction", "prediction_scaled")),
            ("fit", ("train_rmse", "train_r_squared", "n_train", "model_age")),
            ("coefficients", ("intercept",)),
            ("errors", ("realized_error",)),
        ):
            for metric in metrics:
                out[f"linear.{metric}_h{horizon}"] = (horizon, group, metric)
        for i, column in enumerate(columns):
            out[f"linear.coef[{column}]_h{horizon}"] = (horizon, "coefficients", i)
    return out


def _fit(x, y, *, model, alpha, standardize, rmse, r_squared):
    """Center only the training window; constant columns receive scale one."""
    mean = x.mean(axis=0)
    centered = x - mean
    scale = np.sqrt(np.mean(centered**2, axis=0)) if standardize else np.ones(x.shape[1])
    scale = np.where(scale == 0, 1.0, scale)
    design = centered / scale
    target_mean = y.mean()
    target = y - target_mean
    if model == "ridge" and alpha > 0:
        # Augmented least squares also handles tiny penalties and collinear inputs.
        design_fit = np.vstack((design, np.sqrt(alpha) * np.eye(x.shape[1])))
        target_fit = np.r_[target, np.zeros(x.shape[1])]
        beta = np.linalg.lstsq(design_fit, target_fit, rcond=None)[0]
    else:
        # Solve the design directly: no squared condition number from normal equations.
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
    error_scale, score = np.nan, np.nan
    if rmse or r_squared:
        residual = target - design @ beta
        sse = np.dot(residual, residual)
        if rmse:
            error_scale = np.sqrt(sse / len(y))
        if r_squared:
            total = np.dot(target, target)
            score = 1 - sse / total if total > 0 else np.nan
    # Prediction uses centered coordinates to avoid cancellation with large offsets.
    raw_beta = beta / scale
    return mean, raw_beta, target_mean, target_mean - mean @ raw_beta, error_scale, score


def _horizon_features(
    x,
    close,
    valid,
    boundaries,
    horizon,
    metrics,
    *,
    train_window,
    min_train_size,
    refit_every,
    model,
    alpha,
    standardize,
):
    n = len(close)
    output = {metric: np.full(n, np.nan) for metric in metrics}
    needs_prediction = bool({"prediction", "prediction_scaled", "realized_error"} & set(metrics))
    prediction = output.get("prediction")
    if needs_prediction and prediction is None:
        prediction = np.full(n, np.nan)
    needs_fit = any(metric not in {"n_train", "model_age"} for metric in metrics)
    needs_rmse = "train_rmse" in metrics or "prediction_scaled" in metrics
    needs_score = "train_r_squared" in metrics
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start <= horizon:
            continue
        # Targets stay internal, and training below never reads labels beyond i-h.
        target = np.log(close[start + horizon : end]) - np.log(close[start : end - horizon])
        fitted, last_fit, count = None, None, None
        first_refit = start + horizon + min_train_size - 1
        for i in range(first_refit, end, refit_every):
            stop = min(i + refit_every, end)
            right = i - horizon + 1
            left = max(start, right - train_window)
            keep = valid[left:right]
            n_train = np.count_nonzero(keep)
            if n_train >= min_train_size:
                if needs_fit:
                    fitted = _fit(
                        x[left:right][keep],
                        target[left - start : right - start][keep],
                        model=model,
                        alpha=alpha,
                        standardize=standardize,
                        rmse=needs_rmse,
                        r_squared=needs_score,
                    )
                last_fit, count = i, n_train
            if last_fit is None:
                continue
            if "n_train" in output:
                output["n_train"][i:stop] = count
            if "model_age" in output:
                output["model_age"][i:stop] = np.arange(i - last_fit, stop - last_fit)
            if fitted is None:
                continue
            mean, beta, target_mean, intercept, error_scale, score = fitted
            for metric in metrics:
                if isinstance(metric, int):
                    output[metric][i:stop] = beta[metric]
                elif metric in {"intercept", "train_rmse", "train_r_squared"}:
                    output[metric][i:stop] = {
                        "intercept": intercept,
                        "train_rmse": error_scale,
                        "train_r_squared": score,
                    }[metric]
            if needs_prediction:
                block = x[i:stop]
                finite = valid[i:stop]
                values = np.full(stop - i, np.nan)
                values[finite] = (block[finite] - mean) @ beta + target_mean
                prediction[i:stop] = values
                if "prediction_scaled" in output and error_scale > 0:
                    output["prediction_scaled"][i:stop] = values / error_scale
        if "realized_error" in output:
            output["realized_error"][start + horizon : end] = (
                target - prediction[start : end - horizon]
            )
    return output


def compute_linear_features(
    bars: pd.DataFrame,
    *,
    predictors=DEFAULT_LINEAR_PREDICTORS,
    horizons=None,
    train_window: int = 250,
    min_train_size: int | None = None,
    refit_every: int = 20,
    model: str = "ridge",
    alpha: float = 1.0,
    standardize: bool = True,
    names=None,
    groups=None,
    source_frequency: str | pd.Timedelta = "1min",
    gap_policy: str = "reset",
) -> pd.DataFrame:
    """Create chronological model features, not full-sample fitted predictions.

    At row i, an h-bar target is log(close[i+h]/close[i]); training only includes
    rows j <= i-h, whose labels have matured by i's close. Training windows count
    candidate source rows. Refit cadence anchors at each contiguous block's first
    eligible row. Missing predictor rows are excluded; no warm-up filling.

    Predictors can be exact OHLCV feature names or a numeric DataFrame with the
    same index and values already available at each bar close. Explicit availability
    metadata is checked. Input feature construction must itself be causal.

    Select exact names or groups (prediction, fit, coefficients, errors). Horizons
    are inferred from selected names; otherwise default to (5,). Coefficients are
    returned in original predictor units. Training diagnostics are in-sample;
    realized_error uses the model prediction actually made h rows earlier.
    """
    train_window = _positive_integer(train_window, "train_window")
    refit_every = _positive_integer(refit_every, "refit_every")
    if model not in {"ols", "ridge"}:
        raise ValueError("model must be 'ols' or 'ridge'")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("alpha must be a finite non-negative number")
    if not isinstance(standardize, bool):
        raise ValueError("standardize must be a boolean")
    if gap_policy not in {"reset", "allow"}:
        raise ValueError("gap_policy must be 'reset' or 'allow'")
    if names is not None and groups is not None:
        raise ValueError("Choose groups or exact names, not both")
    requested = _names(names) if names is not None else None
    if horizons is None:
        horizons = (
            tuple(
                dict.fromkeys(
                    int(match[1]) for name in requested if (match := re.search(r"_h(\d+)$", name))
                )
            )
            if requested
            else (5,)
        )
    try:
        horizons = tuple(horizons)
    except TypeError as exc:
        raise ValueError("horizons must be a non-empty iterable of positive integers") from exc
    for horizon in horizons:
        _positive_integer(horizon, "horizon")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be non-empty and unique")
    if isinstance(groups, str):
        raise ValueError("groups must be an iterable of names, not a single string")
    selected_groups = set(LINEAR_FEATURE_GROUPS if groups is None else groups)
    if not selected_groups or selected_groups - set(LINEAR_FEATURE_GROUPS):
        raise ValueError(f"Unknown or empty linear feature groups: {selected_groups}")
    step = fixed_duration(source_frequency, "source_frequency")
    data, boundaries = _validate(bars, step)
    if gap_policy == "allow":
        boundaries = [0, len(data)]
    frame = _predictor_frame(data, predictors, step, gap_policy)
    columns = tuple(frame.columns)
    minimum = (
        train_window
        if min_train_size is None
        else _positive_integer(min_train_size, "min_train_size")
    )
    if minimum < len(columns) + 2 or minimum > train_window:
        raise ValueError("min_train_size must be between number of predictors + 2 and train_window")
    catalog = _catalog(columns, horizons)
    if requested is None:
        requested = tuple(
            name for name, (_, group, _) in catalog.items() if group in selected_groups
        )
    if set(requested) - set(catalog):
        raise ValueError(f"Unknown linear features: {set(requested) - set(catalog)}")
    x = frame.to_numpy(dtype=float, copy=False, na_value=np.nan)
    valid = np.empty(len(data), dtype=bool)
    for start in range(0, len(x), 100_000):
        block = x[start : start + 100_000]
        if np.isinf(block).any():
            raise ValueError("Predictors may contain warm-up NaNs but not infinities")
        valid[start : start + len(block)] = np.isfinite(block).all(axis=1)
    close = data.close.to_numpy(dtype=float, copy=False)
    output = {}
    # Only selected horizons are fitted; all outputs of one horizon share its model.
    for horizon in dict.fromkeys(catalog[name][0] for name in requested):
        selected = [name for name in requested if catalog[name][0] == horizon]
        metrics = [catalog[name][2] for name in selected]
        values = _horizon_features(
            x,
            close,
            valid,
            boundaries,
            horizon,
            metrics,
            train_window=train_window,
            min_train_size=minimum,
            refit_every=refit_every,
            model=model,
            alpha=float(alpha),
            standardize=standardize,
        )
        output.update({name: values[catalog[name][2]] for name in selected})
    result = pd.DataFrame({name: output[name] for name in requested}, index=data.index, copy=False)
    result.attrs.update(
        source_frequency=str(step),
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
        model=model,
        alpha=float(alpha) if model == "ridge" else 0.0,
        standardize=standardize,
        predictors=columns,
        horizons=horizons,
        train_window=train_window,
        min_train_size=minimum,
        refit_every=refit_every,
        gap_policy=gap_policy,
        target="future log return; labels are never exported",
    )
    return result


def compute_linear_feature(bars: pd.DataFrame, name: str, **options) -> pd.Series:
    """Compute one linear-model descriptor and only its dependencies."""
    frame = compute_linear_features(bars, names=(name,), **options)
    result = frame[name]
    result.attrs.update(frame.attrs)
    return result
