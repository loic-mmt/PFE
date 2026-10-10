"""Numerical references, label maturity, independent execution and chronological invariance."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    LINEAR_FEATURE_GROUPS,
    compute_linear_feature,
    compute_linear_features,
    compute_selected_features,
    normalize_bars,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(35)
    n = 100
    x = rng.normal(size=n)
    increments = 0.0002 + 0.001 * x[:-1]
    close = 100 * np.exp(np.r_[0, np.cumsum(increments)])
    opening = np.r_[100.0, close[:-1]]
    bars = normalize_bars(
        pd.DataFrame(
            dict(
                open=opening,
                high=np.maximum(opening, close) + 0.1,
                low=np.minimum(opening, close) - 0.1,
                close=close,
                volume=100 + np.arange(n),
            ),
            index=pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
        )
    )
    return bars, pd.DataFrame({"x": x}, index=bars.index)


@pytest.mark.parametrize("standardize", [True, False])
def test_ols_recovers_known_forward_relation_and_original_units(sample, standardize):
    bars, x = sample
    result = compute_linear_features(
        bars,
        predictors=x,
        horizons=(1,),
        train_window=12,
        refit_every=3,
        model="ols",
        standardize=standardize,
    )
    assert result.iloc[:12].isna().all().all()
    assert result["linear.coef[x]_h1"].iloc[12:].to_numpy() == pytest.approx(0.001, abs=1e-13)
    assert result["linear.intercept_h1"].iloc[12:].to_numpy() == pytest.approx(0.0002, abs=1e-13)
    assert result["linear.prediction_h1"].iloc[12:].to_numpy() == pytest.approx(
        0.0002 + 0.001 * x.x.iloc[12:].to_numpy(), abs=1e-13
    )
    assert result["linear.n_train_h1"].iloc[12:].eq(12).all()
    assert result["linear.model_age_h1"].iloc[12:18].tolist() == [0, 1, 2, 0, 1, 2]
    assert result["linear.train_rmse_h1"].iloc[12:].max() < 1e-13
    assert result["linear.realized_error_h1"].iloc[:13].isna().all()
    assert result["linear.realized_error_h1"].iloc[13:].abs().max() < 1e-13


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("alpha", [0.0, 0.7, 20.0])
def test_ridge_against_augmented_least_squares_reference(sample, standardize, alpha):
    bars, frame = sample
    frame = frame.assign(second=np.sin(np.arange(len(frame)) / 7))
    result = compute_linear_features(
        bars,
        predictors=frame,
        horizons=(3,),
        train_window=15,
        refit_every=4,
        alpha=alpha,
        standardize=standardize,
    )
    logs = np.log(bars.close.to_numpy())
    for i in range(17, len(bars), 4):
        right, left = i - 3 + 1, i - 3 + 1 - 15
        train = frame.iloc[left:right].to_numpy()
        target = logs[left + 3 : right + 3] - logs[left:right]
        mean, target_mean = train.mean(axis=0), target.mean()
        scale = train.std(axis=0, ddof=0) if standardize else np.ones(2)
        design = (train - mean) / scale
        augmented = np.vstack([design, np.sqrt(alpha) * np.eye(2)])
        values = np.r_[target - target_mean, np.zeros(2)]
        beta = np.linalg.lstsq(augmented, values, rcond=None)[0] / scale
        residual = target - ((train - mean) @ beta + target_mean)
        assert result["linear.coef[x]_h3"].iloc[i] == pytest.approx(beta[0], abs=1e-12)
        assert result["linear.coef[second]_h3"].iloc[i] == pytest.approx(beta[1], abs=1e-12)
        assert result["linear.prediction_h3"].iloc[i] == pytest.approx(
            (frame.iloc[i].to_numpy() - mean) @ beta + target_mean, abs=1e-12
        )
        assert result["linear.train_rmse_h3"].iloc[i] == pytest.approx(
            np.sqrt(np.mean(residual**2))
        )
        assert result["linear.train_r_squared_h3"].iloc[i] == pytest.approx(
            1 - np.sum(residual**2) / np.sum((target - target_mean) ** 2)
        )


@pytest.mark.parametrize("model", ["ols", "ridge"])
def test_future_bars_and_future_predictors_do_not_change_past_outputs(sample, model):
    bars, x = sample
    options = dict(horizons=(1, 5), train_window=13, min_train_size=8, refit_every=4, model=model)
    before_bars, before_x = bars.copy(deep=True), x.copy(deep=True)
    for end in (1, 9, 18, 41, 76):
        prefix = compute_linear_features(bars.iloc[:end], predictors=x.iloc[:end], **options)
        future_bars, future_x = bars.copy(), x.copy()
        future_bars.iloc[end:] *= 3
        future_x.iloc[end:] += 1000
        full = compute_linear_features(future_bars, predictors=future_x, **options)
        assert_frame_equal(prefix, full.iloc[:end], atol=1e-12, rtol=1e-10)
    assert_frame_equal(before_bars, bars)
    assert_frame_equal(before_x, x)


def test_unmatured_labels_are_excluded_even_after_large_price_move(sample):
    bars, x = sample
    options = dict(names=("linear.prediction_h5",), train_window=12, refit_every=1)
    reference = compute_linear_features(bars, predictors=x, **options)
    altered = bars.copy()
    altered.iloc[41:] *= 100
    actual = compute_linear_features(altered, predictors=x, **options)
    assert_frame_equal(reference.iloc[:41], actual.iloc[:41])
    assert actual.iloc[41, 0] != pytest.approx(reference.iloc[41, 0])


@pytest.mark.parametrize("group", LINEAR_FEATURE_GROUPS)
def test_independent_names_and_groups_match_without_other_horizons(sample, group):
    bars, x = sample
    options = dict(predictors=x, horizons=(1, 5), train_window=12, refit_every=3)
    full = compute_linear_features(bars, **options)
    grouped = compute_linear_features(bars, groups=(group,), **options)
    assert_frame_equal(grouped, full[grouped.columns])
    for name in grouped:
        one = compute_linear_feature(bars, name, **options)
        assert_series_equal(one, full[name])


def test_selection_skips_solver_diagnostics_and_other_horizons(sample, monkeypatch):
    import pfe.features.linear as linear

    bars, x = sample

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrequested model fit executed")

    original = linear._fit
    monkeypatch.setattr(linear, "_fit", forbidden)
    compute_linear_features(
        bars, predictors=x, names=("linear.n_train_h1", "linear.model_age_h1"), train_window=12
    )
    calls = []

    def tracked(*args, **kwargs):
        assert kwargs["rmse"] is False and kwargs["r_squared"] is False
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(linear, "_fit", tracked)
    compute_linear_features(
        bars,
        predictors=x,
        horizons=(1, 5),
        names=("linear.prediction_h1",),
        train_window=12,
        refit_every=3,
    )
    assert len(calls) == len(range(12, len(bars), 3))
    calls.clear()
    compute_linear_features(
        bars,
        predictors=x,
        names=("linear.prediction_h1", "linear.coef[x]_h1"),
        train_window=12,
        refit_every=3,
    )
    assert len(calls) == len(range(12, len(bars), 3))


def test_realized_error_uses_actual_old_forecast_not_refitted_model(sample):
    bars, x = sample
    result = compute_linear_features(
        bars, predictors=x, horizons=(5,), train_window=12, refit_every=1, alpha=30
    )
    target = np.log(bars.close.to_numpy()[5:]) - np.log(bars.close.to_numpy()[:-5])
    expected = target - result["linear.prediction_h5"].to_numpy()[:-5]
    np.testing.assert_allclose(
        result["linear.realized_error_h5"].to_numpy()[5:], expected, equal_nan=True
    )


def test_gaps_reset_models_label_maturity_and_error_history(sample):
    bars, x = sample
    bars.index = bars.index[:50].append(bars.index[50:] + pd.Timedelta("1D"))
    x.index = bars.index
    options = dict(horizons=(5,), train_window=12, refit_every=3)
    reset = compute_linear_features(bars, predictors=x, **options)
    restarted = compute_linear_features(bars.iloc[50:], predictors=x.iloc[50:], **options)
    assert_frame_equal(reset.iloc[50:], restarted)
    assert reset.iloc[50:66].isna().all().all()
    allow = compute_linear_features(bars, predictors=x, gap_policy="allow", **options)
    assert np.isfinite(allow["linear.prediction_h5"].iloc[50])


def test_nan_predictors_and_insufficient_training_preserve_old_model(sample):
    bars, x = sample
    x.iloc[20:35] = np.nan
    result = compute_linear_features(
        bars, predictors=x, horizons=(1,), train_window=12, min_train_size=8, refit_every=1
    )
    assert result["linear.prediction_h1"].iloc[20:35].isna().all()
    assert result["linear.model_age_h1"].iloc[35] > 0
    assert result["linear.n_train_h1"].iloc[35] == result["linear.n_train_h1"].iloc[25]
    assert np.isfinite(result["linear.prediction_h1"].iloc[35])


@pytest.mark.parametrize("model", ["ols", "ridge"])
def test_constant_and_collinear_predictors_and_flat_prices(sample, model):
    bars, x = sample
    predictors = x.assign(copy=x.x, constant=1)
    result = compute_linear_features(
        bars, predictors=predictors, horizons=(1,), train_window=12, model=model
    )
    assert not np.isinf(result.to_numpy()).any()
    assert np.isfinite(result["linear.prediction_h1"].iloc[12:]).all()
    flat = bars.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    result = compute_linear_features(
        flat, predictors=predictors, horizons=(1,), train_window=12, model=model
    )
    assert result["linear.prediction_h1"].iloc[12:].eq(0).all()
    assert result["linear.train_rmse_h1"].iloc[12:].eq(0).all()
    assert result["linear.prediction_scaled_h1"].isna().all()
    assert result["linear.train_r_squared_h1"].isna().all()


def test_named_predictors_match_precomputed_inputs_and_support_custom_frequency(sample):
    bars, _ = sample
    names = ("returns.log", "candles.range_relative")
    x = compute_selected_features(bars, names)
    options = dict(horizons=(1,), train_window=12, refit_every=3)
    automatic = compute_linear_features(bars, predictors=names, **options)
    manual = compute_linear_features(bars, predictors=x, **options)
    assert_frame_equal(automatic, manual)
    default = compute_linear_features(bars, train_window=25, groups=("prediction",))
    assert default["linear.prediction_h5"].notna().any()
    bars = bars.copy()
    bars.index = pd.date_range(bars.index[0], periods=len(bars), freq="7min")
    result = compute_linear_features(bars, predictors=names, source_frequency="7min", **options)
    assert result.attrs["source_frequency"] == str(pd.Timedelta("7min"))


def test_future_availability_and_misaligned_inputs_are_rejected(sample):
    bars, x = sample
    future = x.assign(available_at=x.index + pd.Timedelta("2min"))
    with pytest.raises(ValueError, match="not yet available"):
        compute_linear_features(bars, predictors=future, train_window=12)
    current = x.assign(available_at=x.index + pd.Timedelta("1min"))
    assert compute_linear_features(bars, predictors=current, train_window=12).notna().any().any()
    x.attrs["source_frequency"] = "7min"
    with pytest.raises(ValueError, match="not yet available"):
        compute_linear_features(bars, predictors=x, train_window=12)
    with pytest.raises(ValueError, match="index"):
        compute_linear_features(bars, predictors=x.iloc[1:], train_window=12)


@pytest.mark.parametrize(
    "options",
    [
        {"train_window": 0},
        {"train_window": True},
        {"refit_every": 0},
        {"min_train_size": 2},
        {"min_train_size": 20},
        {"model": "lasso"},
        {"alpha": -1},
        {"alpha": np.nan},
        {"alpha": "bad"},
        {"standardize": "yes"},
        {"horizons": ()},
        {"horizons": (1, 1)},
        {"horizons": (0,)},
        {"horizons": (True,)},
        {"horizons": 2},
        {"gap_policy": "fill"},
        {"groups": "prediction"},
        {"groups": ()},
        {"groups": ("bad",)},
        {"names": ("linear.bad_h1",)},
        {"names": ("linear.prediction_h1",), "horizons": (5,)},
        {"names": ("linear.prediction_h1",), "groups": ("prediction",)},
    ],
)
def test_invalid_configuration(sample, options):
    bars, x = sample
    with pytest.raises(ValueError):
        compute_linear_features(bars, predictors=x, **{"train_window": 12, **options})


def test_infinite_and_non_numeric_predictors_rejected(sample):
    bars, x = sample
    x.iloc[10] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        compute_linear_features(bars, predictors=x, train_window=12)
    with pytest.raises(ValueError, match="numeric"):
        compute_linear_features(bars, predictors=x.astype(str), train_window=12)


def test_nullable_predictor_warmup_and_tiny_ridge_penalty_are_supported(sample):
    bars, x = sample
    x = x.assign(copy=x.x, constant=1).astype("Float64")
    x.iloc[:2] = pd.NA
    result = compute_linear_features(
        bars,
        predictors=x,
        horizons=(1,),
        train_window=12,
        refit_every=1,
        alpha=1e-30,
        standardize=False,
    )
    assert np.isfinite(result["linear.prediction_h1"].iloc[14:]).all()
    assert not np.isinf(result.to_numpy()).any()


def test_group_selection_is_listed_in_public_catalog(sample):
    from pfe import feature_catalog

    supported = set(feature_catalog().name)
    assert {f"linear.{name}" for name in LINEAR_FEATURE_GROUPS} <= supported
    bars, x = sample
    result = compute_linear_features(bars, predictors=x, train_window=12)
    assert list(feature_catalog(result).name) == list(result.columns)
