# Linear-model features

`compute_linear_features` turns chronological OLS/Ridge forecasts into features. It fits
rolling training windows, predicts future **log returns**, and exposes coefficients,
training diagnostics and delayed forecast errors. NumPy fits each eligible training
window.

## API

```python
from pfe import compute_linear_feature, compute_linear_features, feature_catalog

options = dict(
    predictors=("returns.log", "candles.range_relative", "volume.relative_w20"),
    train_window=250,
    refit_every=20,
    model="ridge",
    alpha=1.0,
)
prediction = compute_linear_feature(bars, "linear.prediction_h5", **options)
selected = compute_linear_features(
    bars, names=("linear.prediction_h5", "linear.realized_error_h5"), **options
)
grouped = compute_linear_features(
    bars, horizons=(5, 15), groups=("prediction", "coefficients"), **options
)
catalog = feature_catalog(grouped)
```

Use these dedicated APIs for model outputs; standard `compute_feature` /
`compute_features` keep their existing OHLCV/intrabar behavior. Names and groups
are mutually exclusive. `LINEAR_FEATURE_GROUPS` lists the four supported groups.
Without a selection, every linear group is returned for the configured horizons.
Names preserve caller order and infer horizons when `horizons` is omitted.

## Output groups

For each horizon `hN`, N counts source observations. With `source_frequency="7min"`,
`h5` predicts a 35-minute return on a complete grid.

### `prediction`

| Name | Meaning |
|---|---|
| `linear.prediction_hN` | Forecast of log(close at i+N / close at i) |
| `linear.prediction_scaled_hN` | Forecast divided by the model's training RMSE; NaN when RMSE is zero |

Scaled predictions express a ratio, not a calibrated probability or confidence
interval. Convert a forecast to a simple return with `np.expm1(prediction)`.

### `fit`

| Name | Meaning |
|---|---|
| `linear.train_rmse_hN` | Training residual root mean squared error |
| `linear.train_r_squared_hN` | Training R²; NaN for a constant target |
| `linear.n_train_hN` | Valid training rows at the most recent eligible refit |
| `linear.model_age_hN` | Source observations since that refit |

Training RMSE/R² are in-sample descriptors, not out-of-sample performance scores.

### `coefficients`

| Name | Meaning |
|---|---|
| `linear.intercept_hN` | Intercept in original predictor units |
| `linear.coef[<predictor>]_hN` | Coefficient in original predictor units |

Example: `linear.coef[returns.log]_h5`. Prediction uses centered coordinates to
avoid cancellation when predictors have large offsets.

### `errors`

| Name | Meaning |
|---|---|
| `linear.realized_error_hN` | Realized N-bar log return minus the original forecast, emitted at its maturity |

At row i this is the error of the forecast made at i−N, using the model available at
that time.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `predictors` | Log return, relative candle range, relative volume w20 | Exact standard feature names, or a precomputed numeric DataFrame |
| `horizons` | `(5,)` | Positive, unique source-bar counts; inferred from selected names |
| `train_window` | `250` | Maximum number of candidate, matured source rows per training window |
| `min_train_size` | `train_window` | Minimum valid rows; must be at least predictors + 2 and at most the window |
| `refit_every` | `20` | Refit interval in source observations; 1 refits each bar |
| `model` | `"ridge"` | `"ols"` or `"ridge"` |
| `alpha` | `1.0` | Non-negative Ridge L2 penalty; ignored by OLS; zero uses least squares |
| `standardize` | `True` | Training-window centering and population-std scaling; constant columns use scale 1 |
| `source_frequency` | `"1min"` | Actual input bar duration |
| `gap_policy` | `"reset"` | Reset model/history at timestamp holes; `"allow"` counts consecutive observed rows across holes |

Both methods include an unpenalized intercept. Ridge minimizes squared residuals
plus alpha × squared coefficient norm in the fitted predictor coordinates.
OLS solves the design with SVD least squares, including rank-deficient inputs;
non-identifiable coefficients use the minimum-norm solution in those coordinates.
Normalization is fitted inside each training window and reused until the next fit.

## Availability and warm-up

Indexes label bar opens; outputs become available at `index + source_frequency`,
as recorded in the result attributes. A prediction at i starts **after that close**.
At a refit on i, eligible training examples end at j = i−N: their labels end no
later than i. The newest candidate training range is
`[max(block_start, i−N−train_window+1), i−N]`, inclusive.

Each contiguous block's refit schedule starts at
`block_start + N + min_train_size − 1`, then advances by `refit_every`. Feature
warm-up NaNs can postpone the first valid fit to a later scheduled row. Before a
valid fit all outputs are NaN. If a later window has too few valid rows, the previous
model remains active and its age increases; missing current predictors produce
NaN predictions. Counts/coefficients/diagnostics describe the retained model.

With `gap_policy="reset"`, labels, normalization, forecasts and errors never cross
a source gap. A 250-bar window fits within a complete 390-minute US equity session;
larger windows may yield no fits before each reset. `allow` deliberately changes
N from elapsed intervals to observed-row horizons, which can include overnight
or weekend moves. No filling or hidden session filtering occurs.

## Precomputed predictors

```python
from pfe import compute_selected_features

x = compute_selected_features(bars, ("returns.log", "technical.rsi_w14"))
prediction = compute_linear_feature(bars, "linear.prediction_h5", predictors=x, train_window=200)
```

The predictor frame must have exactly the bars' sorted, unique, aware index,
unique non-empty string column names and numeric values. NaNs are allowed;
infinities are rejected. An optional timezone-aware `available_at` column is
checked against source closes and excluded from inputs. Frequency metadata is
also checked when explicit availability is absent. The caller must construct
custom predictors from information actually known at each decision time;
unmarked forward-looking columns cannot be detected automatically.

## Cost and evaluation

One model per requested horizon/refit is shared across selected descriptors.
Predictions between refits are vectorized; only requested outputs are allocated.
Prediction-only calls skip training RMSE/R²; count/age-only calls skip the solver.
Fit scratch space is bounded by the training window and predictor count. The
input and requested output frames still occupy memory proportional to row count.
These APIs do not yet expose batch-streaming model state.

`uv run python examples/linear_features.py` runs a synthetic example.
`uv run python scripts/benchmark_linear.py --rows 100000 --repeat 3` reproduces
timings saved in [the raw benchmark](benchmarks/linear.json). Cadence changes
predictions, so faster cadence settings are a modeling tradeoff.

Measured on this Mac: 100,000 synthetic 1-minute rows, three predictors,
500-row training window, horizon 5, Ridge alpha 1, median of three runs.
Validation and predictor construction are included; setup and I/O are excluded.

| Outputs | Refit interval | Median time |
|---|---|---|
| One prediction | 20 bars | 0.2941 s |
| All 11 outputs | 20 bars | 0.3292 s |
| One prediction | 1 bar | 5.8630 s |

For downstream stacking, use the chronological predictions produced here; tune
hyperparameters on past training/validation data and keep a final time-based test period
separate. Predictive performance requires a separate backtest.

References: [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html),
[Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[time-series validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
