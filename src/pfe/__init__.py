"""Price Feature Engineering public API."""

from pfe.data import iter_bars, load_bars, normalize_bars
from pfe.features import (
    FEATURE_GROUPS,
    aggregate_selected_features,
    build_features,
    compute_feature,
    compute_features,
    compute_selected_features,
    enrich_bars,
    feature_backend,
    feature_catalog,
    iter_features,
    multi_timeframe_features,
    price_time_profile,
    transform_features,
)
from pfe.plots import (
    plot_bars,
    plot_feature_correlation,
    plot_feature_distribution,
    plot_features,
    plot_intrabar,
    plot_price_time_profile,
    plot_profile_bars,
)

__all__ = [
    "FEATURE_GROUPS",
    "aggregate_selected_features",
    "build_features",
    "compute_features",
    "compute_feature",
    "compute_selected_features",
    "enrich_bars",
    "feature_catalog",
    "feature_backend",
    "iter_features",
    "iter_bars",
    "load_bars",
    "multi_timeframe_features",
    "normalize_bars",
    "plot_bars",
    "plot_feature_correlation",
    "plot_feature_distribution",
    "plot_features",
    "plot_intrabar",
    "plot_price_time_profile",
    "plot_profile_bars",
    "price_time_profile",
    "transform_features",
]
