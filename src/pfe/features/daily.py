"""Reuse price indicators on exchange-session ordinals, never UTC 24h candles."""

import numpy as np
import pandas as pd

from pfe.data.bars import OHLCV
from pfe.data.contracts import _instant
from pfe.data.daily import normalize_daily_bars
from pfe.data.sessions import normalize_session_schedule
from pfe.features.api import FEATURE_GROUPS, compute_features
from pfe.features.calendar import CALENDAR_FEATURE_NAMES, compute_calendar_features


def compute_daily_features(
    frame,
    *,
    schedule,
    windows=(5, 20, 60),
    groups=None,
    names=None,
    gap_policy="reset",
    decision_at=None,
    allow_partial=False,
    include_masks=True,
    structure_threshold=0.01,
    calendar_timezone="America/New_York",
):
    """Return metadata/masks and OHLCV indicators per instrument/exchange session.

    Accepts canonical daily frames from normalize_daily_bars/aggregate_session_bars.
    Rolling windows count exchange sessions; weekends, holidays, DST and early
    closes do not reset state. Missing expected sessions or ineligible bars reset
    it by default; gap_policy='allow' explicitly bridges them. Input NaNs stay
    missing and ineligible feature rows remain NaN. allow_partial only admits
    finite partial aggregated bars when the caller explicitly opts in.

    An internal ordinal UTC index is used solely for the existing price kernels.
    It is never returned or treated as a real timestamp. Intraday clock/session
    indicators are excluded; civil calendar features use actual session dates.
    All real calendar and availability columns are
    retained. No cross-instrument state or implicit adjustment/currency conversion.

    bar_available_at retains input availability; available_at for derived features
    is conservatively the maximum availability of the used chronological prefix.
    An old bar received late therefore cannot contaminate an earlier decision.
    decision_at optionally removes not-yet-known bars before calculation. Resolve
    daily source revisions upstream; this wrapper accepts one row per session,
    not a replay of all historical feature vintages. Civil calendar features
    use real session_date labels; the ordinal index is only for price kernels.

    mask.<feature> records known/NaN outputs, including warm-up. Ages in seconds
    are evaluated at decision_at when supplied, otherwise at feature availability:
    quality.age_seconds measures time since session close; receipt age is null
    if ingestion is unknown or belongs to a later backfill. Availability lag
    preserves the source bar's delay separately from derived feature delay.
    """
    if gap_policy not in ("reset", "allow"):
        raise ValueError("gap_policy must be reset or allow")
    if not isinstance(allow_partial, (bool, np.bool_)):
        raise ValueError("allow_partial requires a boolean")
    if not isinstance(include_masks, (bool, np.bool_)):
        raise ValueError("include_masks requires a boolean")
    if names is not None and groups is not None:
        raise ValueError("Choose groups or exact names, not both")
    if names is not None:
        names = (names,) if isinstance(names, str) else tuple(names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Select distinct daily feature names")
        if any(isinstance(name, str) and name.startswith("session.") for name in names):
            raise ValueError("Daily features do not expose intraday calendar/session indicators")
        civil_names = tuple(name for name in names if name.startswith("calendar."))
        if set(civil_names) - set(CALENDAR_FEATURE_NAMES):
            raise ValueError("Daily features do not expose intraday calendar/session indicators")
        price_names = tuple(name for name in names if name not in civil_names)
        price_groups = () if not price_names else None
    else:
        if isinstance(groups, str):
            raise ValueError("groups must be an iterable of names")
        groups = FEATURE_GROUPS if groups is None else tuple(groups)
        if "session" in groups:
            raise ValueError("Daily features do not expose intraday calendar/session indicators")
        civil_names = CALENDAR_FEATURE_NAMES if "calendar" in groups else ()
        price_names = None
        price_groups = tuple(group for group in groups if group != "calendar")
    calendar = normalize_session_schedule(schedule)
    out = normalize_daily_bars(frame, schedule=calendar, availability_basis="unknown")
    if decision_at is not None:
        out = out.loc[out.available_at <= _instant(decision_at, "decision_at")].reset_index(
            drop=True
        )
    if out.empty:
        return out
    out["bar_available_at"] = out.available_at
    outputs = []
    ordinal = pd.Series(np.arange(len(calendar)), index=calendar.index)
    epoch = pd.Timestamp("2000-01-01T00:00:00Z")
    for _, instrument in out.groupby("instrument_id", sort=False):
        instrument = instrument.copy()
        for column in ("currency", "price_adjustment", "volume_adjustment", "session_scope"):
            if instrument[column].nunique(dropna=False) > 1:
                raise ValueError(f"{column} cannot change within a daily feature series")
        eligible = instrument["quality.feature_eligible"].to_numpy(bool)
        if allow_partial:
            eligible = instrument.loc[:, OHLCV].notna().all(axis=1).to_numpy(bool)
        known = instrument.loc[eligible]
        if known.empty:
            # A valid dummy bar discovers the exact requested feature schema only.
            bars = pd.DataFrame(
                [[1.0, 1.0, 1.0, 1.0, 0.0]], columns=OHLCV, index=pd.DatetimeIndex([epoch])
            )
            schema = compute_features(
                bars,
                windows=windows,
                groups=price_groups,
                names=price_names or None,
                source_frequency="1d",
                structure_threshold=structure_threshold,
            )
            features = pd.DataFrame(np.nan, index=instrument.index, columns=schema.columns)
        else:
            bars = known.loc[:, OHLCV].copy()
            positions = ordinal.loc[known.session_date].to_numpy()
            bars.index = pd.DatetimeIndex(epoch + pd.to_timedelta(positions, unit="D"))
            features = compute_features(
                bars,
                windows=windows,
                groups=price_groups,
                names=price_names or None,
                source_frequency="1d",
                gap_policy=gap_policy,
                structure_threshold=structure_threshold,
            )
            features.index = known.index
            features = features.reindex(instrument.index)
        if civil_names:
            civil = compute_calendar_features(
                instrument.session_open,
                timezone=calendar_timezone,
                session_dates=instrument.session_date.tolist(),
                schedule=calendar,
            ).loc[:, civil_names]
            civil.index = instrument.index
            features = pd.concat([features, civil], axis=1)
        instrument["available_at"] = instrument.bar_available_at.cummax()
        evaluation = (
            pd.Series(_instant(decision_at, "decision_at"), index=instrument.index)
            if decision_at is not None
            else instrument.available_at
        )
        instrument["quality.age_seconds"] = (
            evaluation - instrument.session_close
        ).dt.total_seconds()
        instrument["quality.availability_lag_seconds"] = (
            instrument.bar_available_at - instrument.session_close
        ).dt.total_seconds()
        receipt_age = (evaluation - instrument.ingested_at).dt.total_seconds()
        instrument["quality.receipt_age_seconds"] = receipt_age.where(receipt_age >= 0)
        collisions = set(features.columns) & set(instrument.columns)
        if collisions:
            raise ValueError("Daily inputs already contain requested feature columns")
        combined = pd.concat([instrument, features], axis=1)
        if include_masks:
            masks = features.notna().rename(columns=lambda name: f"mask.{name}")
            if set(masks.columns) & set(combined.columns):
                raise ValueError("Daily inputs already contain requested feature masks")
            combined = pd.concat([combined, masks], axis=1)
        outputs.append(combined)
    result = pd.concat(outputs, ignore_index=True)
    result.attrs.update(
        timestamp_semantics="session_open",
        available_at="maximum known availability of chronological instrument prefix",
        window_units="exchange_sessions",
        gap_policy=gap_policy,
        excluded_groups=("intraday_session",),
        calendar=calendar.attrs.get("calendar", "injected"),
    )
    return result
