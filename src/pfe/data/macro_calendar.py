"""Full-state macro calendar revisions, including withdrawals and cancellations."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _frame, _identifiers, _instant, _timestamps
from pfe.data.macro import _numbers


def normalize_macro_calendar_history(frame):
    """Validate immutable full-state versions; null explicitly withdraws a field.

    Required: event_id, event, scheduled_at, available_at, revision_id. Optional
    status is scheduled/released/cancelled, release_at is a verified actual release
    instant, and unit describes all numeric fields of that version. Each present
    numeric field requires its own known timestamp. A changed field cannot become
    usable before its containing version. Unchanged fields retain their old clock.
    """
    out = _frame(
        frame,
        ("event_id", "event", "scheduled_at", "available_at", "revision_id"),
        "Macro calendar history",
    )
    _identifiers(out, ("event_id", "event", "revision_id"))
    for name in ("scheduled_at", "available_at"):
        out[name] = _timestamps(out[name], name)
    for name, default in (("status", "scheduled"), ("unit", "unknown"), ("point_in_time", False)):
        if name not in out:
            out[name] = default
        else:
            out[name] = out[name].map(lambda v, fallback=default: fallback if pd.isna(v) else v)
    if not out.status.isin(("scheduled", "released", "cancelled")).all():
        raise ValueError("Invalid macro event status")
    _identifiers(out, ("unit",))
    if not out.point_in_time.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("point_in_time requires booleans")
    if "release_at" not in out:
        out["release_at"] = pd.NaT
    out["release_at"] = _timestamps(out.release_at, "release_at", nullable=True)
    if (out.release_at > out.available_at).any():
        raise ValueError("Verified release_at cannot be in the future of its version")
    if (
        out.duplicated(["event_id", "available_at"]).any()
        or out.duplicated(["event_id", "revision_id"]).any()
    ):
        raise ValueError("Duplicate or conflicting calendar revision")
    for field in ("actual", "forecast", "previous"):
        if field not in out:
            out[field] = np.nan
        out[field] = _numbers(out[field], field, nullable=True)
        clock = field + "_available_at"
        if clock not in out:
            out[clock] = pd.NaT
        out[clock] = _timestamps(out[clock], clock, nullable=True)
        if (out[field].notna() & out[clock].isna()).any():
            raise ValueError(f"Present {field} requires {clock}")
    out = out.sort_values("available_at", kind="stable", ignore_index=True)
    state = {}
    for i, row in out.iterrows():
        prior = state.get(row.event_id)
        for field in ("actual", "forecast", "previous"):
            clock = field + "_available_at"
            if pd.isna(row[field]):
                out.loc[i, clock] = pd.NaT
            elif prior is not None and row[field] == prior[field] and row.unit == prior.unit:
                out.loc[i, clock] = max(prior[clock], row[clock])
            else:
                out.loc[i, clock] = max(row.available_at, row[clock])
        state[row.event_id] = out.loc[i]
    return out


def macro_calendar_asof(frame, decision_at):
    """Latest full state per event known at the decision, including cancelled rows."""
    known = normalize_macro_calendar_history(frame)
    return (
        known.loc[known.available_at <= _instant(decision_at, "decision_at")]
        .drop_duplicates("event_id", keep="last")
        .reset_index(drop=True)
    )
