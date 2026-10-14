"""Election phase features and separately available presidency/Congress states."""

import numpy as np
import pandas as pd

from pfe.data.politics import POLITICAL_STATES, normalize_political_history, us_election_calendar
from pfe.features.calendar import calendar_index


def compute_political_features(decisions, *, history=None, source_id=None):
    """US regular election cycles use New York civil dates, on any decision grid.

    cycle_year counts calendar years 1..4 (2001=1), not years of a person's
    presidency. post_midterm_9m runs from the day AFTER election through nine
    calendar months, end exclusive; it encodes timing, never a bullish label.
    Party/majority outputs require explicit history; no result is anticipated.
    """
    index = calendar_index(decisions)
    local = index.tz_convert("America/New_York").tz_localize(None).normalize()
    if len(local) and (local.year.min() < 1950 or local.year.max() > 2195):
        raise ValueError("Political feature decisions require years 1950..2195")
    history = normalize_political_history(history) if history is not None else None
    if source_id is not None:
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Political source_id requires a nonempty string")
        if history is not None:
            history = history.loc[history.source_id.eq(source_id)]
    rows = []
    for decision, day in zip(index, local, strict=True):
        cycle_year = (day.year - 1) % 4 + 1
        row = {
            "politics.us.cycle_year": float(cycle_year),
            "politics.us.cycle_sin": np.sin(2 * np.pi * (cycle_year - 1) / 4),
            "politics.us.cycle_cos": np.cos(2 * np.pi * (cycle_year - 1) / 4),
            "politics.us.is_presidential_year": float(day.year % 4 == 0),
            "politics.us.is_midterm_year": float(day.year % 4 == 2),
        }
        calendar = us_election_calendar(day.year - 4, day.year + 4)
        for kind in ("presidential", "midterm"):
            dates = pd.DatetimeIndex(calendar.loc[calendar.election_type.eq(kind), "election_date"])
            previous = dates[dates <= day].max()
            following = dates[dates > day].min()
            row[f"politics.us.days_since_{kind}"] = float((day - previous).days)
            row[f"politics.us.days_to_next_{kind}"] = float((following - day).days)
            row[f"politics.us.is_{kind}_election_day"] = float(day in dates)
            if kind == "midterm":
                start = previous + pd.Timedelta(days=1)
                end = start + pd.DateOffset(months=9)
                row["politics.us.post_midterm_9m"] = float(start <= day < end)
                row["politics.us.months_since_midterm"] = float(
                    (day.year - previous.year) * 12
                    + day.month
                    - previous.month
                    - (day.day < previous.day)
                )
        known = None
        if history is not None:
            known = history.loc[history.available_at <= decision].drop_duplicates(
                ["source_id", "record_id"], keep="last"
            )
            known = known.loc[
                (known.effective_from <= decision)
                & (known.effective_to.isna() | (known.effective_to > decision))
            ]
        parties = {}
        for kind in POLITICAL_STATES:
            selected = known.loc[known.state_kind.eq(kind)] if known is not None else None
            if selected is not None and len(selected):
                selected = selected.loc[selected.effective_from.eq(selected.effective_from.max())]
            ambiguous = selected is not None and len(selected) > 1
            point = selected.iloc[0] if selected is not None and len(selected) == 1 else None
            party = point.party if point is not None and point.party != "unknown" else None
            parties[kind] = party
            prefix = f"politics.us.{kind}"
            for category in ("democratic", "republican", "other", "split"):
                row[prefix + "." + category] = float(party == category) if party else np.nan
            row[prefix + ".mask"] = float(party is not None)
            row[prefix + ".quality.ambiguous"] = float(ambiguous)
            row[prefix + ".quality.point_in_time"] = (
                float(point.point_in_time) if point is not None else np.nan
            )
            row[prefix + ".age_seconds"] = (
                (decision - point.available_at).total_seconds() if point is not None else np.nan
            )
            row[prefix + ".tenure_days"] = (
                (decision - point.effective_from).total_seconds() / 86400
                if point is not None
                else np.nan
            )
        comparable = all(p in ("democratic", "republican") for p in parties.values())
        unified = len(set(parties.values())) == 1 if comparable else np.nan
        row["politics.us.unified_government"] = float(unified)
        row["politics.us.divided_government"] = float(not unified) if comparable else np.nan
        row["politics.us.mask.government"] = float(comparable)
        rows.append(row)
    result = pd.DataFrame(rows, index=index)
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        timezone="America/New_York",
        cycle_basis="regular federal election calendar years, not personal term tenure",
        post_midterm_window="[election date + 1 day, start + 9 calendar months)",
        state_availability="effective AND available, caller-supplied history",
    )
    return result
