"""Explicit political state histories; effective dates never imply publication."""

import pandas as pd

from pfe.data.contracts import _frame, _identifiers, _provenance, _timestamps, _vintages

POLITICAL_STATES = ("president", "house", "senate")
POLITICAL_PARTIES = ("democratic", "republican", "other", "split", "unknown")
POLITICAL_COLUMNS = (
    "record_id",
    "state_kind",
    "party",
    "effective_from",
    "available_at",
    "source_id",
    "revision_id",
)


def normalize_political_history(frame):
    """Validate user-supplied US presidency/majority vintages, without inferring parties.

    All transition/availability instants must be explicit and aware. Optional
    effective_to is exclusive, and is used only once its record is available.
    Historical reconstructions default to point_in_time=False. President IDs,
    evidence and publication/ingestion metadata may be retained as extra fields.
    """
    out = _frame(frame, POLITICAL_COLUMNS, "Political history")
    _identifiers(out, ("record_id", "source_id", "revision_id"))
    if not out.state_kind.isin(POLITICAL_STATES).all():
        raise ValueError("Political state_kind must be president, house or senate")
    if not out.party.isin(POLITICAL_PARTIES).all():
        raise ValueError("Political party requires an explicit supported category")
    if (out.state_kind.eq("president") & out.party.eq("split")).any():
        raise ValueError("Presidency cannot have a split majority")
    out["effective_from"] = _timestamps(out.effective_from, "effective_from")
    if "effective_to" not in out:
        out["effective_to"] = pd.NaT
    out["effective_to"] = _timestamps(out.effective_to, "effective_to", nullable=True)
    if (out.effective_to <= out.effective_from).any():
        raise ValueError("Political effective_to must follow effective_from")
    _provenance(out)
    _vintages(out, ["source_id", "record_id"])
    if out.groupby(["source_id", "record_id"]).state_kind.nunique().gt(1).any():
        raise ValueError("Political record identity cannot change state_kind")
    return out.sort_values("available_at", kind="stable", ignore_index=True)


def us_election_calendar(start_year, end_year):
    """Modern regular federal election dates (1946 onward), no predicted outcomes.

    Election day is the Tuesday after November's first Monday. Off-year/special
    elections and historical changes in electoral law are outside this calendar.
    """
    if (
        type(start_year) is not int
        or type(end_year) is not int
        or not 1946 <= start_year <= end_year <= 2200
    ):
        raise ValueError("US election calendar requires 1946 <= start <= end <= 2200")
    rows = []
    for year in range(start_year, end_year + 1):
        if year % 2:
            continue
        first = pd.Timestamp(year=year, month=11, day=1)
        election = first + pd.Timedelta(days=(0 - first.weekday()) % 7 + 1)
        rows.append(
            dict(
                election_date=election.date().isoformat(),
                year=year,
                election_type="presidential" if year % 4 == 0 else "midterm",
            )
        )
    return pd.DataFrame(rows, columns=["election_date", "year", "election_type"])
