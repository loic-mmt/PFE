"""Dated, versioned sector/group memberships supplied by explicit sources."""

import pandas as pd

from pfe.data.contracts import _frame, _identifiers, _instant, _provenance, _timestamps, _vintages
from pfe.data.security_master import _intervals


def normalize_classifications(frame):
    out = _frame(
        frame,
        (
            "mapping_id",
            "entity_id",
            "group_id",
            "source_id",
            "revision_id",
            "valid_from",
            "valid_to",
            "available_at",
        ),
        "Classifications",
    )
    _identifiers(out, ("mapping_id", "entity_id", "group_id", "source_id", "revision_id"))
    out["valid_from"] = _timestamps(out.valid_from, "valid_from")
    out["valid_to"] = _timestamps(out.valid_to, "valid_to", nullable=True)
    _provenance(out)
    if (out.valid_to <= out.valid_from).any():
        raise ValueError("Classification valid_to must follow valid_from")
    _vintages(out, ["source_id", "mapping_id"])
    out = out.sort_values("available_at", kind="stable", ignore_index=True)
    state = {}
    for _, batch in out.groupby("available_at", sort=True):
        for row in batch.to_dict("records"):
            state[row["source_id"], row["mapping_id"]] = row
        _intervals(pd.DataFrame(state.values()), ["source_id", "entity_id"])
    return out


def classifications_asof(frame, decision_at, *, source_id=None):
    decision = _instant(decision_at, "decision_at")
    out = normalize_classifications(frame)
    out = out.loc[out.available_at <= decision].drop_duplicates(
        ["source_id", "mapping_id"], keep="last"
    )
    out = out.loc[(out.valid_from <= decision) & (out.valid_to.isna() | (decision < out.valid_to))]
    if source_id is not None:
        out = out.loc[out.source_id.eq(source_id)]
    if out.groupby("entity_id").group_id.nunique().gt(1).any():
        raise ValueError("Select a classification source before resolving conflicting groups")
    return (
        out.sort_values("available_at")
        .drop_duplicates("entity_id", keep="last")
        .reset_index(drop=True)
    )
