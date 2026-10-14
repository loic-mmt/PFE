"""Calendar/political/conflict inputs -> reusable feature panel; fully synthetic/offline."""

import json

import pandas as pd

from pfe import (
    UCDPQuery,
    compute_calendar_features,
    compute_conflict_features,
    compute_political_features,
    context_feature_observations,
    parse_ucdp,
)
from pfe.datasets import build_feature_panel


def main():
    decisions = pd.DatetimeIndex(["2026-01-03T12:00Z", "2026-01-04T12:00Z"])
    body = json.dumps(
        {
            "Result": [
                {
                    "id": 1,
                    "conflict_new_id": 10,
                    "country_id": 678,
                    "region": "Middle East",
                    "type_of_violence": 1,
                    "date_start": "2026-01-02",
                    "date_end": "2026-01-02",
                    "best": 6,
                    "low": 4,
                    "high": 8,
                }
            ]
        }
    ).encode()
    events = parse_ucdp(
        body,
        query=UCDPQuery("gedevents", "26.1", "2026-01-01", "2026-01-31"),
        received_at="2026-01-03T12:00Z",
    )
    features = pd.concat(
        [
            compute_calendar_features(decisions),
            compute_political_features(decisions),
            compute_conflict_features(decisions, events),
        ],
        axis=1,
    )
    units = {
        "calendar.month": "ordinal",
        "politics.us.cycle_year": "ordinal",
        "conflicts.global.state_based.active_conflicts_30D": "count",
    }
    observations = context_feature_observations(
        features,
        units=units,
        entity_ids=["company:example"],
        source_id="synthetic",
        revision_id="config-1",
    )
    registry = [
        dict(name=name, metric=name, unit=unit, source_id="synthetic")
        for name, unit in units.items()
    ]
    panel = build_feature_panel(
        decisions,
        observations,
        entity_ids=["company:example"],
        registry=registry,
    )
    assert panel.mask.all() and panel.values.shape == (2, 1, 3)
    print(features.loc[:, list(units)])
    phase = compute_political_features(
        pd.DatetimeIndex(
            [
                "2022-11-08T15:00Z",
                "2022-11-09T15:00Z",
                "2023-08-09T15:00Z",
            ]
        )
    )
    assert phase["politics.us.post_midterm_9m"].tolist() == [0, 1, 0]
    print("2 decisions, 3 model inputs, no downloads; midterm boundaries checked")


if __name__ == "__main__":
    main()
