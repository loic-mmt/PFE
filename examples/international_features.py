"""Small offline TSMC example: no network, model or persistent dataset."""

from tempfile import TemporaryDirectory

import pfe

body = b"""<h2>2026 Monthly Revenue</h2><p>Millions of New Taiwan Dollars</p>
<table><tr><td>Jan.</td><td>100</td><td>20%</td></tr>
<tr><td>Feb.</td><td>110</td><td>25%</td></tr></table>"""
with TemporaryDirectory(prefix="pfe-international-") as directory:
    store = pfe.RawStore(directory)
    capture = pfe.collect_international(
        "tsmc",
        {"year": 2026, "entity_id": "company:tsmc"},
        store,
        snapshot_id="synthetic",
        offline=(body, "2026-03-10T08:00:00Z"),
    )
    features = pfe.compute_international_features(
        ["2026-03-10T07:59:59Z", "2026-03-10T08:00:00Z"],
        capture.observations,
        series=[
            pfe.InternationalSeries(
                "tsmc_revenue",
                "company:tsmc",
                "revenue",
                "million_TWD",
                "tsmc",
                "monthly",
                max_age="45D",
                max_observation_age="90D",
            )
        ],
    )
    assert features.iloc[0]["international.tsmc_revenue.mask.available"] == 0
    assert features.iloc[1]["international.tsmc_revenue.level"] == 110
    assert round(features.iloc[1]["international.tsmc_revenue.pct_change"], 8) == 0.1
    print(features[["international.tsmc_revenue.level", "international.tsmc_revenue.mask.value"]])
