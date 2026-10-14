"""Offline Lot 2: reference horizons, FRED vintages and SEC actuals."""

import pandas as pd
from pandas.testing import assert_frame_equal

from pfe import (
    compute_fred_features,
    compute_fundamental_features,
    compute_reference_features,
    normalize_company_facts,
    normalize_fred_observations,
    normalize_sec_filings,
)


def reference(instrument, day, price):
    return dict(
        entity_id=f"factor:{instrument}",
        instrument_id=instrument,
        symbol=instrument,
        asset_class="benchmark",
        currency="USD",
        calendar="XNYS",
        frequency="daily",
        price_adjustment="split_adjusted",
        unit="USD/share",
        metric="close",
        source_id="local",
        revision_id="synthetic-1",
        schema_version="1",
        observation_date=f"2026-03-{day:02}",
        observed_at=f"2026-03-{day:02}T21:00:00Z",
        published_at=None,
        available_at=f"2026-03-{day:02}T21:01:00Z",
        value=price,
    )


def main():
    decisions = pd.DatetimeIndex(
        ["2026-03-02T21:01:00Z", "2026-03-03T21:01:00Z", "2026-03-04T21:01:00Z"]
    )
    refs = pd.DataFrame(
        [reference("SPY", 2, 100), reference("SPY", 3, 110), reference("SPY", 4, 132)]
    )
    stock = pd.DataFrame(
        [reference("STOCK", 2, 100), reference("STOCK", 3, 120), reference("STOCK", 4, 168)]
    )
    market = compute_reference_features(
        decisions,
        references=refs,
        target=stock,
        windows=(2,),
        names=("reference[SPY].return", "reference[SPY].beta_w2"),
    )
    assert abs(market["reference[SPY].beta_w2"].iloc[-1] - 2) < 1e-10

    metadata = dict(
        id="SYN",
        units="Index",
        title="Synthetic macro index",
        frequency="Monthly",
        seasonal_adjustment="Not Seasonally Adjusted",
    )
    old = dict(
        date="2026-01-01", realtime_start="2026-03-02", realtime_end="2026-03-09", value="100"
    )
    revision = old | dict(realtime_start="2026-03-10", realtime_end="9999-12-31", value="105")
    observations = normalize_fred_observations(
        [old, revision],
        metadata,
        received_at="2026-10-12T12:00:00Z",
        availability="vintage_next_day",
    )
    macro = compute_fred_features(decisions, observations=observations, names=("macro.SYN.level",))
    initial = compute_fred_features(
        decisions, observations=observations.iloc[:1], names=("macro.SYN.level",)
    )
    assert_frame_equal(initial, macro)
    assert (
        macro["macro.SYN.level"].iloc[0] != macro["macro.SYN.level"].iloc[0]
    )  # Before vintage availability.
    assert macro["macro.SYN.level"].iloc[-1] == 100

    accession = "0000001234-26-000001"
    submissions = dict(
        cik=1234,
        filings=dict(
            recent=dict(
                accessionNumber=[accession],
                filingDate=["2026-03-03"],
                reportDate=["2025-12-31"],
                acceptanceDateTime=["2026-03-03T20:00:00Z"],
                form=["10-K"],
                primaryDocument=["synthetic.htm"],
            )
        ),
    )
    filings = normalize_sec_filings(submissions, received_at="2026-03-03T20:01:00Z")
    facts_payload = dict(cik=1234, facts={"us-gaap": {}})
    for concept, value in (
        ("Revenues", 100),
        ("GrossProfit", 40),
        ("NetCashProvidedByUsedInOperatingActivities", 30),
        ("PaymentsToAcquirePropertyPlantAndEquipment", 10),
    ):
        facts_payload["facts"]["us-gaap"][concept] = dict(
            units={
                "USD": [
                    dict(
                        start="2025-10-01",
                        end="2025-12-31",
                        val=value,
                        accn=accession,
                        filed="2026-03-03",
                        fy=2025,
                        fp="FY",
                        form="10-K",
                    )
                ]
            }
        )
    facts = normalize_company_facts(
        facts_payload, received_at="2026-03-03T20:02:00Z", filings=filings
    )
    fundamentals = compute_fundamental_features(
        decisions,
        facts=facts,
        currency="USD",
        names=("fundamentals.gross_margin", "fundamentals.free_cash_flow"),
    )
    assert fundamentals["fundamentals.gross_margin"].iloc[-1] == 0.4
    assert fundamentals["fundamentals.free_cash_flow"].iloc[-1] == 20
    print(market.drop(columns=[name for name in market if name.startswith("mask.")]))
    print(macro)
    print(fundamentals)
    print(
        "Lot 2 offline: matched beta=2; future vintage leaves past unchanged; margin=40%, FCF=20."
    )


if __name__ == "__main__":
    main()
