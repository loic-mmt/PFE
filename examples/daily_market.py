"""Synthetic stock/ETF features, a split and a cash dividend; no key or network."""

import pandas as pd

from pfe import (
    ACTION_COLUMNS,
    compute_daily_features,
    corporate_action_returns,
    exchange_schedule,
    normalize_corporate_actions,
    normalize_daily_bars,
)


def main():
    schedule = exchange_schedule("2026-03-06", "2026-03-10", calendar="XNYS")
    records = []
    for instrument, entity, prices in (
        ("synthetic:stock", "synthetic:company", (100.0, 50.0, 51.0)),
        ("synthetic:etf", "synthetic:fund", (400.0, 410.0, 414.0)),
    ):
        for session_date, price in zip(schedule.index, prices, strict=True):
            records.append(
                dict(
                    instrument_id=instrument,
                    entity_id=entity,
                    session_date=session_date,
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price,
                    volume=1000.0,
                )
            )
    daily = normalize_daily_bars(
        pd.DataFrame(records),
        schedule=schedule,
        source_id="local",
        currency="USD",
        price_adjustment="raw",
        volume_adjustment="raw",
        session_scope="rth",
        availability_basis="event_time",  # Synthetic timestamps; not provider deliveries.
    )
    facts = []
    for action_id, action_type, day, ratio, amount in (
        ("split-1", "split", "2026-03-09", 2.0, None),
        ("dividend-1", "dividend", "2026-03-10", None, 1.0),
    ):
        facts.append(
            dict(
                action_id=action_id,
                action_type=action_type,
                entity_id="synthetic:company",
                instrument_id="synthetic:stock",
                source_id="local",
                revision_id="fixture-1",
                schema_version="1",
                announcement_at="2026-03-06T19:00:00Z",
                published_at="2026-03-06T19:00:00Z",
                available_at="2026-03-06T19:00:00Z",
                effective_date=day,
                ex_date=day if action_type == "dividend" else None,
                ratio=ratio,
                amount=amount,
                currency="USD" if action_type == "dividend" else None,
                amount_unit="currency_per_share",
                dividend_basis="post_split_share",
            )
        )
    actions = normalize_corporate_actions(
        pd.DataFrame(
            facts,
            columns=[
                *ACTION_COLUMNS,
                "amount_unit",
                "dividend_basis",
            ],
        )
    )
    features = compute_daily_features(
        daily,
        schedule=schedule,
        names=("returns.simple", "trend.sma_w2"),
    )
    returns = corporate_action_returns(
        daily,
        actions,
        price_adjustment="raw",
        volume_adjustment="raw",
        include_dividends=True,
        action_source_id="local",
        coverage_complete=True,
    )
    stock = returns.loc[returns.instrument_id.eq("synthetic:stock")]
    assert stock.price_return.iloc[1] == 0.0
    assert abs(stock.total_return.iloc[2] - 0.04) < 1e-12
    print(
        features[
            [
                "instrument_id",
                "session_date",
                "returns.simple",
                "mask.returns.simple",
                "available_at",
            ]
        ].to_string(index=False)
    )
    print(
        stock[
            [
                "session_date",
                "input_price_return",
                "price_return",
                "total_return",
            ]
        ].to_string(index=False)
    )
    early = exchange_schedule("2026-11-27", "2026-11-27", calendar="XNYS")
    print(f"NYSE early close: {early.close_at.iloc[0]}; synthetic fixtures only")


if __name__ == "__main__":
    main()
