"""Synthetic split/dividend revisions and availability-safe local return bridges."""

import json

import numpy as np
import pandas as pd
import pytest

from pfe.data import storage
from pfe.data.corporate_actions import (
    ACTION_COLUMNS,
    adapt_lse_actions,
    collect_lse_actions,
    corporate_action_returns,
    corporate_actions_asof,
    normalize_corporate_actions,
)
from pfe.data.qualification import DiscoveryLimitError, DiscoveryResponse
from pfe.data.storage import RawStore


@pytest.fixture
def field_maps():
    base = {"symbol": "ticker", "action_id": "id", "effective_date": "date"}
    return {
        "split": {**base, "ratio": "factor"},
        "dividend": {**base, "amount": "cash", "currency": "ccy"},
    }


def _action(**changes):
    return (
        dict(
            action_id="split-1",
            entity_id="company",
            instrument_id="share",
            source_id="synthetic",
            revision_id="r1",
            schema_version="1",
            action_type="split",
            announcement_at=None,
            published_at=None,
            available_at="2026-01-05T20:00:00Z",
            ex_date=None,
            effective_date="2026-01-06",
            ratio=2.0,
            amount=None,
            currency=None,
        )
        | changes
    )


def _actions(*rows):
    return pd.DataFrame(rows, columns=[*ACTION_COLUMNS, "dividend_basis", "amount_unit", "status"])


def _history(*rows):
    frame = _actions(*rows)
    # Missing optional declarations should use their explicit unknown defaults.
    for column in ("dividend_basis", "amount_unit", "status"):
        if frame[column].isna().all():
            frame = frame.drop(columns=column)
        else:
            frame[column] = frame[column].fillna(
                {"dividend_basis": "unknown", "amount_unit": "unknown", "status": "confirmed"}[
                    column
                ]
            )
    return frame


def _prices(closes=(100.0, 50.0, 51.0), *, adjustment="raw"):
    return pd.DataFrame(
        dict(
            entity_id=["company"] * len(closes),
            instrument_id=["share"] * len(closes),
            session_date=[f"2026-01-{i:02}" for i in range(5, 5 + len(closes))],
            available_at=[f"2026-01-{i:02}T21:00:00Z" for i in range(5, 5 + len(closes))],
            session_close=[f"2026-01-{i:02}T21:00:00Z" for i in range(5, 5 + len(closes))],
            close=closes,
            volume=[1000] * len(closes),
            currency=["USD"] * len(closes),
            price_adjustment=[adjustment] * len(closes),
            volume_adjustment=["raw"] * len(closes),
        )
    )


def _returns(prices, actions, **kwargs):
    return corporate_action_returns(
        prices, actions, price_adjustment="raw", volume_adjustment="raw", **kwargs
    )


def test_revisions_date_corrections_and_cancellation_resolved_before_filter():
    original = _action()
    corrected = _action(
        revision_id="r2",
        available_at="2026-01-07T20:00:00Z",
        effective_date="2026-01-08",
        ratio=3.0,
    )
    cancelled = corrected | {
        "revision_id": "r3",
        "available_at": "2026-01-09T20:00:00Z",
        "status": "cancelled",
    }
    frame = _history(original, corrected, cancelled)
    out = normalize_corporate_actions(frame)
    assert out.loc[0, "published_at"] is pd.NaT
    assert out.loc[0, "point_in_time"] == False  # noqa: E712
    early = corporate_actions_asof(frame, "2026-01-06T21:00:00Z", "2026-01-06")
    assert early.ratio.tolist() == [2.0]
    assert corporate_actions_asof(frame, "2026-01-07T21:00:00Z", "2026-01-06").empty
    assert corporate_actions_asof(frame, "2026-01-08T21:00:00Z").ratio.tolist() == [3.0]
    assert corporate_actions_asof(frame, "2026-01-10T21:00:00Z").empty
    assert frame.loc[0, "available_at"] == "2026-01-05T20:00:00Z"


@pytest.mark.parametrize(
    "change",
    [
        {"ratio": 0},
        {"ratio": "2"},
        {"ratio": True},
        {"amount": 1},
        {"effective_date": "2026-02-30"},
        {"available_at": "2026-01-05"},
        {"available_at": "2026-01-05T19:00:00Z", "announcement_at": "2026-01-05T20:00:00Z"},
    ],
)
def test_invalid_action_contract_rejected(change):
    with pytest.raises(ValueError):
        normalize_corporate_actions(_history(_action(**change)))


def test_reused_revision_and_equal_time_vintages_rejected():
    original = _action()
    with pytest.raises(ValueError, match="revision_id"):
        normalize_corporate_actions(
            _history(original, original | {"available_at": "2026-01-07T20:00:00Z"})
        )
    with pytest.raises(ValueError, match="simultaneous"):
        normalize_corporate_actions(_history(original, original | {"revision_id": "r2"}))


def test_receipt_requires_ingestion_and_never_backdates_publication():
    frame = _history(_action())
    frame["availability_basis"] = "receipt"
    with pytest.raises(ValueError, match="ingested_at"):
        normalize_corporate_actions(frame)
    frame["ingested_at"] = "2026-01-06T20:00:00Z"
    with pytest.raises(ValueError, match="ingested_at"):
        normalize_corporate_actions(frame)


def test_forward_and_reverse_splits_bridge_only_current_interval():
    prices = _prices()
    future = _action(
        action_id="future", revision_id="future-r1", effective_date="2026-01-08", ratio=10.0
    )
    out = _returns(prices, _history(_action(), future))
    assert np.isnan(out.loc[0, "price_return"])
    assert out.loc[1, "price_return"] == pytest.approx(0)
    assert out.loc[1, "input_price_return"] == pytest.approx(-0.5)
    assert out.loc[2, "price_return"] == pytest.approx(0.02)
    assert out.loc[1, "split_action_ids"] == ["split-1"]
    assert out.close.tolist() == prices.close.tolist()
    assert out.volume.tolist() == prices.volume.tolist()
    reverse = _returns(_prices((10, 100)), _history(_action(ratio=0.1)))
    assert reverse.loc[1, "price_return"] == pytest.approx(0)


def test_late_split_and_revision_never_rewrite_old_return_decisions():
    late = _action(available_at="2026-01-07T20:00:00Z")
    out = _returns(_prices(), _history(late))
    assert out.loc[1, "price_return"] == pytest.approx(-0.5)
    assert out.loc[1, "split_action_ids"] == []
    assert out.loc[2, "price_return"] == pytest.approx(0.02)
    revision = _action(revision_id="r2", available_at="2026-01-07T20:00:00Z", ratio=3)
    out = _returns(_prices(), _history(_action(), revision))
    assert out.loc[1, "split_factor"] == 2
    assert out.loc[2, "split_factor"] == 1


def _dividend(**changes):
    return (
        _action(
            action_id="div-1",
            action_type="dividend",
            ratio=None,
            amount=1.0,
            currency="USD",
            ex_date="2026-01-06",
            dividend_basis="post_split_share",
            amount_unit="currency_per_share",
        )
        | changes
    )


def test_dividends_separate_total_from_price_return_and_reject_implicit_fx():
    history = _history(_dividend())
    out = _returns(_prices((100, 99)), history, include_dividends=True)
    assert out.loc[1, "price_return"] == pytest.approx(-0.01)
    assert out.loc[1, "total_return"] == pytest.approx(0)
    assert out.loc[1, "cash_dividend"] == 1
    assert _returns(_prices((100, 99)), history).total_return.isna().all()
    with pytest.raises(ValueError, match="currency"):
        _returns(_prices((100, 99)), _history(_dividend(currency="EUR")), include_dividends=True)
    with pytest.raises(ValueError, match="convention"):
        _returns(
            _prices((100, 99)),
            _history(_dividend(dividend_basis="unknown")),
            include_dividends=True,
        )


@pytest.mark.parametrize("basis,cash", [("post_split_share", 1.0), ("pre_split_share", 0.5)])
def test_same_day_split_cash_uses_explicit_share_convention(basis, cash):
    dividends = _dividend(amount=0.5, dividend_basis=basis)
    out = _returns(_prices((100, 49.5)), _history(_action(), dividends), include_dividends=True)
    assert out.loc[1, "price_return"] == pytest.approx(-0.01)
    assert out.loc[1, "cash_dividend"] == cash
    assert out.loc[1, "total_return"] == pytest.approx((99 + cash) / 100 - 1)


def test_dividend_is_not_multiplied_by_later_splits_within_missing_sessions():
    prices = _prices((100, 16))
    prices.loc[1, "session_date"] = "2026-01-08"
    prices.loc[1, "available_at"] = "2026-01-08T21:00:00Z"
    prices.loc[1, "session_close"] = "2026-01-08T21:00:00Z"
    second = _action(action_id="split-2", revision_id="r2", effective_date="2026-01-08", ratio=3)
    div = _dividend(ex_date="2026-01-07")
    out = _returns(prices, _history(_action(), second, div), include_dividends=True)
    assert out.loc[1, "split_factor"] == 6
    assert out.loc[1, "cash_dividend"] == 2
    assert out.loc[1, "total_return"] == pytest.approx(-0.02)


@pytest.mark.parametrize(
    "price,volume", [("unknown", "raw"), ("total_return", "raw"), ("raw", "unknown")]
)
def test_unknown_or_double_adjustment_rejected(price, volume):
    with pytest.raises(ValueError):
        corporate_action_returns(
            _prices(), _history(_action()), price_adjustment=price, volume_adjustment=volume
        )


def test_already_split_adjusted_prices_are_not_adjusted_again_or_mixed_with_raw_cash():
    prices = _prices((50, 50), adjustment="split_adjusted")
    out = corporate_action_returns(
        prices, _history(_action()), price_adjustment="split_adjusted", volume_adjustment="raw"
    )
    assert out.loc[1, "price_return"] == 0
    assert out.loc[1, "split_factor"] == 1
    with pytest.raises(ValueError, match="per-share scale"):
        corporate_action_returns(
            prices,
            _history(_action()),
            price_adjustment="split_adjusted",
            volume_adjustment="raw",
            include_dividends=True,
        )


def test_cross_source_duplicates_require_explicit_reconciliation():
    other = _action(source_id="other")
    with pytest.raises(ValueError, match="one action source"):
        _returns(_prices(), _history(_action(), other))
    out = _returns(_prices(), _history(_action(), other), action_source_id="synthetic")
    assert out.loc[1, "split_factor"] == 2
    assert out.action_coverage_complete.eq(False).all()


def test_session_future_to_decision_and_unavailable_previous_price_rejected():
    prices = _prices()
    prices.loc[1, "available_at"] = "2026-01-05T21:00:00Z"
    with pytest.raises(ValueError, match="session_close"):
        _returns(prices, _history(_action()))
    prices = _prices()
    prices.loc[0, "available_at"] = "2026-01-07T21:00:00Z"
    with pytest.raises(ValueError, match="Previous price"):
        _returns(prices, _history(_action()))


def test_explicit_provider_mapping_ratio_direction_and_date_only_announcements(field_maps):
    fields = field_maps["split"] | {"announcement_at": "announcement"}
    out = adapt_lse_actions(
        [
            {
                "id": 7,
                "date": "2026-01-06",
                "factor": 0.5,
                "ticker": "SYN",
                "announcement": "2026-01-02",
            }
        ],
        action_type="split",
        entity_id="company",
        instrument_id="share",
        received_at="2026-01-07T20:00:00Z",
        field_map=fields,
        ratio_convention="old_per_new",
        expected_symbol="SYN",
    )
    assert out.loc[0, "ratio"] == 2
    assert out.loc[0, "action_id"] == "7"
    assert pd.isna(out.loc[0, "announcement_at"])
    assert pd.isna(out.loc[0, "published_at"])
    assert out.loc[0, "availability_basis"] == "receipt"
    assert out.loc[0, "point_in_time"] == False  # noqa: E712
    assert out.loc[0, "source_payload"]["announcement"] == "2026-01-02"
    with pytest.raises(ValueError, match="ratio_convention"):
        adapt_lse_actions(
            [],
            action_type="split",
            entity_id="company",
            instrument_id="share",
            received_at="2026-01-07T20:00:00Z",
            field_map=fields,
        )
    with pytest.raises(ValueError, match="symbol"):
        adapt_lse_actions(
            [{"id": 7, "date": "2026-01-06", "factor": 2, "ticker": "OTHER"}],
            action_type="split",
            entity_id="company",
            instrument_id="share",
            received_at="2026-01-07T20:00:00Z",
            field_map=fields,
            ratio_convention="new_per_old",
            expected_symbol="SYN",
        )


class _Transport:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, endpoint, params, *, max_bytes):
        self.calls.append((endpoint, params))
        rows = self.rows[endpoint]
        size = len(json.dumps(rows).encode())
        if size > max_bytes:
            raise DiscoveryLimitError("Synthetic byte cap")
        return DiscoveryResponse(rows, size)


def _collect(transport, store, field_maps, **kwargs):
    return collect_lse_actions(
        transport,
        store,
        symbol="SYN",
        entity_id="company",
        instrument_id="share",
        start="2026-01-01",
        end="2026-01-31",
        snapshot_id="one",
        field_maps=field_maps,
        ratio_convention="new_per_old",
        **kwargs,
    )


def test_collector_receipts_replay_and_full_cap_do_not_prove_absence(tmp_path, field_maps):
    transport = _Transport(
        {
            "/ref/stock_splits": [{"id": "s", "date": "2026-01-06", "ticker": "SYN", "factor": 2}],
            "/ref/dividends": [],
        }
    )
    store = RawStore(tmp_path)
    collection_id, actions, summary = _collect(transport, store, field_maps, limit=1)
    assert len(actions) == 1
    assert summary["window_complete"] is False
    assert store.manifest(collection_id)["complete"] is False
    first = store.manifest(collection_id)
    replay_id, replay, replay_summary = _collect(None, store, field_maps, limit=1)
    assert replay_id == collection_id and replay_summary == summary
    assert replay.available_at.tolist() == actions.available_at.tolist()
    assert store.manifest(collection_id) == first
    assert len(transport.calls) == 2


def test_collector_crash_recovers_original_receipt_without_refetch(
    tmp_path, field_maps, monkeypatch
):
    transport = _Transport(
        {
            "/ref/stock_splits": [{"id": "s", "date": "2026-01-06", "ticker": "SYN", "factor": 2}],
            "/ref/dividends": [],
        }
    )
    store = RawStore(tmp_path)
    publish = storage._atomic_write
    commits = 0

    def fail_once(path, body, *, replace):
        nonlocal commits
        if path.name == "manifest.json" and replace:
            commits += 1
            if commits == 1:
                raise OSError("interrupt")
        return publish(path, body, replace=replace)

    monkeypatch.setattr(storage, "_atomic_write", fail_once)
    with pytest.raises(OSError, match="interrupt"):
        _collect(transport, store, field_maps)
    collection_id, actions, summary = _collect(transport, store, field_maps)
    assert len(transport.calls) == 2
    assert summary["window_complete"] is False
    assert summary["provider_completeness"] == "unqualified"
    assert summary["request_completed"] is True
    assert store.manifest(collection_id)["complete"] is True
    raw = json.loads(store.read_page(collection_id, "split"))
    assert actions.loc[0, "available_at"] == pd.Timestamp(raw["received_at"])


def test_collector_rejects_wrong_window_without_false_complete(tmp_path, field_maps):
    transport = _Transport(
        {
            "/ref/stock_splits": [{"id": "s", "date": "2025-01-06", "ticker": "SYN", "factor": 2}],
            "/ref/dividends": [],
        }
    )
    store = RawStore(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        _collect(transport, store, field_maps)
    (manifest_path,) = tmp_path.glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["complete"] is False
    assert manifest["page_count"] == 1
