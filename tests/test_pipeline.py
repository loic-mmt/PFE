"""Reusable DAGs, content hashes, partial status, bounded publication and resume."""

import numpy as np
import pandas as pd
import pytest

from pfe.pipeline import PipelineHandler, PipelineLimits, StageResult, run_pipeline


def test_failure_resume_dependency_order_and_partial_propagation(tmp_path):
    calls = []
    fail = True

    def first(parameters, inputs, context):
        calls.append("first")
        return StageResult(
            {"table": pd.DataFrame({"value": [1.0]})},
            complete=False,
            issues=("selected_sample_partial",),
        )

    def second(parameters, inputs, context):
        nonlocal fail
        calls.append("second")
        if fail:
            fail = False
            raise ValueError("injected interruption")
        return StageResult({"table": inputs["first"].tables["table"]})

    handlers = dict(one=PipelineHandler(first, "v1"), two=PipelineHandler(second, "v1"))
    steps = [
        dict(
            name="second",
            operation="two",
            needs=["first"],
            outputs={"table": "market/table.parquet"},
        ),
        dict(name="first", operation="one"),
    ]
    with pytest.raises(ValueError, match="interruption"):
        run_pipeline(steps, tmp_path, handlers=handlers)
    assert not (tmp_path / "manifest.json").exists()
    result = run_pipeline(steps, tmp_path, handlers=handlers)
    assert calls == ["first", "second", "second"]
    assert not result["manifest"]["complete"] and not result["results"]["second"].complete
    again = run_pipeline(steps, tmp_path, handlers=handlers)
    assert again["manifest"] == result["manifest"] and len(calls) == 3


def test_local_multitimeframe_to_dataset_pipeline_and_tampered_cache(tmp_path):
    times = pd.date_range("2026-01-01T10:00:00Z", periods=20, freq="min")
    values = 100 + np.arange(20)
    frame = pd.DataFrame(
        dict(
            open=values,
            high=values + 1,
            low=values - 1,
            close=values,
            volume=1.0,
            available_at=times + pd.Timedelta("1min"),
        ),
        index=times,
    )
    path = tmp_path / "input.parquet"
    frame.to_parquet(path)
    specs = [
        dict(
            frequency=f,
            names=["close", "returns.simple"],
            units={"close": "USD", "returns.simple": "fraction"},
        )
        for f in ("1min", "5min")
    ]
    registry = [
        dict(
            name=f"tf[{s['frequency']}].{name}",
            metric=f"tf[{s['frequency']}].{name}",
            unit=s["units"][name],
            source_id="fixture",
        )
        for s in specs
        for name in s["names"]
    ]
    steps = [
        dict(name="source", operation="local_tables", parameters=dict(paths={"A": str(path)})),
        dict(
            name="features",
            operation="timeframes",
            needs=["source"],
            parameters=dict(
                bars={"A": "source.A"},
                timeframes=specs,
                source_frequency="1min",
                source_id="fixture",
                revision_id="1",
            ),
            outputs={"observations": "market/features.parquet"},
        ),
        dict(
            name="dataset",
            operation="dataset",
            needs=["features"],
            parameters=dict(
                directory="datasets/intraday",
                config=dict(
                    decisions=[d.isoformat() for d in times + pd.Timedelta("1min")],
                    entity_ids=["A"],
                    features=registry,
                    lookback=3,
                    labels=dict(price_feature="tf[1min].close", horizon=2),
                ),
            ),
        ),
    ]
    root = tmp_path / "run"
    one = run_pipeline(steps, root)
    two = run_pipeline(steps, root)
    assert one["manifest"] == two["manifest"] and two["requests"] == 0
    assert (root / "datasets/intraday/arrays.npz").exists()
    manifest = root / "datasets/intraday/manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        run_pipeline(steps, root)


def test_pipeline_cycles_paths_secrets_and_global_caps(tmp_path):
    bad = [
        dict(name="a", operation="local_tables", needs=["b"]),
        dict(name="b", operation="local_tables", needs=["a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        run_pipeline(bad, tmp_path / "cycle")
    with pytest.raises(ValueError, match="relative"):
        run_pipeline(
            [dict(name="a", operation="local_tables", outputs={"x": "../escape.parquet"})],
            tmp_path / "paths",
        )
    with pytest.raises(ValueError):
        run_pipeline(
            [dict(name="a", operation="local_tables", parameters=dict(api_key="SYNTHETIC_SECRET"))],
            tmp_path / "secret",
        )
    handler = PipelineHandler(
        lambda p, i, c: StageResult({"a": pd.DataFrame({"x": np.ones(100)})}), "v1"
    )
    with pytest.raises(ValueError, match="byte budget"):
        run_pipeline(
            [dict(name="a", operation="test")],
            tmp_path / "tiny",
            handlers={"test": handler},
            limits=PipelineLimits(max_bytes=100),
        )
    assert not (tmp_path / "tiny/manifest.json").exists()


def test_network_disabled_and_request_budget_are_enforced(tmp_path):
    from pfe.data.http import HTTPResponse

    class Fake:
        def get(self, *args, **kwargs):
            return HTTPResponse(b"{}", "2026-01-01T00:00:00Z")

    def handler(p, i, c):
        c.transport("fake").get("https://example.org/", max_bytes=10)
        return StageResult({})

    handlers = {"test": PipelineHandler(handler, "v1")}
    steps = [dict(name="a", operation="test"), dict(name="b", operation="test", needs=["a"])]
    with pytest.raises(ValueError, match="Network disabled"):
        run_pipeline(steps, tmp_path / "off", handlers=handlers, transports={"fake": Fake()})
    with pytest.raises(ValueError, match="request budget"):
        run_pipeline(
            steps,
            tmp_path / "cap",
            handlers=handlers,
            transports={"fake": Fake()},
            allow_network=True,
            limits=PipelineLimits(max_requests=1),
        )


def test_fred_tuple_result_is_normalized_and_cached(tmp_path, monkeypatch):
    calls = []

    def collect(*args, **kwargs):
        calls.append(1)
        return (
            "fixture",
            pd.DataFrame({"value": [1.0]}),
            dict(
                pagination_complete=False,
                issues=["row_cap_reached"],
                coverage={"observations": False},
            ),
        )

    monkeypatch.setattr("pfe.data.fred.collect_fred_series", collect)
    steps = [dict(name="fred", operation="collect", parameters=dict(kind="fred", options={}))]
    first = run_pipeline(steps, tmp_path)
    second = run_pipeline(steps, tmp_path)
    assert first["manifest"] == second["manifest"] and calls == [1]
    assert not first["manifest"]["complete"]
    assert first["results"]["fred"].issues == ("row_cap_reached",)
