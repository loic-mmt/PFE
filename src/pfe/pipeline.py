"""Reusable, resumable feature pipelines with explicit operations and budgets.

JSON selects registered operations only, never imports or executes user code.
Python callers may register trusted functions. Network is opt-in; cache replay
does not need credentials. Source-specific collectors keep their raw provenance.
"""

import hashlib
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from pfe.data.http import BoundedHTTPTransport
from pfe.data.storage import RawStore, _json_bytes, _reject_secrets
from pfe.datasets.io import _publish


def _pipeline_json(value):
    def arrays(item):
        if isinstance(item, dict):
            return {key: arrays(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [arrays(nested) for nested in item]
        return item

    value = arrays(value)
    _reject_secrets(value)
    return _json_bytes(value)


@dataclass(frozen=True)
class PipelineLimits:
    max_bytes: int = 8_388_608
    max_response_bytes: int = 524_288
    max_download_bytes: int = 2_097_152
    max_requests: int = 16
    max_rows: int = 25_000
    max_steps: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Pipeline budgets must be positive integers")


@dataclass(frozen=True)
class PipelineStep:
    name: str
    operation: str
    parameters: dict = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    outputs: dict = field(default_factory=dict)
    files: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.name):
            raise ValueError("Pipeline step name must be a simple identifier")
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("Pipeline step needs an operation")
        _reject_secrets(self.parameters)
        _reject_secrets(self.outputs)
        object.__setattr__(self, "needs", tuple(self.needs))
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True)
class StageResult:
    tables: dict[str, pd.DataFrame]
    complete: bool = True
    issues: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineHandler:
    function: object
    revision: str

    def __post_init__(self):
        if not callable(self.function) or not isinstance(self.revision, str) or not self.revision:
            raise ValueError("Handler requires a trusted callable and explicit code revision")


def _target(root, name):
    path = Path(name)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError("Pipeline outputs must be relative paths inside their run directory")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in [result, *result.parents] if p == root or root in p.parents
    ):
        raise ValueError("Pipeline outputs must not cross symlinks")
    return result


def _order(steps):
    by_name = {s.name: s for s in steps}
    if len(by_name) != len(steps) or any(set(s.needs) - by_name.keys() for s in steps):
        raise ValueError("Duplicate step names or unknown dependencies")
    ordered = []
    pending = list(steps)
    while pending:
        ready = [step for step in pending if set(step.needs) <= set(ordered)]
        if not ready:
            raise ValueError("Pipeline dependency cycle")
        for step in ready:
            ordered.append(step.name)
            pending.remove(step)
    return [by_name[name] for name in ordered]


class _PipelineStore(RawStore):
    def __init__(self, context):
        super().__init__(context.root / "raw", max_page_bytes=context.limits.max_bytes)
        self.context = context

    def write_page(self, collection_id, page_id, payload, **options):
        # Count immutable raw bodies once; reserve a small metadata margin.
        body = payload if isinstance(payload, bytes) else _json_bytes(payload)
        directory = self._directory(collection_id)
        page, _ = self._page_paths(directory, page_id)
        if not page.exists():
            self.context.check_disk(len(body) + 4096)
        return super().write_page(collection_id, page_id, payload, **options)


class PipelineContext:
    def __init__(self, root, limits, allow_network, transports):
        self.root, self.limits, self.allow_network = Path(root), limits, allow_network
        self.requests, self.download_bytes = 0, 0
        self.transports = transports or {}
        self.store = _PipelineStore(self)

    def check_disk(self, additional=0):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if used + additional > self.limits.max_bytes:
            raise ValueError("Pipeline exceeds global archive byte budget")

    @property
    def remaining_bytes(self):
        return self.limits.max_bytes - sum(
            p.stat().st_size for p in self.root.rglob("*") if p.is_file()
        )

    def transport(self, provider, *, hosts=None, headers=None):
        context = self
        base = self.transports.get(provider) or BoundedHTTPTransport(
            allowed_hosts=hosts, headers=headers, min_interval=1
        )

        class Budgeted:
            def get(self, *args, max_bytes, **kwargs):
                if not context.allow_network:
                    raise ValueError("Network disabled; provide snapshots or use cached stages")
                if context.requests >= context.limits.max_requests:
                    raise ValueError("Pipeline exceeds request budget")
                remaining = context.limits.max_download_bytes - context.download_bytes
                if remaining < 1:
                    raise ValueError("Pipeline exceeds download byte budget")
                context.requests += 1
                response = base.get(
                    *args,
                    max_bytes=min(max_bytes, remaining, context.limits.max_response_bytes),
                    **kwargs,
                )
                context.download_bytes += response.byte_count
                if context.download_bytes > context.limits.max_download_bytes:
                    raise ValueError("Pipeline source exceeded byte budget")
                return response

        return Budgeted()


def run_pipeline(
    steps, directory, *, handlers=None, limits=None, allow_network=False, transports=None
):
    """Execute a DAG, replay checked stages and publish caller-selected tables.

    Files listed on a step are content-hashed, so editing a source invalidates its
    cache and dependent stages. A changed output is never overwritten: select a
    new output directory for a changed run. A failed step leaves reusable caches,
    but cannot publish a final complete manifest. Partial source status propagates.
    """
    limits = limits or PipelineLimits()
    steps = [PipelineStep(**step) if isinstance(step, dict) else step for step in steps]
    if not steps or len(steps) > limits.max_steps or type(allow_network) is not bool:
        raise ValueError("Pipeline needs a bounded plan and explicit boolean network policy")
    steps = _order(steps)
    registered = default_pipeline_handlers() | (handlers or {})
    if any(
        step.operation not in registered
        or not isinstance(registered[step.operation], PipelineHandler)
        for step in steps
    ):
        raise ValueError("Unregistered or unversioned pipeline operation")
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Pipeline directory must not be a symlink")
    targets = [str(_target(root, path)) for step in steps for path in step.outputs.values()]
    if len(set(targets)) != len(targets):
        raise ValueError("Two stages publish the same output; merge explicitly")
    context = PipelineContext(root, limits, allow_network, transports)
    context.check_disk()
    results, digests, report = {}, {}, []
    for step in steps:
        handler = registered[step.operation]
        paths = set(step.files)
        if step.operation == "local_tables":
            paths.update(step.parameters.get("paths", {}).values())
        inputs = {}
        for filename in sorted(paths):
            path = Path(filename)
            if path.stat().st_size > limits.max_bytes:
                raise ValueError("Pipeline local input exceeds byte budget")
            inputs[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = dict(
            step=asdict(step),
            handler_revision=handler.revision,
            inputs=inputs,
            dependencies={name: digests[name] for name in step.needs},
        )
        key = hashlib.sha256(_pipeline_json(identity)).hexdigest()
        cache = root / "stages" / key
        manifest = cache / "manifest.json"
        cached = manifest.exists()
        if cached:
            body = manifest.read_bytes()
            if len(body) > limits.max_bytes:
                raise ValueError("Pipeline cache manifest exceeds byte budget")
            metadata = json.loads(body)
            if metadata["identity"] != identity:
                # JSON canonicalization converts tuples to arrays.
                if _pipeline_json(metadata["identity"]) != _pipeline_json(identity):
                    raise ValueError("Pipeline cache identity mismatch")
            tables = {}
            for name, details in metadata["tables"].items():
                path = _target(cache, details["path"])
                if path.stat().st_size > limits.max_bytes:
                    raise ValueError("Pipeline cached table exceeds byte budget")
                body = path.read_bytes()
                if hashlib.sha256(body).hexdigest() != details["sha256"]:
                    raise ValueError("Pipeline cached table checksum mismatch")
                import pyarrow.parquet as pq

                parquet = pq.ParquetFile(io.BytesIO(body))
                if (
                    parquet.metadata.num_rows > limits.max_rows
                    or sum(
                        parquet.metadata.row_group(i).total_byte_size
                        for i in range(parquet.num_row_groups)
                    )
                    > limits.max_bytes
                ):
                    raise ValueError("Pipeline cached Parquet expanded size exceeds budget")
                tables[name] = pd.read_parquet(io.BytesIO(body))
            for relative, proof in metadata.get("artifacts", {}).items():
                from pfe.datasets.core import DatasetLimits
                from pfe.datasets.io import load_model_dataset

                artifact = _target(root, relative)
                if (
                    proof.get("kind") not in ("model_dataset", "partitioned_dataset")
                    or hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
                    != proof["manifest_sha256"]
                ):
                    raise ValueError("Pipeline artifact manifest checksum mismatch")
                if proof["kind"] == "model_dataset":
                    load_model_dataset(artifact, limits=DatasetLimits(max_bytes=limits.max_bytes))
                else:
                    from pfe.datasets.partitioned import iter_saved_partitions

                    for _ in iter_saved_partitions(
                        artifact, limits=DatasetLimits(max_bytes=limits.max_bytes)
                    ):
                        pass
            result = StageResult(
                tables,
                metadata["complete"],
                tuple(metadata["issues"]),
                metadata["provenance"],
                metadata.get("artifacts", {}),
            )
        else:
            result = handler.function(
                step.parameters, {name: results[name] for name in step.needs}, context
            )
            if not isinstance(result, StageResult) or type(result.complete) is not bool:
                raise ValueError("Pipeline handler must return StageResult")
            complete = result.complete and all(results[name].complete for name in step.needs)
            result = StageResult(
                result.tables, complete, tuple(result.issues), result.provenance, result.artifacts
            )
            metadata = dict(
                identity=identity,
                complete=complete,
                issues=list(result.issues),
                provenance=result.provenance,
                tables={},
                artifacts=result.artifacts,
            )
            _reject_secrets(metadata)
            cache.mkdir(parents=True, exist_ok=True)
            for name, frame in result.tables.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(frame, pd.DataFrame)
                    or len(frame) > limits.max_rows
                    or frame.memory_usage(index=True, deep=True).sum() > limits.max_bytes
                ):
                    raise ValueError("Pipeline stage table exceeds row/byte budgets")
                stream = io.BytesIO()
                _reject_secrets(frame.to_dict("records"))
                frame.to_parquet(stream)
                body = stream.getvalue()
                filename = "table-" + hashlib.sha256(name.encode()).hexdigest()[:16] + ".parquet"
                context.check_disk(len(body))
                _publish(cache / filename, body)
                metadata["tables"][name] = dict(
                    path=filename, sha256=hashlib.sha256(body).hexdigest(), rows=len(frame)
                )
            body = _pipeline_json(metadata)
            context.check_disk(len(body))
            _publish(manifest, body)
        if set(step.outputs) - result.tables.keys():
            raise ValueError("Pipeline publication names a missing stage table")
        for name, relative in step.outputs.items():
            body = (cache / metadata["tables"][name]["path"]).read_bytes()
            target = _target(root, relative)
            context.check_disk(0 if target.exists() else len(body))
            target.parent.mkdir(parents=True, exist_ok=True)
            _publish(target, body)
        results[step.name] = result
        digests[step.name] = hashlib.sha256(_pipeline_json(metadata)).hexdigest()
        report.append(
            dict(
                name=step.name,
                operation=step.operation,
                complete=result.complete,
                issues=list(result.issues),
                cache_key=key,
                tables=metadata["tables"],
                outputs=step.outputs,
                artifacts=result.artifacts,
            )
        )
    summary = dict(
        schema_version="1",
        complete=all(r.complete for r in results.values()),
        steps=report,
        limits=asdict(limits),
        network_policy="explicit_opt_in",
    )
    # Runtime counters aren't in the immutable run manifest; cached replays must
    # produce the same bytes. They remain available on the returned result only.
    body = _pipeline_json(summary)
    context.check_disk(0 if (root / "manifest.json").exists() else len(body))
    _publish(root / "manifest.json", body)
    return dict(
        manifest=summary,
        results=results,
        requests=context.requests,
        download_bytes=context.download_bytes,
    )


def _local_tables(parameters, inputs, context):
    import pyarrow.parquet as pq

    tables = {}
    for name, filename in parameters["paths"].items():
        file = pq.ParquetFile(filename)
        if (
            file.metadata.num_rows > context.limits.max_rows
            or sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups))
            > context.limits.max_bytes
        ):
            raise ValueError("Pipeline Parquet expanded size exceeds budget")
        tables[name] = file.read().to_pandas()
    return StageResult(tables)


def _merge(parameters, inputs, context):
    from pfe.data.captures import merge_observation_captures

    table = parameters.get("table", "observations")
    return StageResult(
        {table: merge_observation_captures(*(r.tables[table] for r in inputs.values()))}
    )


def _collect(parameters, inputs, context):
    kind = parameters["kind"]
    options = parameters["options"].copy()
    if kind == "fred":
        from pfe.data.fred import collect_fred_series

        result = collect_fred_series(
            context.transport("fred"),
            context.store,
            api_key=os.environ.get("FRED_API_KEY"),
            **options,
        )
        collection, observations, summary = result
        return StageResult(
            {"observations": observations},
            summary["pagination_complete"],
            tuple(summary["issues"]),
            dict(collection_id=collection, coverage=summary["coverage"]),
        )
    elif kind == "eia":
        from pfe.data.eia import ContextLimits, EIAQuery, collect_eia

        options["query"] = EIAQuery(**options["query"])
        options["limits"] = ContextLimits(**options.get("limits", {}))
        result = collect_eia(
            context.transport("eia"),
            context.store,
            api_key=os.environ.get("EIA_API_KEY"),
            **options,
        )
    elif kind == "international":
        from pfe.data.international import (
            INTERNATIONAL_SOURCES,
            InternationalLimits,
            collect_international,
        )

        source = options.pop("source")
        options["limits"] = InternationalLimits(**options.get("limits", {}))
        key = {"opendart": "OPENDART_API_KEY", "estat": "ESTAT_APP_ID"}.get(source)
        result = collect_international(
            source,
            store=context.store,
            credential=os.environ.get(key) if key else None,
            transport=context.transport(source, hosts=INTERNATIONAL_SOURCES[source][1]),
            **options,
        )
    elif kind == "security_master":
        from urllib.parse import urlsplit

        from pfe.data.security_sources import SECURITIES_URLS, collect_security_master

        source = options["source"]
        previous = next(iter(inputs.values())).tables["security_master"] if inputs else None
        result = collect_security_master(
            context.store,
            previous=previous,
            transport=context.transport(source, hosts=[urlsplit(SECURITIES_URLS[source]).hostname]),
            api_key=os.environ.get("ALPHAVANTAGE_API_KEY"),
            **options,
        )
        return StageResult(
            {"security_master": result.history},
            result.complete,
            result.issues,
            dict(collection_id=result.collection_id),
        )
    elif kind == "ucdp":
        from pfe.data.captures import CaptureLimits
        from pfe.data.conflicts import collect_ucdp

        options["limits"] = CaptureLimits(**options.get("limits", {}))
        result = collect_ucdp(
            store=context.store,
            api_token=os.environ.get("UCDP_API_TOKEN"),
            transport=context.transport("ucdp", hosts=["ucdpapi.pcr.uu.se"]),
            **options,
        )
    else:
        raise ValueError("Unsupported registered source collector")
    return StageResult(
        {"observations": result.observations},
        result.complete,
        result.issues,
        dict(collection_id=result.collection_id),
    )


def _timeframes(parameters, inputs, context):
    from pfe.datasets.core import DatasetLimits
    from pfe.datasets.timeframes import build_timeframe_observations

    options = parameters.copy()
    bars = {}
    for entity, reference in options.pop("bars").items():
        stage, table = reference.split(".", 1)
        bars[entity] = inputs[stage].tables[table]
    options["limits"] = DatasetLimits(**options.get("limits", {}))
    result = build_timeframe_observations(bars, **options)
    return StageResult(
        {"observations": result.observations},
        provenance=dict(registry=result.registry.to_dict(), configuration=result.configuration),
    )


def _financial(parameters, inputs, context):
    from pfe.data.financial_mappings import apply_financial_mappings, derive_debt_totals

    facts = next(iter(inputs.values())).tables["observations"]
    mapped = apply_financial_mappings(
        facts, parameters.get("mappings", []), periods=parameters.get("periods", [])
    )
    for recipe in parameters.get("debt_recipes", []):
        mapped = derive_debt_totals(mapped, recipe)
    return StageResult({"observations": mapped})


def default_pipeline_handlers():
    return {
        name: PipelineHandler(
            function,
            "calendar-context-v2"
            if name
            in {
                "calendar_features",
                "political_features",
                "conflict_features",
                "graph_conflict_features",
                "timeframes",
            }
            else "immediate-v1",
        )
        for name, function in (
            ("local_tables", _local_tables),
            ("merge_observations", _merge),
            ("collect", _collect),
            ("timeframes", _timeframes),
            ("financial_mappings", _financial),
            ("dataset", _dataset),
            ("graph", _graph),
            ("graph_bindings", _graph_bindings),
            ("graph_features", _graph_features),
            ("graph_pairs", _graph_pairs),
            ("graph_dataset", _graph_dataset),
            ("calendar_features", _calendar_features),
            ("political_features", _political_features),
            ("conflict_features", _conflict_features),
            ("graph_conflict_features", _graph_conflict_features),
        )
    }


def _dataset(parameters, inputs, context):
    from pfe.data.captures import merge_observation_captures
    from pfe.datasets.core import (
        DatasetLimits,
        build_feature_panel,
        build_sequence_dataset,
        forward_return_labels,
    )
    from pfe.datasets.io import save_model_dataset
    from pfe.datasets.temporal import TrainOnlyScaler, temporal_split

    config = parameters["config"]
    observations = merge_observation_captures(*(r.tables["observations"] for r in inputs.values()))
    limits = DatasetLimits(**config.get("limits", {}))
    from dataclasses import replace

    limits = replace(
        limits, max_bytes=min(limits.max_bytes, max(1, context.remaining_bytes - 4096))
    )
    panel = build_feature_panel(
        config["decisions"],
        observations,
        entity_ids=config["entity_ids"],
        registry=config["features"],
        limits=limits,
    )
    labels = forward_return_labels(panel, **config["labels"]) if config.get("labels") else None
    data = build_sequence_dataset(
        panel,
        lookback=config.get("lookback", 1),
        stride=config.get("stride", 1),
        labels=labels,
        limits=limits,
    )
    split = temporal_split(data, **config["split"]) if config.get("split") else None
    scaler = TrainOnlyScaler.fit(data, split) if config.get("scale", False) else None
    relative = parameters["directory"]
    output = _target(context.root, relative)
    estimated = (
        sum(
            a.nbytes
            for a in (
                panel.values,
                panel.mask,
                panel.available_at_ns,
                panel.observation_at_ns,
                panel.provenance_codes,
                data.x,
                data.mask,
            )
        )
        + 4096
    )
    context.check_disk(0 if (output / "manifest.json").exists() else estimated)
    save_model_dataset(
        output,
        panel,
        sequences=data,
        split=split,
        scaler=scaler,
        configuration=config,
        limits=limits,
    )
    context.check_disk()
    return StageResult(
        {},
        artifacts={
            relative: dict(
                kind="model_dataset",
                manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
            )
        },
    )


def _table_reference(inputs, reference):
    stage, table = reference.split(".", 1)
    return inputs[stage].tables[table]


def _graph_history(inputs, reference):
    from pfe.data.graph import GraphHistory

    result = inputs[reference].tables
    return GraphHistory(result["nodes"], result["edges"], result["issues"])


def _graph(parameters, inputs, context):
    from pfe.data.graph_inputs import entity_registry_from_security_master, import_graph_relations

    if parameters.get("security_master"):
        registry = entity_registry_from_security_master(
            _table_reference(inputs, parameters["security_master"]),
            parameters["decision_at"],
            source_id=parameters.get("master_source_id"),
            aliases=parameters.get("aliases", ()),
        )
    else:
        registry = parameters["entities"]
        if isinstance(registry, str):
            registry = _table_reference(inputs, registry)
    history = import_graph_relations(
        _table_reference(inputs, parameters["relations"]),
        registry,
        max_rows=context.limits.max_rows,
    )
    return StageResult(dict(nodes=history.nodes, edges=history.edges, issues=history.issues))


def _graph_bindings(parameters, inputs, context):
    from pfe.data.graph_inputs import bind_graph_observations, entity_registry_from_security_master

    master = _table_reference(inputs, parameters["security_master"])
    data = bind_graph_observations(
        _table_reference(inputs, parameters["observations"]),
        master,
        parameters["decision_at"],
        **parameters.get("options", {}),
    )
    registry = entity_registry_from_security_master(
        master, parameters["decision_at"], source_id=parameters.get("options", {}).get("source_id")
    )
    return StageResult(dict(observations=data, entities=registry.frame))


def _graph_features(parameters, inputs, context):
    from pfe.datasets.graph import graph_feature_observations

    observations, registry, frame = graph_feature_observations(
        parameters["decisions"],
        _graph_history(inputs, parameters["history"]),
        _table_reference(inputs, parameters["observations"]),
        entity_ids=parameters["entity_ids"],
        feature_units=parameters["feature_units"],
        options=parameters.get("options"),
    )
    return StageResult(
        dict(observations=observations, features=frame),
        provenance=dict(registry=registry.to_dict()),
    )


def _graph_dataset(parameters, inputs, context):
    from dataclasses import replace

    from pfe.data.captures import merge_observation_captures
    from pfe.datasets.core import DatasetLimits
    from pfe.datasets.graph import (
        build_graph_dataset,
        iter_graph_partitions,
        save_partitioned_graph_dataset,
    )
    from pfe.datasets.io import save_model_dataset

    config = parameters["config"]
    history = _graph_history(inputs, parameters["history"])
    references = parameters["observations"]
    observations = merge_observation_captures(
        *(_table_reference(inputs, ref) for ref in references)
    )
    limits = DatasetLimits(**config.get("limits", {}))
    limits = replace(
        limits, max_bytes=min(limits.max_bytes, max(1, context.remaining_bytes - 4096))
    )
    dataset = build_graph_dataset(
        config["decisions"],
        observations,
        history,
        registry=config["features"],
        entity_ids=config["entity_ids"],
        limits=limits,
        include_context=config.get("include_context", True),
        graph_feature_units=config.get("graph_feature_units"),
        graph_feature_options=config.get("graph_feature_options"),
    )
    relative = parameters["directory"]
    output = _target(context.root, relative)
    if config.get("partition_size"):
        data = dataset.observations.sort_values("available_at", kind="stable")
        batches = (data.iloc[start : start + 256] for start in range(0, len(data), 256))
        parts = iter_graph_partitions(
            config["decisions"],
            batches,
            history,
            entity_ids=dataset.panel.entity_ids,
            registry=dataset.panel.registry,
            partition_size=config["partition_size"],
            limits=limits,
        )
        save_partitioned_graph_dataset(
            output,
            parts,
            history,
            configuration=config,
            limits=limits,
            max_archive_bytes=max(1, context.remaining_bytes - 4096),
        )
        kind = "partitioned_dataset"
    else:
        save_model_dataset(
            output,
            dataset.panel,
            graphs=dataset.snapshots(),
            configuration=dict(config=config, target_entity_ids=dataset.target_entity_ids),
            limits=limits,
        )
        kind = "model_dataset"
    context.check_disk()
    return StageResult(
        {},
        artifacts={
            relative: dict(
                kind=kind,
                manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
            )
        },
    )


def _graph_pairs(parameters, inputs, context):
    from pfe.features.graph_analytics import compute_graph_pair_features

    pairs = compute_graph_pair_features(
        parameters["decisions"],
        _graph_history(inputs, parameters["history"]),
        pairs=parameters["pairs"],
        **parameters.get("options", {}),
    )
    return StageResult({"pairs": pairs})


def _context_result(frame, parameters):
    from pfe.datasets.context import context_feature_observations

    tables = {"features": frame}
    if "observation_options" in parameters:
        tables["observations"] = context_feature_observations(
            frame, **parameters["observation_options"]
        )
    return StageResult(tables, provenance=dict(semantics=frame.attrs))


def _calendar_features(parameters, inputs, context):
    from pfe.features.calendar import compute_calendar_features

    options = parameters.get("options", {}).copy()
    if "schedule" in parameters:
        options["schedule"] = _table_reference(inputs, parameters["schedule"])
    return _context_result(
        compute_calendar_features(parameters["decisions"], **options), parameters
    )


def _political_features(parameters, inputs, context):
    from pfe.features.politics import compute_political_features

    history = _table_reference(inputs, parameters["history"]) if "history" in parameters else None
    return _context_result(
        compute_political_features(
            parameters["decisions"], history=history, **parameters.get("options", {})
        ),
        parameters,
    )


def _conflict_features(parameters, inputs, context):
    from pfe.features.conflicts import compute_conflict_features

    return _context_result(
        compute_conflict_features(
            parameters["decisions"],
            _table_reference(inputs, parameters["observations"]),
            **parameters.get("options", {}),
        ),
        parameters,
    )


def _graph_conflict_features(parameters, inputs, context):
    from pfe.features.conflicts import compute_graph_conflict_features

    options = parameters.get("options", {}).copy()
    mapping = parameters["country_nodes"]
    # JSON object keys are strings; only this explicit declared mapping is converted.
    mapping = {int(key): value for key, value in mapping.items()}
    frame = compute_graph_conflict_features(
        parameters["decisions"],
        _table_reference(inputs, parameters["observations"]),
        _graph_history(inputs, parameters["history"]),
        entity_ids=parameters["entity_ids"],
        country_nodes=mapping,
        **options,
    )
    return _context_result(frame, parameters)
