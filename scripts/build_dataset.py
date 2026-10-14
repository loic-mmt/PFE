"""Build small GRU/TCN, GNN and Event Encoder artifacts from local inputs only."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pfe import (
    DatasetLimits,
    EventHistory,
    GraphHistory,
    TrainOnlyScaler,
    build_event_tensor,
    build_feature_panel,
    build_sequence_dataset,
    forward_return_labels,
    iter_graph_tensors,
    save_model_dataset,
    temporal_split,
)
from pfe.data.graph import ISSUE_COLUMNS
from pfe.data.storage import _reject_secrets


def read_small(path, maximum):
    with Path(path).open("rb") as stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("Local input exceeds byte budget")
    return body


def read_parquet(path, maximum):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("Parquet source exceeds byte budget")
    table = pq.ParquetFile(path)
    meta = table.metadata
    if (
        meta.num_rows > 20_000
        or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)) > maximum * 4
    ):
        raise ValueError("Parquet expanded rows/bytes exceed budget")
    return pd.read_parquet(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=Path)
    parser.add_argument("--edges", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--event-links", type=Path)
    parser.add_argument("--enrichments", type=Path)
    parser.add_argument("--max-bytes", type=int, default=8_388_608)
    parser.add_argument("--max-samples", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        limits = DatasetLimits(max_bytes=args.max_bytes, max_samples=args.max_samples)
        config = json.loads(read_small(args.config, min(65536, limits.max_bytes)))
        _reject_secrets(config)
        if bool(args.nodes) != bool(args.edges) or bool(args.events) != bool(args.event_links):
            raise ValueError("Supply paired graph nodes/edges or events/links")
        paths = [
            p
            for p in (
                args.observations,
                args.config,
                args.nodes,
                args.edges,
                args.events,
                args.event_links,
                args.enrichments,
            )
            if p is not None
        ]
        if any(args.output.resolve() == p.resolve() for p in paths):
            raise ValueError("Dataset output must be separate from input files")
        hashes = {
            str(p): hashlib.sha256(read_small(p, limits.max_bytes)).hexdigest() for p in paths
        }
        panel = build_feature_panel(
            config["decisions"],
            read_parquet(args.observations, limits.max_bytes),
            entity_ids=config["entity_ids"],
            registry=config["features"],
            limits=limits,
        )
        labels = forward_return_labels(panel, **config["labels"]) if config.get("labels") else None
        data = build_sequence_dataset(
            panel,
            lookback=config.get("lookback", 1),
            labels=labels,
            stride=config.get("stride", 1),
            limits=limits,
        )
        split = temporal_split(data, **config["split"]) if config.get("split") else None
        scaler = TrainOnlyScaler.fit(data, split) if config.get("scale", False) else None
        graphs = ()
        issues = pd.DataFrame(columns=ISSUE_COLUMNS)
        if args.nodes:
            graph = GraphHistory(
                read_parquet(args.nodes, limits.max_bytes),
                read_parquet(args.edges, limits.max_bytes),
                issues,
            )
            graphs = iter_graph_tensors(panel, graph, limits=limits)
        events = None
        if args.events:
            history = EventHistory(
                read_parquet(args.events, limits.max_bytes),
                read_parquet(args.event_links, limits.max_bytes),
                issues,
            )
            outputs = (
                json.loads(read_small(args.enrichments, limits.max_bytes))
                if args.enrichments
                else ()
            )
            events = build_event_tensor(
                config["decisions"],
                history,
                entity_ids=config["entity_ids"],
                enrichments=outputs,
                limits=limits,
                **config.get("event_config", {}),
            )
        manifest = save_model_dataset(
            args.output,
            panel,
            sequences=data,
            graphs=graphs,
            events=events,
            split=split,
            scaler=scaler,
            configuration={"config": config, "input_sha256": hashes},
            limits=limits,
        )
        print(f"Saved {len(data.x)} samples: {args.output}; {manifest['data_sha256'][:16]}")
        return manifest
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
