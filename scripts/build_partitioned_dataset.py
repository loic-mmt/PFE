"""Build immutable sequence shards from availability-sorted local Parquet files."""

import argparse
import hashlib
import json
from pathlib import Path

from pfe import (
    DatasetLimits,
    fit_partitioned_scaler,
    iter_parquet_observations,
    iter_partitioned_sequences,
    save_partitioned_dataset,
)
from pfe.data.storage import _reject_secrets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-size", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--max-bytes", type=int, default=8_388_608)
    parser.add_argument("--max-archive-bytes", type=int, default=8_388_608)
    args = parser.parse_args(argv)
    try:
        with args.config.open("rb") as stream:
            body = stream.read(65_537)
        if len(body) > 65_536:
            raise ValueError("Dataset configuration exceeds 64 KiB")
        config = json.loads(body)
        _reject_secrets(config)
        limits = DatasetLimits(**{**config.get("limits", {}), "max_bytes": args.max_bytes})
        identities = {}
        for path in args.observations:
            if path.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("Observation files must be outside the output directory")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(65_536), b""):
                    digest.update(block)
            identities[str(path)] = digest.hexdigest()

        def partitions():
            return iter_partitioned_sequences(
                config["decisions"],
                iter_parquet_observations(
                    args.observations,
                    batch_rows=args.batch_rows,
                    max_batch_bytes=limits.max_bytes,
                    max_row_group_bytes=limits.max_bytes,
                ),
                entity_ids=config["entity_ids"],
                registry=config["features"],
                lookback=config.get("lookback", 1),
                stride=config.get("stride", 1),
                label_feature=config.get("labels", {}).get("price_feature"),
                label_horizon=config.get("labels", {}).get("horizon", 1),
                partition_size=args.partition_size,
                limits=limits,
            )

        scaler = (
            fit_partitioned_scaler(partitions(), train_end=config["train_end"])
            if config.get("scale", False)
            else None
        )
        manifest = save_partitioned_dataset(
            args.output,
            partitions(),
            scaler=scaler,
            limits=limits,
            max_archive_bytes=args.max_archive_bytes,
            configuration=dict(config=config, input_sha256=identities),
        )
        print(json.dumps(dict(complete=manifest["complete"], shards=len(manifest["shards"]))))
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
