"""Run a bounded library pipeline; network and bulk data are never implicit."""

import argparse
import json
from pathlib import Path

from pfe.pipeline import PipelineLimits, run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    try:
        with args.config.open("rb") as stream:
            body = stream.read(65537)
        if len(body) > 65536:
            raise ValueError("Pipeline config exceeds 64 KiB")
        config = json.loads(body)
        result = run_pipeline(
            config["steps"],
            args.output,
            limits=PipelineLimits(**config.get("limits", {})),
            allow_network=args.allow_network,
        )
        print(
            json.dumps(
                dict(
                    complete=result["manifest"]["complete"],
                    steps=len(result["manifest"]["steps"]),
                    requests=result["requests"],
                    download_bytes=result["download_bytes"],
                )
            )
        )
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
