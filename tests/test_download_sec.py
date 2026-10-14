"""Offline SEC CLI replay and canonical/raw consistency."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from pfe.data.http import HTTPResponse


class Fake:
    def get(self, url, **options):
        accession = "0000001234-25-000001"
        payload = dict(
            cik=1234,
            filings=dict(
                recent=dict(
                    accessionNumber=[accession],
                    filingDate=["2025-05-01"],
                    acceptanceDateTime=["2025-05-01T20:00:00Z"],
                    form=["10-Q"],
                ),
                files=[],
            ),
        )
        if "/companyfacts/" in url:
            payload = dict(
                cik=1234,
                facts={
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    dict(
                                        start="2025-01-01",
                                        end="2025-03-31",
                                        val=100,
                                        accn=accession,
                                        filed="2025-05-01",
                                        fy=2025,
                                        fp="Q1",
                                    )
                                ]
                            }
                        }
                    }
                },
            )
        return HTTPResponse(json.dumps(payload).encode(), "2026-10-12T10:00:00Z")


def cli():
    path = Path(__file__).parents[1] / "scripts" / "download_sec.py"
    spec = importlib.util.spec_from_file_location("download_sec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_output_replay_and_recovery(tmp_path):
    command = cli()
    args = [
        "--cik",
        "1234",
        "--user-agent",
        "PFE contact@example.com",
        "--audit-id",
        "fixture",
        "--store-dir",
        str(tmp_path / "raw"),
        "--output-dir",
        str(tmp_path / "canonical"),
    ]
    data = command.main(args, transport=Fake())
    paths = list((tmp_path / "canonical").glob("*.parquet"))
    assert len(paths) == 2
    old = {path: path.read_bytes() for path in paths}
    command.main(args, transport=Fake())
    assert {path: path.read_bytes() for path in paths} == old
    for path in paths:
        assert pd.read_parquet(path).attrs["raw_collection_id"] == data.collection_id
    paths[0].unlink()
    assert command.main(args, transport=Fake()).collection_id == data.collection_id


def test_bad_user_agent_creates_no_raw_request(tmp_path):
    with pytest.raises(SystemExit):
        cli().main(
            [
                "--cik",
                "1234",
                "--user-agent",
                "bad",
                "--audit-id",
                "bad",
                "--store-dir",
                str(tmp_path / "raw"),
            ],
            transport=Fake(),
        )
    assert not (tmp_path / "raw").exists()
