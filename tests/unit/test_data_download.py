import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from data.download import DataValidationError, download_datasets


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _source(path: str, content: bytes, **extra: object) -> dict[str, object]:
    revision = "a" * 40
    return {
        "path": path,
        "url": f"https://example.test/{revision}/{path}",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "format": "jsonl" if path.endswith("jsonl") else "json",
        "expected_rows": 1,
        **extra,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    cab_row = {
        "attribute": "gender",
        "domain": "domain",
        "domain_explicit": "explicit domain",
        "example": "example",
        "expl_impl": "explicit",
        "superdomain": "superdomain",
        "superdomain_explicit": "explicit superdomain",
        "topic": "topic",
    }
    bbq_fields = {
        "additional_metadata": {},
        "ans0": "a",
        "ans1": "b",
        "ans2": "c",
        "answer_info": {},
        "context": "context",
        "context_condition": "ambig",
        "example_id": 1,
        "label": 0,
        "question": "question",
        "question_index": "1",
        "question_polarity": "neg",
    }
    content = {"cab.json": _json_bytes([cab_row])}
    for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender"):
        content[f"{category}.jsonl"] = (
            _json_bytes({**bbq_fields, "category": category}) + b"\n"
        )
    required_bbq = sorted({*bbq_fields, "category"})
    revision = "a" * 40
    config = {
        "schema_version": 1,
        "config_id": "test-raw-data-v1",
        "manifest_name": "raw_data_manifest.json",
        "datasets": [
            {
                "key": "D1",
                "name": "CAB",
                "repo_id": "test/cab",
                "revision": revision,
                "license": "MIT",
                "files": [
                    _source(
                        "cab.json",
                        content["cab.json"],
                        required_fields=sorted(cab_row),
                        required_inventory={"gender|explicit": 1},
                    )
                ],
            },
            {
                "key": "D2",
                "name": "BBQ",
                "repo_id": "test/bbq",
                "revision": revision,
                "license": "CC-BY-4.0",
                "required_fields": required_bbq,
                "files": [
                    _source(
                        f"{category}.jsonl",
                        content[f"{category}.jsonl"],
                        expected_category=category,
                    )
                    for category in (
                        "Gender_identity",
                        "Race_ethnicity",
                        "Race_x_gender",
                    )
                ],
            },
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, content


def test_download_is_immutable_idempotent_and_remotely_verifiable(
    tmp_path: Path,
) -> None:
    config_path, content = _fixture(tmp_path)

    def fetcher(url: str) -> bytes:
        return content[url.rsplit("/", 1)[-1]]

    shared = tmp_path / "shared"
    manifest_path = download_datasets(
        shared,
        config_path,
        fetcher=fetcher,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    first_manifest = manifest_path.read_bytes()
    assert len(json.loads(first_manifest)["files"]) == 4

    download_datasets(shared, config_path, fetcher=fetcher, verify_remote=True)
    assert manifest_path.read_bytes() == first_manifest


def test_download_rejects_tampered_immutable_file(tmp_path: Path) -> None:
    config_path, content = _fixture(tmp_path)

    def fetcher(url: str) -> bytes:
        return content[url.rsplit("/", 1)[-1]]

    shared = tmp_path / "shared"
    download_datasets(shared, config_path, fetcher=fetcher)
    raw = shared / "raw_immutable" / "D1" / ("a" * 40) / "cab.json"
    raw.write_text("tampered", encoding="utf-8")

    with pytest.raises(DataValidationError, match="size"):
        download_datasets(shared, config_path, fetcher=fetcher)


def test_download_rejects_schema_drift_before_writing(tmp_path: Path) -> None:
    config_path, content = _fixture(tmp_path)
    content["cab.json"] = _json_bytes([{"attribute": "gender"}])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["datasets"][0]["files"][0]
    source["size_bytes"] = len(content["cab.json"])
    source["sha256"] = hashlib.sha256(content["cab.json"]).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fetcher(url: str) -> bytes:
        return content[url.rsplit("/", 1)[-1]]

    with pytest.raises(DataValidationError, match="missing fields"):
        download_datasets(tmp_path / "shared", config_path, fetcher=fetcher)

    assert not list((tmp_path / "shared" / "raw_immutable").rglob("cab.json"))
