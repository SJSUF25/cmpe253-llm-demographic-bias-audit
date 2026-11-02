"""Immutable downloader and schema gate for the pinned CAB and BBQ sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/raw_data_sources.json"
USER_AGENT = "cmpe253-llm-bias-audit/0.1"
PIN_PATTERN = re.compile(r"^[0-9a-f]{40}$")
KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]*$")


class DataValidationError(ValueError):
    """Raised when downloaded bytes violate the frozen source contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DataValidationError(f"Unsafe source path: {value}")
    return Path(*pure.parts)


def _require_fields(
    rows: list[dict[str, Any]], required: set[str], source_name: str
) -> list[str]:
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise DataValidationError(
                f"{source_name} row {index} is missing fields: {names}"
            )
    return sorted(set.intersection(*(set(row) for row in rows))) if rows else []


def _parse_json(content: bytes, source_name: str) -> Any:
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataValidationError(f"Invalid JSON in {source_name}: {error}") from error


def validate_cab(content: bytes, specification: dict[str, Any]) -> dict[str, Any]:
    rows = _parse_json(content, specification["path"])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise DataValidationError("CAB must be a JSON list of objects")
    if len(rows) != specification["expected_rows"]:
        raise DataValidationError(
            f"CAB row count {len(rows)} != {specification['expected_rows']}"
        )
    fields = _require_fields(
        rows, set(specification["required_fields"]), specification["path"]
    )
    inventory = Counter(f"{row['attribute']}|{row['expl_impl']}" for row in rows)
    for cell, expected in specification["required_inventory"].items():
        if inventory[cell] != expected:
            raise DataValidationError(
                f"CAB inventory {cell}={inventory[cell]} != {expected}"
            )
    return {
        "row_count": len(rows),
        "fields": fields,
        "inventory": dict(sorted(inventory.items())),
    }


def validate_bbq(
    content: bytes,
    specification: dict[str, Any],
    required_fields: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise DataValidationError(
                f"{specification['path']} contains blank line {line_number}"
            )
        row = _parse_json(line, f"{specification['path']}:{line_number}")
        if not isinstance(row, dict):
            raise DataValidationError(
                f"{specification['path']} line {line_number} is not an object"
            )
        rows.append(row)
    if len(rows) != specification["expected_rows"]:
        raise DataValidationError(
            f"{specification['path']} row count {len(rows)} != "
            f"{specification['expected_rows']}"
        )
    fields = _require_fields(rows, required_fields, specification["path"])
    categories = {row["category"] for row in rows}
    if categories != {specification["expected_category"]}:
        raise DataValidationError(
            f"{specification['path']} has unexpected categories: {sorted(categories)}"
        )
    return {
        "row_count": len(rows),
        "fields": fields,
        "category": specification["expected_category"],
    }


def _validate_bytes(
    content: bytes,
    dataset: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    actual_size = len(content)
    actual_hash = sha256_bytes(content)
    if actual_size != specification["size_bytes"]:
        raise DataValidationError(
            f"{specification['path']} size {actual_size} != "
            f"{specification['size_bytes']}"
        )
    if actual_hash != specification["sha256"]:
        raise DataValidationError(
            f"{specification['path']} SHA-256 {actual_hash} != "
            f"{specification['sha256']}"
        )
    if dataset["key"] == "D1":
        return validate_cab(content, specification)
    return validate_bbq(content, specification, set(dataset["required_fields"]))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise DataValidationError("Unsupported raw-data source schema")
    keys: set[str] = set()
    for dataset in config.get("datasets", []):
        key = dataset.get("key", "")
        if not KEY_PATTERN.fullmatch(key) or key in keys:
            raise DataValidationError(f"Invalid or duplicate dataset key: {key}")
        keys.add(key)
        revision = dataset.get("revision", "")
        if not PIN_PATTERN.fullmatch(revision):
            raise DataValidationError(f"Dataset {key} has an unpinned revision")
        for source in dataset.get("files", []):
            _safe_relative_path(source["path"])
            if revision not in source["url"]:
                raise DataValidationError(
                    f"Dataset {key} URL is not pinned to its revision"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")):
                raise DataValidationError(f"Dataset {key} has an invalid SHA-256")
    if keys != {"D1", "D2"}:
        raise DataValidationError(f"Expected D1 and D2, found {sorted(keys)}")
    by_key = {dataset["key"]: dataset for dataset in config["datasets"]}
    d1_paths = {source["path"] for source in by_key["D1"]["files"]}
    if d1_paths != {"cab.json"}:
        raise DataValidationError(f"Unexpected CAB source files: {sorted(d1_paths)}")
    d2_categories = {
        source.get("expected_category") for source in by_key["D2"]["files"]
    }
    expected_categories = {"Gender_identity", "Race_ethnicity", "Race_x_gender"}
    if d2_categories != expected_categories:
        raise DataValidationError(
            f"Unexpected BBQ categories: {sorted(str(x) for x in d2_categories)}"
        )


def download_datasets(
    shared_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    *,
    verify_remote: bool = False,
    fetcher: Callable[[str], bytes] = fetch_url,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """Download or verify pinned sources without ever overwriting raw bytes."""
    shared_root = shared_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)

    raw_root = shared_root / "raw_immutable"
    manifest_root = shared_root / "manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    file_entries: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        destination_dir = raw_root / dataset["key"] / dataset["revision"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in dataset["files"]:
            destination = destination_dir / _safe_relative_path(source["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                local_content = destination.read_bytes()
                validation = _validate_bytes(local_content, dataset, source)
                if verify_remote:
                    remote_content = fetcher(source["url"])
                    _validate_bytes(remote_content, dataset, source)
                    if remote_content != local_content:
                        raise DataValidationError(
                            f"Remote bytes differ from immutable {destination}"
                        )
            else:
                local_content = fetcher(source["url"])
                validation = _validate_bytes(local_content, dataset, source)
                with destination.open("xb") as stream:
                    stream.write(local_content)
                    stream.flush()
                    os.fsync(stream.fileno())
            file_entries.append(
                {
                    "dataset_key": dataset["key"],
                    "repo_id": dataset["repo_id"],
                    "revision": dataset["revision"],
                    "license": dataset["license"],
                    "source_url": source["url"],
                    "storage_path": str(destination.relative_to(shared_root)),
                    "size_bytes": len(local_content),
                    "sha256": sha256_bytes(local_content),
                    "validation": validation,
                }
            )

    stable_manifest = {
        "schema_version": 1,
        "source_config_id": config["config_id"],
        "source_config_sha256": sha256_bytes(config_bytes),
        "files": file_entries,
    }
    manifest_path = manifest_root / config["manifest_name"]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing[key] for key in stable_manifest}
        if comparable != stable_manifest:
            raise DataValidationError(
                f"Existing manifest does not match verified data: {manifest_path}"
            )
        return manifest_path

    clock = now or (lambda: datetime.now(UTC))
    manifest = {
        **stable_manifest,
        "retrieved_at_utc": clock().astimezone(UTC).isoformat(),
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest_path
