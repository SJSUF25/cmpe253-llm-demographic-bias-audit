"""Deterministically expand CAB families into the locked prompt contract."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cab_build.json"
MARKER_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")
CANONICAL_FIELDS = (
    "attribute",
    "superdomain_explicit",
    "domain_explicit",
    "topic",
)
MODE_ORDER = {"explicit": 0, "implicit": 1}


class CabBuildError(ValueError):
    """Raised when raw CAB data violates the frozen build contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    )


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CabBuildError(f"Unsafe configured path: {value}")
    return Path(*path.parts)


def _link_key(row: dict[str, Any]) -> tuple[str, ...]:
    try:
        return tuple(canonical_text(str(row[field])) for field in CANONICAL_FIELDS)
    except KeyError as error:
        raise CabBuildError(f"CAB row is missing {error.args[0]}") from error


def _markers(example: str, arity: int) -> list[list[str]]:
    markers = [match.split("/") for match in MARKER_PATTERN.findall(example)]
    if not markers:
        raise CabBuildError("CAB example has no {{...}} identity marker")
    if any(
        len(marker) != arity or any(not value for value in marker) for marker in markers
    ):
        raise CabBuildError(f"CAB identity marker does not have arity {arity}")
    return markers


def _expand(
    example: str, markers: list[list[str]], index: int
) -> tuple[str, list[str]]:
    values = [marker[index] for marker in markers]
    value_iterator = iter(values)
    expanded = MARKER_PATTERN.sub(lambda _match: next(value_iterator), example)
    if MARKER_PATTERN.search(expanded):
        raise CabBuildError("CAB expansion left an identity marker behind")
    return prompt_text(expanded), values


def _scenario_id(explicit_row: dict[str, Any], revision: str) -> str:
    example = MARKER_PATTERN.sub("{IDENTITY}", str(explicit_row["example"]))
    canonical = {
        "revision": revision,
        "attribute": canonical_text(str(explicit_row["attribute"])),
        "superdomain_explicit": canonical_text(
            str(explicit_row["superdomain_explicit"])
        ),
        "domain_explicit": canonical_text(str(explicit_row["domain_explicit"])),
        "topic": canonical_text(str(explicit_row["topic"])),
        "example": canonical_text(example),
    }
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256_bytes(serialized)[:20]


def _split_for(scenario_id: str, split_config: dict[str, Any]) -> str:
    namespace = split_config["namespace"]
    digest = hashlib.sha256(f"{namespace}{scenario_id}".encode()).hexdigest()
    bucket = int(digest, 16) % int(split_config["modulus"])
    return (
        "development" if bucket in set(split_config["development_buckets"]) else "test"
    )


def build_prompt_rows(
    raw_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate, link, expand, and identify every race/gender CAB prompt."""
    attributes: dict[str, list[str]] = config["attributes"]
    selected = [row for row in raw_rows if row.get("attribute") in attributes]
    by_key: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        mode = row.get("expl_impl")
        if mode not in MODE_ORDER:
            raise CabBuildError(f"Unexpected CAB explicit/implicit value: {mode}")
        key = _link_key(row)
        if mode in by_key[key]:
            raise CabBuildError(f"Duplicate {mode} CAB row for linkage key {key}")
        by_key[key][mode] = row

    incomplete = [key for key, modes in by_key.items() if set(modes) != set(MODE_ORDER)]
    if incomplete:
        raise CabBuildError(
            f"CAB has {len(incomplete)} linkage keys without one explicit "
            "and implicit row"
        )

    records: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    for modes in by_key.values():
        explicit = modes["explicit"]
        attribute = canonical_text(str(explicit["attribute"]))
        groups = attributes[attribute]
        scenario_id = _scenario_id(explicit, config["source_revision"])
        if scenario_id in seen_scenarios:
            raise CabBuildError(f"Duplicate scenario_id: {scenario_id}")
        seen_scenarios.add(scenario_id)
        split = _split_for(scenario_id, config["split"])
        for mode in ("explicit", "implicit"):
            row = modes[mode]
            markers = _markers(str(row["example"]), len(groups))
            for index, group in enumerate(groups):
                text, identity_values = _expand(str(row["example"]), markers, index)
                text_hash = sha256_bytes(text.encode())
                prompt_id = sha256_bytes(
                    f"{scenario_id}|{mode}|{group}|{text_hash}".encode()
                )[:20]
                records.append(
                    {
                        "scenario_id": scenario_id,
                        "prompt_id": prompt_id,
                        "dataset": config["dataset"],
                        "source_revision": config["source_revision"],
                        "split": split,
                        "attribute": attribute,
                        "group": group,
                        "identity_values": identity_values,
                        "intersection_group": None,
                        "explicit_implicit": mode,
                        "superdomain": canonical_text(
                            str(explicit["superdomain_explicit"])
                        ),
                        "domain": canonical_text(str(explicit["domain_explicit"])),
                        "template_id": scenario_id,
                        "prompt_text": text,
                        "prompt_sha256": text_hash,
                        "paired_variant_ids": [],
                        "license": config["license"],
                    }
                )

    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    prompt_counts = Counter(record["prompt_id"] for record in records)
    duplicate_prompts = [
        prompt_id for prompt_id, count in prompt_counts.items() if count != 1
    ]
    if duplicate_prompts:
        raise CabBuildError(f"Duplicate prompt_id values: {duplicate_prompts}")
    for record in records:
        pair_groups[(record["scenario_id"], record["explicit_implicit"])].append(record)
    for variants in pair_groups.values():
        ids = sorted(record["prompt_id"] for record in variants)
        for record in variants:
            record["paired_variant_ids"] = [
                prompt_id for prompt_id in ids if prompt_id != record["prompt_id"]
            ]

    group_order = {
        (attribute, group): index
        for attribute, groups in attributes.items()
        for index, group in enumerate(groups)
    }
    records.sort(
        key=lambda row: (
            row["scenario_id"],
            MODE_ORDER[row["explicit_implicit"]],
            group_order[(row["attribute"], row["group"])],
        )
    )
    _validate_counts(records, config)
    return records


def _validate_counts(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    expected = config["expected"]
    families = Counter(record["explicit_implicit"] for record in records)
    unique = {
        mode: len(
            {
                record["scenario_id"]
                for record in records
                if record["explicit_implicit"] == mode
            }
        )
        for mode in MODE_ORDER
    }
    if unique["explicit"] != expected["explicit_families"]:
        raise CabBuildError("Unexpected explicit CAB family count")
    if unique["implicit"] != expected["implicit_families"]:
        raise CabBuildError("Unexpected implicit CAB family count")
    if families["explicit"] != expected["explicit_prompts"]:
        raise CabBuildError("Unexpected explicit CAB prompt count")
    if families["implicit"] != expected["implicit_prompts"]:
        raise CabBuildError("Unexpected implicit CAB prompt count")
    if len(records) != expected["total_prompts"]:
        raise CabBuildError("Unexpected total CAB prompt count")


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode()


def build_cab(shared_root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Build the CAB prompt JSONL and deterministic checksum manifest."""
    shared_root = shared_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1:
        raise CabBuildError("Unsupported CAB build configuration")
    input_path = shared_root / _safe_relative_path(config["input_path"])
    output_path = shared_root / _safe_relative_path(config["output_path"])
    manifest_path = shared_root / _safe_relative_path(config["manifest_path"])
    raw_bytes = input_path.read_bytes()
    actual_input_hash = sha256_bytes(raw_bytes)
    if actual_input_hash != config["input_sha256"]:
        raise CabBuildError(
            f"CAB input SHA-256 {actual_input_hash} != {config['input_sha256']}"
        )
    raw_rows = json.loads(raw_bytes)
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, dict) for row in raw_rows
    ):
        raise CabBuildError("CAB source must be a JSON list of objects")
    records = build_prompt_rows(raw_rows, config)
    output_bytes = _jsonl_bytes(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_bytes() != output_bytes:
            raise CabBuildError(f"Existing CAB output differs: {output_path}")
    else:
        output_path.write_bytes(output_bytes)

    counts = Counter(
        (record["attribute"], record["explicit_implicit"], record["split"])
        for record in records
    )
    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": sha256_bytes(config_bytes),
        "source_revision": config["source_revision"],
        "input_sha256": actual_input_hash,
        "output_sha256": sha256_bytes(output_bytes),
        "output_rows": len(records),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "output_path": config["output_path"],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise CabBuildError(f"Existing CAB manifest differs: {manifest_path}")
    else:
        manifest_path.write_bytes(manifest_bytes)
    return manifest_path
