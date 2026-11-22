"""Deterministic BBQ sampling over rows enriched with audited metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.download import _validate_bytes, sha256_bytes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/bbq_build.json"

STRATA = (
    ("ambiguous", "negative"),
    ("ambiguous", "non-negative"),
    ("disambiguated", "negative"),
    ("disambiguated", "non-negative"),
)


class BBQSamplingError(ValueError):
    """The source or requested quota violates the frozen sampling contract."""


def _hash(category: str, example_id: object) -> str:
    return hashlib.sha256(f"253|{category}|{example_id}".encode()).hexdigest()


def _stratum(row: dict[str, Any]) -> tuple[str, str]:
    context = {"ambig": "ambiguous", "disambig": "disambiguated"}.get(
        row.get("context_condition")
    )
    polarity = {"neg": "negative", "nonneg": "non-negative"}.get(
        row.get("question_polarity")
    )
    if context is None or polarity is None:
        raise BBQSamplingError("Unrecognized BBQ context or question polarity")
    return context, polarity


def _deduplicate(rows: Iterable[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("category") != category:
            raise BBQSamplingError("Category mismatch")
        for field in ("example_id", "template_id", "target_group"):
            if field not in row or row[field] in (None, ""):
                raise BBQSamplingError(f"Missing enriched BBQ field: {field}")
        key = str(row["example_id"])
        if key in by_id and by_id[key] != row:
            raise BBQSamplingError(f"Conflicting duplicate example_id: {key}")
        by_id[key] = row
    return list(by_id.values())


def _round_robin(rows: list[dict[str, Any]], category: str) -> deque[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["template_id"]), str(row["target_group"]))].append(row)
    queues = {
        key: deque(
            sorted(
                bucket,
                key=lambda row: (
                    _hash(category, row["example_id"]),
                    str(row["example_id"]),
                ),
            )
        )
        for key, bucket in buckets.items()
    }
    result: deque[dict[str, Any]] = deque()
    while any(queues.values()):
        for key in sorted(queues):
            if queues[key]:
                result.append(queues[key].popleft())
    return result


def select_category(
    rows: Iterable[dict[str, Any]], category: str, *, target: int = 500
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a category with fixed stratum quotas and deterministic redistribution."""
    if target < 1 or target % 4:
        raise BBQSamplingError("Target must be a positive multiple of four")
    unique = _deduplicate(rows, category)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        by_stratum[_stratum(row)].append(row)
    queues = {key: _round_robin(by_stratum[key], category) for key in STRATA}
    quota = target // 4
    chosen: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for key in STRATA:
        for _ in range(min(quota, len(queues[key]))):
            chosen.append(queues[key].popleft())
            counts[key] += 1
    deficit = target - len(chosen)
    while deficit:
        progress = False
        for key in STRATA:
            if queues[key] and deficit:
                chosen.append(queues[key].popleft())
                counts[key] += 1
                deficit -= 1
                progress = True
        if not progress:
            raise BBQSamplingError(
                f"Only {len(chosen)} unique eligible rows; need {target}"
            )
    selected_ids = [str(row["example_id"]) for row in chosen]
    report = {
        "category": category,
        "source_unique_rows": len(unique),
        "selected_count": len(chosen),
        "requested_per_stratum": quota,
        "realized_by_stratum": {"|".join(key): counts[key] for key in STRATA},
        "realized_by_template_target": dict(
            sorted(
                Counter(
                    f"{row['template_id']}|{row['target_group']}" for row in chosen
                ).items()
            )
        ),
        "selected_ids": selected_ids,
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(selected_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return chosen, report


def cyclic_option_orders(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all cyclic answer orders with reversible source-index maps."""
    answers = [row[f"ans{index}"] for index in range(3)]
    label = row["label"]
    if not isinstance(label, int) or label not in (0, 1, 2):
        raise BBQSamplingError("Invalid original answer label")
    variants = []
    for shift in range(3):
        source_indices = [(shift + index) % 3 for index in range(3)]
        variants.append(
            {
                "order": shift,
                "answers": [answers[index] for index in source_indices],
                "source_indices": source_indices,
                "correct_index": source_indices.index(label),
            }
        )
    return variants


def enrich_rows(
    rows: list[dict[str, Any]], metadata: list[dict[str, str]], category: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join official supplemental metadata and exclude missing target locations."""
    matches: dict[tuple[str, str], dict[str, str]] = {}
    collapsed = 0
    for entry in metadata:
        if entry["category"] != category:
            continue
        key = (entry["category"], entry["example_id"])
        if key in matches:
            first = matches[key]
            join_fields = ("question_index", "target_loc", "label_type")
            if any(first[field] != entry[field] for field in join_fields):
                raise BBQSamplingError(f"Conflicting supplemental key {key}")
            collapsed += 1
            continue
        matches[key] = entry
    enriched = []
    missing_location = 0
    missing_group = 0
    for row in rows:
        entry = matches.get((category, str(row["example_id"])))
        if entry is None or entry["question_index"] != str(row["question_index"]):
            raise BBQSamplingError("Missing or mismatched supplemental metadata")
        location = entry["target_loc"]
        if location == "NA":
            missing_location += 1
            continue
        if location not in {"0", "1", "2"}:
            raise BBQSamplingError(f"Invalid target location: {location}")
        target = row["answer_info"][f"ans{location}"][1]
        if not target:
            missing_group += 1
            continue
        if target == "unknown":
            raise BBQSamplingError("Target location resolves to unknown answer")
        enriched.append(
            {
                **row,
                "template_id": entry["question_index"],
                "target_group": target,
                "target_loc": int(location),
            }
        )
    return enriched, {
        "excluded_missing_target_loc": missing_location,
        "excluded_missing_target_group": missing_group,
        "collapsed_metadata_duplicates": collapsed,
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise BBQSamplingError(f"Existing BBQ artifact differs: {path}")
    else:
        path.write_bytes(content)


def build_bbq(shared_root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Verify pinned inputs and write deterministic private BBQ sample artifacts."""
    shared_root = shared_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1 or config.get("target_per_category") != 500:
        raise BBQSamplingError("Unsupported BBQ build configuration")
    sources = json.loads((ROOT / config["source_config"]).read_text(encoding="utf-8"))
    dataset = next(source for source in sources["datasets"] if source["key"] == "D2")
    if dataset["revision"] != config["source_revision"]:
        raise BBQSamplingError("BBQ source revision mismatch")
    raw_root = shared_root / "raw_immutable" / "D2" / dataset["revision"]
    for source in [*dataset["files"], dataset["supplemental"]]:
        _validate_bytes((raw_root / source["path"]).read_bytes(), dataset, source)

    with (raw_root / dataset["supplemental"]["path"]).open(
        newline="", encoding="utf-8"
    ) as stream:
        metadata = list(csv.DictReader(stream))
    items: list[dict[str, Any]] = []
    permutations: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for category in config["categories"]:
        rows = [
            json.loads(line)
            for line in (raw_root / f"{category}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        enriched, join_report = enrich_rows(rows, metadata, category)
        selected, report = select_category(
            enriched, category, target=config["target_per_category"]
        )
        report.update(join_report)
        report["permutation_count"] = len(selected) * 3
        reports[category] = report
        for row in selected:
            item_id = f"{category}|{row['example_id']}"
            items.append({"item_id": item_id, **row})
            for variant in cyclic_option_orders(row):
                permutations.append({"item_id": item_id, **variant})
    if (
        len(items) != 1500
        or len(permutations) != 4500
        or len({r["item_id"] for r in items}) != 1500
    ):
        raise BBQSamplingError("Unexpected BBQ sample or permutation count")
    item_bytes, permutation_bytes = _jsonl_bytes(items), _jsonl_bytes(permutations)
    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": sha256_bytes(config_bytes),
        "source_revision": dataset["revision"],
        "items_path": config["items_path"],
        "items_count": len(items),
        "items_sha256": sha256_bytes(item_bytes),
        "permutations_path": config["permutations_path"],
        "permutations_count": len(permutations),
        "permutations_sha256": sha256_bytes(permutation_bytes),
        "allocation": reports,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_immutable(shared_root / config["items_path"], item_bytes)
    _write_immutable(shared_root / config["permutations_path"], permutation_bytes)
    manifest_path = shared_root / config["manifest_path"]
    _write_immutable(manifest_path, manifest_bytes)
    return manifest_path
