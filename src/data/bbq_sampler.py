"""Deterministic BBQ sampling over rows enriched with audited metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from typing import Any

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
