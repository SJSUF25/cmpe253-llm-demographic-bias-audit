"""Freeze the preregistered 60-item BBQ explanation sample from source inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from inference import bbq_runner, runner

STRATA = (
    ("ambig", "neg"),
    ("ambig", "nonneg"),
    ("disambig", "neg"),
    ("disambig", "nonneg"),
)
PER_STRATUM = 5


class ExplanationSampleError(ValueError):
    """The frozen BBQ inventory cannot supply the registered explanation sample."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_items(items: list[dict]) -> list[dict]:
    """Select five source items per category/context/polarity stratum."""
    if len(items) != 1500 or len({row["item_id"] for row in items}) != 1500:
        raise ExplanationSampleError("Expected 1,500 distinct frozen BBQ items")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in items:
        key = row["category"], row["context_condition"], row["question_polarity"]
        grouped[key].append(row)
    categories = sorted({row["category"] for row in items})
    if len(categories) != 3 or set(grouped) != {
        (category, context, polarity)
        for category in categories
        for context, polarity in STRATA
    }:
        raise ExplanationSampleError("BBQ explanation strata differ")
    selected = []
    for category in categories:
        for context, polarity in STRATA:
            eligible = grouped[category, context, polarity]
            if len(eligible) < PER_STRATUM:
                raise ExplanationSampleError("Explanation stratum has fewer than five")
            eligible = sorted(
                eligible,
                key=lambda row: (
                    _sha(f"253|explain60|{category}|{row['example_id']}".encode()),
                    str(row["example_id"]),
                ),
            )
            selected.extend(
                {
                    "item_id": row["item_id"],
                    "example_id": str(row["example_id"]),
                    "category": category,
                    "context_condition": context,
                    "question_polarity": polarity,
                    "template_id": row["template_id"],
                }
                for row in eligible[:PER_STRATUM]
            )
    if len(selected) != 60 or len({row["item_id"] for row in selected}) != 60:
        raise ExplanationSampleError("Explanation sample is not 60 unique items")
    return selected


def freeze(root: Path) -> Path:
    root = root.expanduser().resolve()
    items, _, build_hash = bbq_runner._inputs(root)
    selected = select_items(items)
    payload = {
        "schema_version": 1,
        "selection": "sha256_253_explain60_five_per_category_context_polarity",
        "bbq_build_manifest_sha256": build_hash,
        "items": selected,
        "status": "frozen_before_attribution",
    }
    path = root / "manifests/explain/explain60.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ExplanationSampleError("Existing explanation sample differs")
    else:
        runner._atomic_write(path, content)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(freeze(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
