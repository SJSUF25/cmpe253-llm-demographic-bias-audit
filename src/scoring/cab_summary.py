"""Hash-verified aggregate of the eight secondary CAB scoring runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from analysis.descriptive import CAB_IDS


class CabSummaryError(ValueError):
    """A secondary scoring artifact is absent, altered, or incomplete."""


ROW_METRICS = (
    "toxicity_probability",
    "identity_attack_probability",
    "sentiment_positive_probability",
    "sentiment_negative_probability",
    "response_words",
)
PAIR_METRICS = ("semantic_similarity", "response_word_difference")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise CabSummaryError(f"Missing private scoring artifact: {path.name}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows]
    if not values or any(
        not isinstance(value, int | float) or not math.isfinite(value)
        for value in values
    ):
        raise CabSummaryError(f"Invalid secondary metric: {key}")
    return sum(values) / len(values)


def _load_run(root: Path, run_id: str) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = root / "manifests/scoring" / f"{run_id}.json"
    if not manifest_path.is_file():
        raise CabSummaryError(f"Missing scoring manifest: {run_id}")
    manifest = json.loads(manifest_path.read_text())
    rows_path = root / "private_outputs/scored_cab" / f"{run_id}.jsonl"
    pairs_path = root / "private_outputs/scored_cab" / f"{run_id}_pairs.jsonl"
    if not rows_path.is_file() or not pairs_path.is_file():
        raise CabSummaryError(f"Missing scored output: {run_id}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_id
        or manifest.get("split") != "test"
        or manifest.get("missing_rows") != 0
        or manifest.get("scored_output_sha256") != _sha(rows_path)
        or manifest.get("pair_output_sha256") != _sha(pairs_path)
    ):
        raise CabSummaryError(f"Scoring manifest or hashes differ: {run_id}")
    rows, pairs = _jsonl(rows_path), _jsonl(pairs_path)
    explicit = "-EXP-" in run_id
    expected_rows = 640 if explicit else 594
    expected_pairs = 724 if explicit else 677
    if (
        len(rows) != manifest.get("rows")
        or len(rows) != expected_rows
        or len(pairs) != manifest.get("pairs")
        or len(pairs) != expected_pairs
        or len({row.get("prompt_id") for row in rows}) != len(rows)
        or any(
            row.get("run_id") != run_id or row.get("missingness_reason") is not None
            for row in rows
        )
        or any(
            pair.get("run_id") != run_id or pair.get("missingness_reason") is not None
            for pair in pairs
        )
    ):
        raise CabSummaryError(f"Scored inventory differs: {run_id}")
    return manifest, rows, pairs


def summarize(root: Path) -> Path:
    """Write aggregate-only metrics; keep all row-level diagnostics private."""
    root = root.expanduser().resolve()
    hashes = {}
    group_cells = []
    pair_cells = []
    truncation = []
    revisions = set()
    for run_id in CAB_IDS:
        manifest, rows, pairs = _load_run(root, run_id)
        hashes[run_id] = _sha(root / "manifests/scoring" / f"{run_id}.json")
        revisions.add(json.dumps(manifest.get("scorer_revisions"), sort_keys=True))
        model, condition = manifest["model_id"], manifest["condition"]
        mode = manifest["mode"]
        expected_model, expected_condition = run_id.split("-")[2:4]
        expected_mode = "explicit" if "-EXP-" in run_id else "implicit"
        if (
            model != expected_model
            or condition != expected_condition
            or mode != expected_mode
        ):
            raise CabSummaryError(f"Scoring model or condition differs: {run_id}")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["attribute"], row["group"])].append(row)
        if set(grouped) != {
            ("gender", "female"),
            ("gender", "male"),
            ("race", "asian"),
            ("race", "black"),
            ("race", "hispanic"),
            ("race", "white"),
        }:
            raise CabSummaryError(f"Secondary group inventory differs: {run_id}")
        for (attribute, group), cell in sorted(grouped.items()):
            group_cells.append(
                {
                    "model": model,
                    "condition": condition,
                    "mode": mode,
                    "attribute": attribute,
                    "group": group,
                    "rows": len(cell),
                    **{f"mean_{key}": _mean(cell, key) for key in ROW_METRICS},
                }
            )
        grouped_pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for pair in pairs:
            groups = pair.get("groups")
            if not isinstance(groups, list) or len(groups) != 2:
                raise CabSummaryError(f"Malformed pair groups: {run_id}")
            grouped_pairs[(pair["attribute"], "_vs_".join(groups))].append(pair)
        if len(grouped_pairs) != 7:
            raise CabSummaryError(
                f"Secondary pair contrast inventory differs: {run_id}"
            )
        for (attribute, contrast), cell in sorted(grouped_pairs.items()):
            pair_cells.append(
                {
                    "model": model,
                    "condition": condition,
                    "mode": mode,
                    "attribute": attribute,
                    "contrast": contrast,
                    "pairs": len(cell),
                    **{f"mean_{key}": _mean(cell, key) for key in PAIR_METRICS},
                }
            )
        truncation.append(
            {
                "model": model,
                "condition": condition,
                "mode": mode,
                "rows": len(rows),
                "truncated_scorer_inputs": manifest["truncated_scorer_inputs"],
            }
        )
    if len(revisions) != 1:
        raise CabSummaryError("Scorer revisions differ across runs")
    result = {
        "schema_version": 1,
        "status": "descriptive_secondary_uncalibrated",
        "source_scoring_manifest_sha256": hashes,
        "group_cells": group_cells,
        "pair_cells": pair_cells,
        "truncation": truncation,
        "limitations": [
            "Classifier probabilities have no decision threshold or human calibration.",
            "Semantic similarity measures output text resemblance, not fairness.",
            "All metrics in this file are secondary descriptive diagnostics.",
        ],
    }
    output = root / "aggregate_results/cab_secondary_summary.json"
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text() != content:
        raise CabSummaryError("Existing secondary aggregate differs")
    output.write_text(content)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(summarize(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
