"""Aggregate all preregistered BBQ ablation items without selecting examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from analysis.descriptive import clustered_interval
from explain.ablation import FEATURES
from explain.runner import RUN_IDS, ExplanationRunError
from inference import bbq_runner, runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        for effect in row["effects"]:
            if effect["feature"] not in FEATURES:
                raise ExplanationRunError("Unknown explanation feature")
            grouped[
                row["model"], row["condition"], row["category"], effect["feature"]
            ].append(
                {
                    "template_id": row["template_id"],
                    "margin_drop": effect["margin_drop"],
                }
            )
    if len(grouped) != 2 * 2 * 3 * len(FEATURES):
        raise ExplanationRunError("Explanation aggregate cell inventory differs")
    result = []
    for (model, condition, category, feature), cell in sorted(grouped.items()):
        if len(cell) != 20:
            raise ExplanationRunError("Explanation category must contain 20 items")
        measured = [row for row in cell if row["margin_drop"] is not None]

        def mean(sample: list[dict]) -> float:
            return sum(row["margin_drop"] for row in sample) / len(sample)

        result.append(
            {
                "model": model,
                "condition": condition,
                "category": category,
                "feature": feature,
                "items": len(cell),
                "measured_items": len(measured),
                "not_present_items": len(cell) - len(measured),
                "mean_margin_drop": mean(measured) if measured else None,
                "ci95": (
                    clustered_interval(measured, "template_id", mean)
                    if measured
                    else {"lower": None, "upper": None, "valid_draws": 0}
                ),
            }
        )
    return result


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, preserving the original feature order."""
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = (position + end - 1) / 2
        for index in ordered[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    x, y = _ranks(left), _ranks(right)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _top_two(features: list[str], values: list[float]) -> set[str]:
    return set(
        sorted(
            features, key=lambda feature: (-values[features.index(feature)], feature)
        )[:2]
    )


def summarize_order_stability(rows: list[dict]) -> list[dict]:
    """Compare the same feature ablations under the three cyclic answer orders."""
    grouped = defaultdict(list)
    for row in rows:
        by_order = row.get("order_effects")
        if not isinstance(by_order, dict) or set(by_order) != {"0", "1", "2"}:
            raise ExplanationRunError("Missing per-order explanation effects")
        for first, second in combinations(("0", "1", "2"), 2):
            left, right = by_order[first], by_order[second]
            common = [
                feature
                for feature in FEATURES
                if left.get(feature) is not None and right.get(feature) is not None
            ]
            if len(common) < 3:
                correlation = overlap = None
            else:
                left_values = [left[feature] for feature in common]
                right_values = [right[feature] for feature in common]
                correlation = _spearman(left_values, right_values)
                overlap = (
                    len(_top_two(common, left_values) & _top_two(common, right_values))
                    / 2
                )
            grouped[
                row["model"], row["condition"], row["category"], first, second
            ].append(
                {
                    "template_id": row["template_id"],
                    "spearman": correlation,
                    "top2_overlap": overlap,
                }
            )
    if len(grouped) != 2 * 2 * 3 * 3:
        raise ExplanationRunError("Order stability cell inventory differs")
    cells = []
    for (model, condition, category, first, second), group in sorted(grouped.items()):
        if len(group) != 20:
            raise ExplanationRunError("Order stability category must contain 20 items")
        cell = {
            "model": model,
            "condition": condition,
            "category": category,
            "order_pair": [int(first), int(second)],
            "items": len(group),
        }
        for metric in ("spearman", "top2_overlap"):
            measured = [entry for entry in group if entry[metric] is not None]

            def mean(sample: list[dict], key: str = metric) -> float:
                return sum(entry[key] for entry in sample) / len(sample)

            cell[metric] = {
                "measured_items": len(measured),
                "missing_items": len(group) - len(measured),
                "mean": mean(measured) if measured else None,
                "ci95": (
                    clustered_interval(measured, "template_id", mean)
                    if measured
                    else {"lower": None, "upper": None, "valid_draws": 0}
                ),
            }
        cells.append(cell)
    return cells


def report(root: Path) -> Path:
    root = root.expanduser().resolve()
    runner._check_test_gate(root)
    bbq_runner.report(root)
    sample_path = root / "manifests/explain/explain60.json"
    sample = json.loads(sample_path.read_text())
    sample_ids = [entry["item_id"] for entry in sample["items"]]
    _, _, build_hash = bbq_runner._inputs(root)
    if (
        len(sample_ids) != 60
        or len(set(sample_ids)) != 60
        or sample.get("bbq_build_manifest_sha256") != build_hash
    ):
        raise ExplanationRunError("Frozen explanation sample differs")
    all_rows = []
    hashes = {}
    sample_hashes = set()
    for run_id in RUN_IDS:
        manifest_path = root / "manifests/explain" / f"{run_id}.json"
        if not manifest_path.is_file():
            raise ExplanationRunError(f"Missing explanation manifest: {run_id}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("items") != 60
            or len(manifest.get("parts", [])) != 60
            or manifest.get("context", {}).get("sample_sha256") != _sha(sample_path)
            or manifest.get("context", {}).get("full_manifest_sha256")
            != _sha(
                root
                / "manifests/bbq"
                / f"BBQ-{run_id.split('-')[1]}-{run_id.split('-')[2]}.json"
            )
        ):
            raise ExplanationRunError(f"Incomplete explanation run: {run_id}")
        hashes[run_id] = _sha(manifest_path)
        sample_hashes.add(manifest["context"]["sample_sha256"])
        seen = set()
        for index, part in enumerate(manifest["parts"]):
            path = root / part["path"]
            expected = (
                root
                / "private_outputs/explain"
                / run_id
                / "items"
                / f"item-{index:03}.json"
            )
            if path != expected or not path.is_file() or _sha(path) != part["sha256"]:
                raise ExplanationRunError("Explanation item hash differs")
            row = json.loads(path.read_text())
            if (
                row["context"] != manifest["context"]
                or row["item_id"] != part["item_id"]
                or row["item_id"] != sample_ids[index]
                or row["item_id"] in seen
                or len(row["effects"]) != len(FEATURES)
            ):
                raise ExplanationRunError("Explanation item metadata differs")
            seen.add(row["item_id"])
            all_rows.append(
                {
                    **row,
                    "model": run_id.split("-")[1],
                    "condition": run_id.split("-")[2],
                }
            )
    if len(sample_hashes) != 1:
        raise ExplanationRunError("Explanation cells used different 60-item samples")
    payload = {
        "schema_version": 1,
        "status": "descriptive_section_ablation_and_order_stability",
        "source_manifest_sha256": hashes,
        "rows": len(all_rows),
        "cells": summarize(all_rows),
        "answer_order_stability": summarize_order_stability(all_rows),
        "limitations": [
            "Exact target-group matching may be absent for name/proxy items.",
            "Section ablations measure answer-margin sensitivity, "
            "not causal reasoning traces.",
            "Answer-order stability compares three cyclic option orders only.",
            "Token-level faithfulness and paraphrase stability remain pending.",
        ],
    }
    output = root / "aggregate_results/explain_summary.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        if output.read_bytes() != content:
            raise ExplanationRunError("Existing explanation summary differs")
    else:
        runner._atomic_write(output, content)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(report(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
