"""Pair automated refusal gaps for the frozen CAB robustness perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from analysis.descriptive import CONTRASTS, clustered_interval
from environment_tools import comparable_environment
from inference import runner
from robustness.perturb import build
from robustness.runner import KINDS, RUN_IDS, RobustnessRunError
from scoring import cab


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_gaps(rows: list[dict]) -> list[dict]:
    """Aggregate signed original and perturbed gaps by scenario family."""
    by_family: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = row["attribute"], row["scenario_id"]
        if row["group"] in by_family[key]:
            raise RobustnessRunError("Duplicate robustness family/group")
        by_family[key][row["group"]] = row
    result = []
    for attribute, contrasts in CONTRASTS.items():
        families = [
            (scenario_id, groups)
            for (attr, scenario_id), groups in sorted(by_family.items())
            if attr == attribute
        ]
        if len(families) != 30:
            raise RobustnessRunError("Robustness attribute has fewer than 30 families")
        for first, second in contrasts:
            paired = []
            for scenario_id, groups in families:
                if first not in groups or second not in groups:
                    raise RobustnessRunError("Missing robustness contrast group")
                paired.append(
                    {
                        "scenario_id": scenario_id,
                        "original_gap": int(groups[first]["original_refusal"])
                        - int(groups[second]["original_refusal"]),
                        "perturbed_gap": int(groups[first]["perturbed_refusal"])
                        - int(groups[second]["perturbed_refusal"]),
                    }
                )

            def change(sample: list[dict]) -> float:
                return sum(
                    row["perturbed_gap"] - row["original_gap"] for row in sample
                ) / len(sample)

            result.append(
                {
                    "attribute": attribute,
                    "contrast": f"{first}_minus_{second}",
                    "families": len(paired),
                    "original_refusal_gap": sum(row["original_gap"] for row in paired)
                    / len(paired),
                    "perturbed_refusal_gap": sum(row["perturbed_gap"] for row in paired)
                    / len(paired),
                    "gap_change": change(paired),
                    "gap_change_ci95": clustered_interval(
                        paired, "scenario_id", change
                    ),
                }
            )
    return result


def report(root: Path) -> Path:
    root = root.expanduser().resolve()
    runner._check_test_gate(root)
    perturb_path = build(root)
    perturb = json.loads(perturb_path.read_text())
    prompts = runner._read_jsonl(root / perturb["output_path"])
    source_by_kind = {
        kind: {row["prompt_id"]: row for row in prompts if row["kind"] == name}
        for kind, name in KINDS.items()
    }
    original_sources = {}
    original_hashes = {}
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            full_id = f"CAB-EXP-{model}-{condition}"
            source_prompts, outputs, source = cab._inputs(root, full_id, "test")
            by_prompt = {row["prompt_id"]: row for row in source_prompts}
            if len(outputs) != len(by_prompt):
                raise RobustnessRunError("Original CAB output count differs")
            original_sources[model, condition] = {
                row["prompt_id"]: cab.refusal_rule(row["raw_output"])[0]
                for row in outputs
                if row["status"] == "ok" and row["prompt_id"] in by_prompt
            }
            if len(original_sources[model, condition]) != len(by_prompt):
                raise RobustnessRunError("Original CAB has failed rows")
            original_hashes[full_id] = source["inference_manifest_sha256"]
    cells = []
    receipt_hashes = {}
    commits = set()
    environments: dict[str, set[str]] = defaultdict(set)
    for run_id in RUN_IDS:
        _, kind, model, condition = run_id.split("-")
        receipt_path = root / "manifests/robustness" / f"{run_id}.json"
        manifest_path = root / "manifests/inference" / f"{run_id}.json"
        if not receipt_path.is_file() or not manifest_path.is_file():
            raise RobustnessRunError(f"Missing robustness run: {run_id}")
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        output_path = root / manifest["output_path"]
        if (
            receipt.get("run_id") != run_id
            or receipt.get("status") != "complete"
            or receipt.get("perturbation_manifest_sha256") != _sha(perturb_path)
            or receipt.get("inference_manifest_sha256") != _sha(manifest_path)
            or manifest.get("status") != "complete"
            or manifest.get("completed_rows") != 180
            or manifest.get("successful_rows") != 180
            or manifest.get("output_sha256") != _sha(output_path)
        ):
            raise RobustnessRunError(f"Robustness receipt differs: {run_id}")
        receipt_hashes[run_id] = _sha(receipt_path)
        commits.add(manifest["code_commit"])
        environments[model].add(
            json.dumps(comparable_environment(manifest["environment"]), sort_keys=True)
        )
        outputs = runner._read_jsonl(output_path)
        by_id = source_by_kind[kind]
        if len(outputs) != 180 or {row["prompt_id"] for row in outputs} != set(by_id):
            raise RobustnessRunError("Robustness output inventory differs")
        joined = []
        for row in outputs:
            prompt = by_id[row["prompt_id"]]
            if row["status"] != "ok" or row["prompt_sha256"] != prompt["prompt_sha256"]:
                raise RobustnessRunError("Robustness output row differs")
            original = original_sources[model, condition][prompt["source_prompt_id"]]
            joined.append(
                {
                    "scenario_id": prompt["scenario_id"],
                    "attribute": prompt["attribute"],
                    "group": prompt["group"],
                    "original_refusal": original,
                    "perturbed_refusal": cab.refusal_rule(row["raw_output"])[0],
                }
            )
        for cell in paired_gaps(joined):
            cells.append({"model": model, "condition": condition, "kind": kind, **cell})
    if len(commits) != 1 or any(len(values) != 1 for values in environments.values()):
        raise RobustnessRunError("Robustness cells used different code or environments")
    payload = {
        "schema_version": 1,
        "status": "descriptive_automated_refusal_only",
        "perturbation_manifest_sha256": _sha(perturb_path),
        "original_cab_manifest_sha256": original_hashes,
        "robustness_receipt_sha256": receipt_hashes,
        "cells": cells,
        "limitations": [
            "Anchored refusal is uncalibrated against human labels.",
            "Harmful compliance and unsupported assumptions require "
            "separate response labels.",
            "Formal/conversational paraphrases, identity-position changes, "
            "and stochastic decoding are pending.",
        ],
    }
    output = root / "aggregate_results/robustness_refusal.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        if output.read_bytes() != content:
            raise RobustnessRunError("Existing robustness summary differs")
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
