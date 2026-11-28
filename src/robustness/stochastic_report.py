"""Summarize three-seed robustness without treating sampled outputs as new families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from environment_tools import comparable_environment
from inference import runner
from robustness.perturb import build
from robustness.stochastic import RUN_IDS, StochasticRunError
from scoring import cab


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> list[dict]:
    """Compute descriptive seed variability per model/condition/attribute."""
    cells: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        cells[row["model"], row["condition"], row["attribute"], row["seed"]].append(row)
    if len(cells) != 2 * 2 * 2 * 3:
        raise StochasticRunError("Stochastic report has missing seed cells")
    result = []
    for (model, condition, attribute, seed), cell in sorted(cells.items()):
        expected = 60 if attribute == "gender" else 120
        if len(cell) != expected or len({row["prompt_id"] for row in cell}) != expected:
            raise StochasticRunError("Stochastic cell prompt inventory differs")
        result.append(
            {
                "model": model,
                "condition": condition,
                "attribute": attribute,
                "seed": seed,
                "prompts": len(cell),
                "scenario_families": len({row["scenario_id"] for row in cell}),
                "anchored_refusal_rate": sum(row["sampled_refusal"] for row in cell)
                / len(cell),
                "refusal_disagreement_with_greedy": sum(
                    row["sampled_refusal"] != row["greedy_refusal"] for row in cell
                )
                / len(cell),
                "exact_output_match_with_greedy": sum(
                    row["sampled_output"] == row["greedy_output"] for row in cell
                )
                / len(cell),
            }
        )
    return result


def report(root: Path) -> Path:
    root = root.expanduser().resolve()
    runner._check_test_gate(root)
    perturb_path = build(root)
    perturb = json.loads(perturb_path.read_text())
    source_path = root / perturb["output_path"]
    if perturb["output_sha256"] != _sha(source_path):
        raise StochasticRunError("Frozen robustness prompts changed")
    originals = {
        row["prompt_id"]: row
        for row in runner._read_jsonl(source_path)
        if row["kind"] == "original"
    }
    if len(originals) != 180:
        raise StochasticRunError("Original robustness prompt inventory differs")
    greedy = {}
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            _, outputs, _ = cab._inputs(root, f"CAB-EXP-{model}-{condition}", "test")
            greedy[model, condition] = {row["prompt_id"]: row for row in outputs}
    all_rows, hashes, commits = [], {}, set()
    environments: dict[str, set[str]] = defaultdict(set)
    for run_id in RUN_IDS:
        _, seed_field, model, condition = run_id.split("-")
        seed = int(seed_field.removeprefix("SEED"))
        receipt_path = root / "manifests/robustness" / f"{run_id}.json"
        manifest_path = root / "manifests/inference" / f"{run_id}.json"
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        output_path = root / manifest["output_path"]
        if (
            receipt.get("status") != "complete"
            or receipt.get("purpose") != "exploratory_stochastic_robustness"
            or receipt.get("run_id") != run_id
            or receipt.get("seed") != seed
            or receipt.get("perturbation_manifest_sha256") != _sha(perturb_path)
            or receipt.get("inference_manifest_sha256") != _sha(manifest_path)
            or receipt.get("output_sha256") != _sha(output_path)
            or manifest.get("status") != "complete"
            or manifest.get("completed_rows") != 180
            or manifest.get("successful_rows") != 180
            or manifest.get("seed") != seed
        ):
            raise StochasticRunError(f"Stochastic receipt differs: {run_id}")
        hashes[run_id] = _sha(receipt_path)
        commits.add(manifest["code_commit"])
        environments[model].add(
            json.dumps(comparable_environment(manifest["environment"]), sort_keys=True)
        )
        outputs = runner._read_jsonl(output_path)
        if len(outputs) != 180 or {row["prompt_id"] for row in outputs} != set(
            originals
        ):
            raise StochasticRunError(f"Stochastic output inventory differs: {run_id}")
        for row in outputs:
            prompt = originals[row["prompt_id"]]
            earlier = greedy[model, condition][prompt["source_prompt_id"]]
            if (
                row["status"] != "ok"
                or earlier["status"] != "ok"
                or row["prompt_sha256"] != prompt["prompt_sha256"]
                or row["run_id"] != run_id
            ):
                raise StochasticRunError("Stochastic row source differs")
            all_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "seed": seed,
                    "prompt_id": row["prompt_id"],
                    "scenario_id": prompt["scenario_id"],
                    "attribute": prompt["attribute"],
                    "sampled_refusal": cab.refusal_rule(row["raw_output"])[0],
                    "greedy_refusal": cab.refusal_rule(earlier["raw_output"])[0],
                    "sampled_output": row["raw_output"],
                    "greedy_output": earlier["raw_output"],
                }
            )
    if len(commits) != 1 or any(len(values) != 1 for values in environments.values()):
        raise StochasticRunError("Stochastic cells changed code or model environment")
    result = {
        "schema_version": 1,
        "status": "descriptive_stochastic_robustness_proxy",
        "purpose": "exploratory_h4_sensitivity_not_confirmatory_family",
        "source_receipt_sha256": hashes,
        "cells": summarize(all_rows),
        "limitations": [
            "Anchored refusal is an uncalibrated proxy until human validation.",
            "Each seed repeats the same 60 scenario families; "
            "seeds are not independent examples.",
            "Exact text agreement is a strict stability diagnostic, "
            "not semantic equivalence.",
        ],
    }
    output = root / "aggregate_results/robustness_stochastic.json"
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if output.is_file():
        if output.read_bytes() != content:
            raise StochasticRunError("Existing stochastic summary differs")
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
