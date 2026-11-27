"""Resumable, source-verified BBQ feature-ablation runs on the frozen 60 items."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from environment_tools import collect_environment, comparable_environment
from explain.ablation import explain_item
from explain.sample import freeze
from inference import bbq_runner, runner
from inference.adapter import load_model

RUN_IDS = tuple(
    f"EXPLAIN-{model}-{condition}"
    for model in ("M1", "M2")
    for condition in ("B0", "F1")
)


class ExplanationRunError(ValueError):
    """The explanation run is incomplete or differs from the frozen inputs."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def _inputs(
    root: Path, run_id: str
) -> tuple[list[tuple[dict, list[dict], dict]], dict]:
    if run_id not in RUN_IDS:
        raise ExplanationRunError("Unregistered explanation run")
    sample_path = freeze(root)
    sample = json.loads(sample_path.read_text())
    items, permutations, build_hash = bbq_runner._inputs(root)
    if sample["bbq_build_manifest_sha256"] != build_hash:
        raise ExplanationRunError("Explanation sample source changed")
    model, condition = run_id.split("-")[1:]
    full_id = f"BBQ-{model}-{condition}"
    manifest_path = root / "manifests/bbq" / f"{full_id}.json"
    manifest = json.loads(manifest_path.read_text())
    output = root / manifest["output_path"]
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_rows") != 1500
        or manifest.get("output_sha256") != _sha(output.read_bytes())
    ):
        raise ExplanationRunError("Full BBQ source run is incomplete or changed")
    scored = runner._read_jsonl(output)
    if len(scored) != 1500 or len({row["item_id"] for row in scored}) != 1500:
        raise ExplanationRunError("Full BBQ scored inventory differs")
    item_by_id = {row["item_id"]: row for row in items}
    scored_by_id = {row["item_id"]: row for row in scored}
    variants_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in permutations:
        variants_by_id[row["item_id"]].append(row)
    selected = []
    for entry in sample["items"]:
        item_id = entry["item_id"]
        item = item_by_id.get(item_id)
        result = scored_by_id.get(item_id)
        variants = variants_by_id.get(item_id)
        if item is None or result is None or variants is None or len(variants) != 3:
            raise ExplanationRunError("Selected explanation item is missing")
        if (
            item["category"] != entry["category"]
            or item["template_id"] != entry["template_id"]
            or result["category"] != item["category"]
            or result["template_id"] != item["template_id"]
            or result["context_condition"] != item["context_condition"]
            or result["question_polarity"] != item["question_polarity"]
            or result["model_id"] != model
            or result["condition"] != {"B0": "baseline", "F1": "mitigation"}[condition]
        ):
            raise ExplanationRunError("Full BBQ model or condition differs")
        selected.append((item, variants, result))
    if len(selected) != 60 or len({item["item_id"] for item, _, _ in selected}) != 60:
        raise ExplanationRunError("Explanation sample is not 60 distinct items")
    source = {
        "run_id": run_id,
        "sample_sha256": _sha(sample_path.read_bytes()),
        "full_manifest_sha256": _sha(manifest_path.read_bytes()),
        "full_output_sha256": _sha(output.read_bytes()),
    }
    return selected, source


def run(root: Path, run_id: str) -> Path:
    root = root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise ExplanationRunError("Explanation run requires a clean Git checkout")
    runner._check_test_gate(root)
    bbq_runner.report(root)
    selected, source = _inputs(root, run_id)
    environment = collect_environment()
    if environment["gpu"].get("name") != "NVIDIA L4":
        raise ExplanationRunError("Explanation runs require the locked NVIDIA L4")
    model_key, condition_key = run_id.split("-")[1:]
    condition = {"B0": "baseline", "F1": "mitigation"}[condition_key]
    part_dir = root / "private_outputs/explain" / run_id / "items"
    manifest_path = root / "manifests/explain" / f"{run_id}.json"
    context = {
        **source,
        "code_commit": commit,
        "environment": comparable_environment(environment),
    }
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("status") != "complete" or prior.get("context") != context:
            raise ExplanationRunError("Completed explanation manifest differs")
        if len(prior.get("parts", [])) != 60:
            raise ExplanationRunError("Completed explanation part count differs")
        for index, (item, _, _) in enumerate(selected):
            path = part_dir / f"item-{index:03}.json"
            if (
                not path.is_file()
                or prior["parts"][index]["item_id"] != item["item_id"]
                or prior["parts"][index]["sha256"] != _sha(path.read_bytes())
            ):
                raise ExplanationRunError("Completed explanation part changed")
        return manifest_path
    model = None
    parts = []
    for index, (item, variants, frozen_result) in enumerate(selected):
        path = part_dir / f"item-{index:03}.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            if (
                payload.get("context") != context
                or payload.get("item_id") != item["item_id"]
            ):
                raise ExplanationRunError("Existing explanation item differs")
        else:
            if model is None:
                model = load_model(model_key)
            measured = explain_item(
                item,
                variants,
                frozen_result,
                lambda prompt, answer, loaded_model=model: (
                    loaded_model.conditional_log_likelihood(prompt, answer, condition)
                ),
            )
            payload = {
                **measured,
                "category": item["category"],
                "template_id": item["template_id"],
                "context_condition": item["context_condition"],
                "question_polarity": item["question_polarity"],
                "context": context,
            }
            runner._atomic_write(path, _json_bytes(payload))
        parts.append(
            {
                "item_id": item["item_id"],
                "path": str(path.relative_to(root)),
                "sha256": _sha(path.read_bytes()),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"{run_id}: {index + 1}/60 explanation items", flush=True)
    result = {
        "schema_version": 1,
        "status": "complete",
        "context": context,
        "captured_environment": environment,
        "items": 60,
        "parts": parts,
        "method_limitations": [
            "Exact target-group matching can be absent for name/proxy items.",
            "Section ablations change input wording. They are behavioral "
            "explanations, not model reasoning traces.",
            "Answer-order stability is measured from per-order effects; "
            "token-level faithfulness and paraphrase stability are pending.",
        ],
    }
    runner._atomic_write(manifest_path, _json_bytes(result))
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", choices=RUN_IDS, required=True)
    args = parser.parse_args(argv)
    print(run(args.root, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
