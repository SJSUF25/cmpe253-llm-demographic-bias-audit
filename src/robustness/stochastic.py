"""Run frozen stochastic-seed CAB sensitivity cells on the 60-family subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from environment_tools import collect_environment
from inference import runner
from inference.adapter import decode_config, load_model
from robustness.perturb import build

SEEDS = (17, 42, 2026)
RUN_IDS = tuple(
    f"ROB-SEED{seed}-{model}-{condition}"
    for model in ("M1", "M2")
    for condition in ("B0", "F1")
    for seed in SEEDS
)


class StochasticRunError(ValueError):
    """A stochastic sensitivity cell differs from the frozen source or settings."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, run_id: str) -> Path:
    if run_id not in RUN_IDS:
        raise StochasticRunError("Unregistered stochastic sensitivity run")
    root = root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise StochasticRunError("Sensitivity run requires a clean Git checkout")
    runner._check_test_gate(root)
    perturb_path = build(root)
    perturb = json.loads(perturb_path.read_text())
    source_path = root / perturb["output_path"]
    if perturb["output_sha256"] != _sha(source_path):
        raise StochasticRunError("Frozen robustness source changed")
    prompts = [
        row for row in runner._read_jsonl(source_path) if row["kind"] == "original"
    ]
    if len(prompts) != 180 or len({row["prompt_id"] for row in prompts}) != 180:
        raise StochasticRunError("Expected 180 original prompts")
    _, seed_field, model_key, condition_key = run_id.split("-")
    seed = int(seed_field.removeprefix("SEED"))
    settings = decode_config(sensitivity_seed=seed)
    if not settings.do_sample or settings.temperature != 0.7 or settings.top_p != 0.9:
        raise StochasticRunError("Pinned sampling settings changed")
    receipt_path = root / "manifests/robustness" / f"{run_id}.json"
    manifest_path = root / "manifests/inference" / f"{run_id}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        output_path = root / manifest["output_path"]
        if (
            receipt.get("status") != "complete"
            or receipt.get("purpose") != "exploratory_stochastic_robustness"
            or receipt.get("run_id") != run_id
            or receipt.get("code_commit") != commit
            or receipt.get("seed") != seed
            or receipt.get("perturbation_manifest_sha256") != _sha(perturb_path)
            or receipt.get("inference_manifest_sha256") != _sha(manifest_path)
            or receipt.get("output_sha256") != _sha(output_path)
            or manifest.get("status") != "complete"
            or manifest.get("completed_rows") != 180
            or manifest.get("successful_rows") != 180
        ):
            raise StochasticRunError("Completed sensitivity receipt differs")
        return receipt_path
    environment = collect_environment()
    if environment["gpu"].get("name") != "NVIDIA L4":
        raise StochasticRunError("Sensitivity cells require the locked NVIDIA L4")
    condition = {"B0": "baseline", "F1": "mitigation"}[condition_key]
    adapter = load_model(model_key)
    spec = runner.RunSpec(
        run_id=run_id,
        model_id=model_key,
        condition=condition,
        output_path=f"private_outputs/robustness/{run_id}.jsonl",
        confirmatory=True,
    )
    manifest_path = runner.run_inference(
        prompts, adapter, spec, root, settings=settings, checkpoint_interval=20
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("successful_rows") != 180
        or manifest.get("error_summary") != {}
        or manifest.get("seed") != seed
        or manifest.get("environment", {}).get("gpu", {}).get("name") != "NVIDIA L4"
    ):
        raise StochasticRunError("Sensitivity inference did not reconcile")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "exploratory_stochastic_robustness",
        "run_id": run_id,
        "seed": seed,
        "code_commit": commit,
        "perturbation_manifest_sha256": _sha(perturb_path),
        "inference_manifest_sha256": _sha(manifest_path),
        "output_sha256": manifest["output_sha256"],
        "rows": 180,
        "gpu_name": "NVIDIA L4",
    }
    runner._atomic_write(
        receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    )
    return receipt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", choices=RUN_IDS, required=True)
    args = parser.parse_args(argv)
    print(run(args.root, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
