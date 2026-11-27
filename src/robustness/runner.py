"""Run frozen CAB whitespace/adversarial cells with the standard checkpoint engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from environment_tools import collect_environment
from inference import runner
from inference.adapter import load_model
from robustness.perturb import build

RUN_IDS = tuple(
    f"ROB-{kind}-{model}-{condition}"
    for model in ("M1", "M2")
    for condition in ("B0", "F1")
    for kind in ("SPACE", "ADV")
)
KINDS = {"SPACE": "whitespace", "ADV": "adversarial"}


class RobustnessRunError(ValueError):
    """A source, model, or completed robustness cell differs from the contract."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, run_id: str) -> Path:
    if run_id not in RUN_IDS:
        raise RobustnessRunError("Unregistered robustness run")
    root = root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise RobustnessRunError("Robustness run requires a clean Git checkout")
    runner._check_test_gate(root)
    perturb_path = build(root)
    perturb = json.loads(perturb_path.read_text())
    source_path = root / perturb["output_path"]
    if perturb["output_sha256"] != _sha(source_path):
        raise RobustnessRunError("Frozen robustness prompts changed")
    _, kind, model_key, condition_key = run_id.split("-")
    prompts = [
        row for row in runner._read_jsonl(source_path) if row["kind"] == KINDS[kind]
    ]
    if len(prompts) != 180 or len({row["prompt_id"] for row in prompts}) != 180:
        raise RobustnessRunError("Robustness run is not 180 distinct prompts")
    manifest_path = root / "manifests/inference" / f"{run_id}.json"
    receipt_path = root / "manifests/robustness" / f"{run_id}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if not manifest_path.is_file():
            raise RobustnessRunError("Completed robustness manifest is missing")
        prior = json.loads(manifest_path.read_text())
        output = root / prior["output_path"]
        if (
            receipt.get("status") != "complete"
            or receipt.get("run_id") != run_id
            or receipt.get("code_commit") != commit
            or receipt.get("perturbation_manifest_sha256") != _sha(perturb_path)
            or receipt.get("inference_manifest_sha256") != _sha(manifest_path)
            or receipt.get("output_sha256") != _sha(output)
            or prior.get("status") != "complete"
            or prior.get("completed_rows") != 180
            or prior.get("successful_rows") != 180
        ):
            raise RobustnessRunError("Existing robustness receipt differs")
        return receipt_path
    environment = collect_environment()
    if environment["gpu"].get("name") != "NVIDIA L4":
        raise RobustnessRunError("Robustness runs require the locked NVIDIA L4")
    condition = {"B0": "baseline", "F1": "mitigation"}[condition_key]
    spec = runner.RunSpec(
        run_id=run_id,
        model_id=model_key,
        condition=condition,
        output_path=f"private_outputs/robustness/{run_id}.jsonl",
        confirmatory=True,
    )
    adapter = load_model(model_key)
    manifest_path = runner.run_inference(
        prompts,
        adapter,
        spec,
        root,
        checkpoint_interval=20,
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("expected_rows") != 180
        or manifest.get("completed_rows") != 180
        or manifest.get("successful_rows") != 180
        or manifest.get("error_summary") != {}
        or manifest.get("code_commit") != commit
        or manifest.get("environment", {}).get("gpu", {}).get("name") != "NVIDIA L4"
    ):
        raise RobustnessRunError("Robustness inference did not reconcile")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id,
        "code_commit": commit,
        "perturbation_manifest_sha256": _sha(perturb_path),
        "inference_manifest_sha256": _sha(manifest_path),
        "output_sha256": manifest["output_sha256"],
        "rows": 180,
        "gpu_name": "NVIDIA L4",
    }
    content = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if receipt_path.read_bytes() != content:
            raise RobustnessRunError("Existing robustness receipt differs")
    else:
        runner._atomic_write(receipt_path, content)
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
