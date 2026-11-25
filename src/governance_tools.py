"""Create a content-addressed lock for governance and experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_FILES = (
    "LLM_Bias_Audit_PRD.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "configs/experiment_registry.json",
    "configs/upstream_registry.json",
    "configs/model_manifest.yaml",
    "configs/data_manifest.yaml",
    "configs/cab_build.json",
    "configs/cab_scoring.json",
    "configs/confirmatory_analysis.json",
    "configs/bbq_build.json",
    "configs/privacy_data.json",
    "configs/privacy_controls.json",
    "configs/canary_training.json",
    "configs/membership_data.json",
    "configs/inference.json",
    "configs/pilot.json",
    "configs/raw_data_sources.json",
    "configs/smoke_fixture.json",
    "docs/system_boundary.md",
    "docs/experiment_registry.md",
    "docs/data_qa_checklist.md",
    "docs/data_qa_ai_review.md",
    "docs/privacy_controls.md",
    "docs/privacy_evaluation.md",
    "docs/canary_training.md",
    "docs/canary_attack.md",
    "docs/privacy_h5.md",
    "docs/bbq_scoring.md",
    "docs/cab_scoring.md",
    "docs/cab_annotation.md",
    "docs/analysis_plan.md",
    "docs/analysis_statistics.md",
    "docs/model_adapter.md",
    "docs/inference_runner.md",
    "docs/pilot.md",
    "notebooks/e0_t04_colab_smoke_test.ipynb",
    "notebooks/e1_t06_data_gate_colab.ipynb",
    "notebooks/e2_t04_pilot_colab.ipynb",
    "src/environment_tools.py",
    "src/data/cab_builder.py",
    "src/data/cab_quality.py",
    "src/data/ai_review_gate.py",
    "src/data/bbq_sampler.py",
    "src/data/download.py",
    "src/data/validate.py",
    "src/data/token_lengths.py",
    "src/privacy/synthetic_data.py",
    "src/privacy/controls.py",
    "src/privacy/evaluation.py",
    "src/privacy/metrics.py",
    "src/privacy/train_canary.py",
    "src/privacy/canary_attack.py",
    "src/privacy/h5.py",
    "src/privacy/utility.py",
    "src/analysis/statistics.py",
    "src/privacy/membership_data.py",
    "src/inference/smoke_test.py",
    "src/inference/adapter.py",
    "src/inference/runner.py",
    "src/inference/pilot.py",
    "src/scoring/bbq.py",
    "src/scoring/cab.py",
    "src/scoring/annotation.py",
    "src/scoring/annotation_review.py",
    "src/analysis/plan.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_lock(output: Path) -> Path:
    missing = [relative for relative in LOCKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing governance inputs: {', '.join(missing)}")
    entries = [
        {
            "path": relative,
            "sha256": sha256_file(ROOT / relative),
            "size_bytes": (ROOT / relative).stat().st_size,
        }
        for relative in LOCKED_FILES
    ]
    manifest = {
        "schema_version": 1,
        "lock_version": 1,
        "approval_status": "pending_operator_review",
        "files": entries,
    }
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(create_lock(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
