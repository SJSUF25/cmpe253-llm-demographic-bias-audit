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
    "pyproject.toml",
    "uv.lock",
    "configs/experiment_registry.json",
    "configs/upstream_registry.json",
    "configs/model_manifest.yaml",
    "configs/data_manifest.yaml",
    "configs/raw_data_sources.json",
    "configs/smoke_fixture.json",
    "docs/system_boundary.md",
    "docs/experiment_registry.md",
    "notebooks/e0_t04_colab_smoke_test.ipynb",
    "src/environment_tools.py",
    "src/data/download.py",
    "src/inference/smoke_test.py",
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
        "approval_status": "pending_both_members",
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
