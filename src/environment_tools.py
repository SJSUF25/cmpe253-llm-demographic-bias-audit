"""Capture reproducibility metadata and configure deterministic execution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

TRACKED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "numpy",
    "peft",
    "safetensors",
    "sentencepiece",
    "torch",
    "transformers",
)


def configure_determinism(seed: int = 253) -> dict[str, object]:
    """Seed supported libraries and enable deterministic CUDA behavior."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_smi() -> dict[str, str | None]:
    query = "driver_version,name,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"driver_gpu_memory": None}
    return {"driver_gpu_memory": result.stdout.strip()}


def collect_environment() -> dict[str, object]:
    """Return a JSON-serializable software and accelerator inventory."""
    import torch

    gpu: dict[str, object] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        gpu.update(
            {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "cuda_runtime": torch.version.cuda,
            }
        )
    gpu.update(_nvidia_smi())
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "os_release": platform.release(),
        "packages": _package_versions(),
        "gpu": gpu,
    }


def write_environment(path: Path) -> Path:
    """Atomically write an environment manifest."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(collect_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(write_environment(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
