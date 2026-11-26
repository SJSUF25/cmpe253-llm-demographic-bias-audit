"""Run and reconcile the four registered, full-size BBQ likelihood cells."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from environment_tools import collect_environment
from inference import runner
from inference.adapter import load_model
from inference.pilot import _json_bytes, _read_jsonl, _run_bbq, _sha

ROOT = Path(__file__).resolve().parents[2]
BUILD_CONFIG = ROOT / "configs/bbq_build.json"
REGISTRY = ROOT / "configs/experiment_registry.json"
EXPECTED_IDS = (
    "BBQ-M1-B0",
    "BBQ-M1-F1",
    "BBQ-M2-B0",
    "BBQ-M2-F1",
)


class FullBBQError(ValueError):
    """A full BBQ input, run, or receipt breaks the frozen contract."""


def _entry(run_id: str) -> dict[str, Any]:
    if run_id not in EXPECTED_IDS:
        raise FullBBQError(f"Unregistered full BBQ run: {run_id}")
    matches = [
        row
        for row in json.loads(REGISTRY.read_text())["runs"]
        if row.get("id") == run_id
    ]
    if len(matches) != 1 or matches[0].get("dataset") != "D2-sample1500":
        raise FullBBQError("Full BBQ registry entry missing or changed")
    entry = matches[0]
    if (
        entry.get("model") != run_id.split("-")[1]
        or entry.get("condition") != run_id.split("-")[2]
    ):
        raise FullBBQError("Full BBQ registry model or condition changed")
    return entry


def _inputs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    config_bytes = BUILD_CONFIG.read_bytes()
    config = json.loads(config_bytes)
    manifest_path = root / config["manifest_path"]
    if not manifest_path.is_file():
        raise FullBBQError("Missing frozen BBQ build manifest")
    manifest = json.loads(manifest_path.read_text())
    item_path = root / config["items_path"]
    permutation_path = root / config["permutations_path"]
    if not item_path.is_file() or not permutation_path.is_file():
        raise FullBBQError("Missing frozen BBQ item or permutation file")
    if (
        manifest.get("config_id") != config["config_id"]
        or manifest.get("config_sha256") != _sha(config_bytes)
        or manifest.get("source_revision") != config["source_revision"]
        or manifest.get("items_path") != config["items_path"]
        or manifest.get("permutations_path") != config["permutations_path"]
        or manifest.get("items_sha256") != _sha(item_path.read_bytes())
        or manifest.get("permutations_sha256") != _sha(permutation_path.read_bytes())
        or manifest.get("items_count") != 1500
        or manifest.get("permutations_count") != 4500
    ):
        raise FullBBQError("Frozen BBQ build manifest or inputs differ")
    items = _read_jsonl(item_path)
    permutations = _read_jsonl(permutation_path)
    if (
        len(items) != 1500
        or len(permutations) != 4500
        or len({row["item_id"] for row in items}) != 1500
        or Counter(row["category"] for row in items)
        != {category: 500 for category in config["categories"]}
    ):
        raise FullBBQError("Full BBQ item inventory differs from the locked plan")
    counts = Counter(row.get("item_id") for row in permutations)
    if counts != {row["item_id"]: 3 for row in items}:
        raise FullBBQError("Full BBQ permutations differ from the locked inventory")
    return items, permutations, _sha(manifest_path.read_bytes())


def run(shared_root: Path, run_id: str) -> Path:
    root = shared_root.expanduser().resolve()
    if root == ROOT:
        raise FullBBQError("Full BBQ output must use private shared storage")
    entry = _entry(run_id)
    commit, dirty = runner.git_state()
    if dirty:
        raise FullBBQError("Full BBQ requires a clean Git worktree")
    runner._check_test_gate(root)
    items, permutations, build_sha = _inputs(root)
    environment = collect_environment()
    if environment["gpu"].get("name") != "NVIDIA L4":
        raise FullBBQError("Primary BBQ requires the locked NVIDIA L4 GPU")
    model = entry["model"]
    condition = {"B0": "baseline", "F1": "mitigation"}[entry["condition"]]
    adapter = load_model(model)
    manifest_path = _run_bbq(
        root,
        adapter,
        condition,
        items,
        permutations,
        checkpoint=10,
        commit=commit,
        registered_run_id=run_id,
        registered_output=entry["output"],
        registered_manifest=f"manifests/bbq/{run_id}.json",
        environment=environment,
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_rows") != 1500
        or manifest.get("successful_rows") != 1500
        or manifest.get("error_summary") != {}
        or manifest.get("output_path") != entry["output"]
        or manifest.get("code_commit") != commit
    ):
        raise FullBBQError("Full BBQ run did not reconcile")
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "code_commit": commit,
        "data_gate_sha256": _sha(
            (root / "manifests/data_gate_approval_v3.json").read_bytes()
        ),
        "bbq_build_manifest_sha256": build_sha,
        "run_manifest_path": str(manifest_path.relative_to(root)),
        "run_manifest_sha256": _sha(manifest_path.read_bytes()),
        "gpu_name": environment["gpu"]["name"],
    }
    receipt_path = root / f"manifests/bbq/{run_id}-receipt.json"
    content = _json_bytes(receipt)
    if receipt_path.exists():
        if receipt_path.read_bytes() != content:
            raise FullBBQError("Existing full BBQ receipt differs")
    else:
        runner._atomic_write(receipt_path, content)
    return receipt_path


def report(shared_root: Path) -> Path:
    root = shared_root.expanduser().resolve()
    rows = []
    commits = set()
    environments: dict[str, set[tuple[str, ...]]] = {}
    for run_id in EXPECTED_IDS:
        receipt_path = root / f"manifests/bbq/{run_id}-receipt.json"
        if not receipt_path.is_file():
            raise FullBBQError(f"Missing full BBQ receipt: {run_id}")
        receipt = json.loads(receipt_path.read_text())
        manifest_path = root / receipt["run_manifest_path"]
        if receipt.get("run_id") != run_id or _sha(
            manifest_path.read_bytes()
        ) != receipt.get("run_manifest_sha256"):
            raise FullBBQError(f"Changed full BBQ receipt or manifest: {run_id}")
        manifest = json.loads(manifest_path.read_text())
        output = root / manifest["output_path"]
        if (
            manifest.get("status") != "complete"
            or manifest.get("completed_rows") != 1500
            or manifest.get("successful_rows") != 1500
            or _sha(output.read_bytes()) != manifest.get("output_sha256")
        ):
            raise FullBBQError(f"Incomplete full BBQ run: {run_id}")
        environment = manifest.get("environment", {})
        gpu = environment.get("gpu", {})
        packages = environment.get("packages", {})
        if gpu.get("name") != receipt["gpu_name"]:
            raise FullBBQError(f"BBQ GPU receipt mismatch: {run_id}")
        environments.setdefault(run_id.split("-")[1], set()).add(
            (
                str(gpu.get("name")),
                str(gpu.get("cuda_runtime")),
                str(packages.get("torch")),
                str(packages.get("transformers")),
            )
        )
        scored = _read_jsonl(output)
        if len(scored) != 1500 or len({row["item_id"] for row in scored}) != 1500:
            raise FullBBQError(f"Full BBQ item count differs: {run_id}")
        commits.add(receipt["code_commit"])
        rows.append(
            {
                "run_id": run_id,
                "items": 1500,
                "accuracy": sum(row["correct"] for row in scored) / 1500,
                "order_disagreement_count": sum(
                    row["order_disagreement"] for row in scored
                ),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "gpu_name": receipt["gpu_name"],
                "run_manifest_sha256": receipt["run_manifest_sha256"],
            }
        )
    if (
        len(commits) != 1
        or any(row["gpu_name"] != "NVIDIA L4" for row in rows)
        or any(len(values) != 1 for values in environments.values())
    ):
        raise FullBBQError("Full BBQ cells used different code or GPU environments")
    result = {
        "schema_version": 1,
        "status": "complete",
        "code_commit": next(iter(commits)),
        "runs": rows,
        "items_per_cell": 1500,
        "likelihood_forward_passes": 54000,
    }
    path = root / "manifests/bbq/full_report.json"
    content = _json_bytes(result)
    if path.exists():
        if path.read_bytes() != content:
            raise FullBBQError("Existing full BBQ report differs")
    else:
        runner._atomic_write(path, content)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", choices=EXPECTED_IDS)
    args = parser.parse_args(argv)
    if args.action == "run":
        if args.run_id is None:
            parser.error("run requires --run-id")
        path = run(args.root, args.run_id)
    else:
        if args.run_id is not None:
            parser.error("report does not accept --run-id")
        path = report(args.root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
