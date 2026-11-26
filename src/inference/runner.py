"""Resumable, content-verified inference over private prompt manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from data.cab_quality import MANIFEST as CAB_QUALITY_MANIFEST
from data.cab_quality import CabQualityError
from data.cab_quality import verify_manifest as verify_cab_quality_manifest
from environment_tools import collect_environment
from inference.adapter import DecodeConfig, ModelAdapter, decode_config, load_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/inference.json"
LOCK_PATH = ROOT / "uv.lock"
REGISTRY_PATH = ROOT / "configs/experiment_registry.json"
CAB_CONFIG_PATH = ROOT / "configs/cab_build.json"
GOVERNANCE_LOCK_PATH = ROOT / "results/manifests/governance_lock.json"
CHECKPOINT_INTERVAL = 100
RUN_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")
REQUIRED_FIELDS = {
    "run_id",
    "prompt_id",
    "scenario_id",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "condition",
    "system_prompt_version",
    "precision",
    "device",
    "seed",
    "do_sample",
    "temperature",
    "top_p",
    "max_new_tokens",
    "input_token_count",
    "output_token_count",
    "raw_output",
    "normalized_output",
    "finish_reason",
    "runtime_seconds",
    "timestamp_utc",
    "code_commit",
    "config_sha256",
    "status",
    "error_type",
    "decoding_config_hash",
    "output_sha256",
    "prompt_sha256",
    "retry_count",
}


class RunnerError(ValueError):
    """An inference plan, checkpoint, or output violates the frozen run contract."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_id: str
    condition: str
    output_path: str
    confirmatory: bool


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _safe_output(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.parts[0] != "private_outputs"
    ):
        raise RunnerError(f"Unsafe private output path: {relative}")
    return root / Path(*pure.parts)


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise RunnerError(f"Empty or blank inference partition: {path}")
        rows = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"Malformed inference partition: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise RunnerError(f"Non-object inference row: {path}")
    return rows


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"\n".join(_canonical(row) for row in rows) + b"\n"


def _prompt_inventory(prompts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    for prompt in prompts:
        for field in ("prompt_id", "scenario_id", "prompt_text", "prompt_sha256"):
            if field not in prompt:
                raise RunnerError(f"Prompt missing {field}")
        prompt_id = prompt["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in by_id:
            raise RunnerError(f"Duplicate or invalid prompt ID: {prompt_id}")
        if _hash(prompt["prompt_text"].encode()) != prompt["prompt_sha256"]:
            raise RunnerError(f"Prompt hash mismatch: {prompt_id}")
        by_id[prompt_id] = prompt
    if not by_id:
        raise RunnerError("Inference input is empty")
    return by_id


def _key(
    prompt_id: str, revision: str, condition: str, settings: DecodeConfig
) -> tuple[str, ...]:
    decode_hash = _hash(_canonical(asdict(settings)))
    return prompt_id, revision, condition, str(settings.seed), decode_hash


def _validate_record(
    row: dict[str, Any],
    spec: RunSpec,
    adapter: ModelAdapter,
    prompts: dict[str, dict[str, Any]],
    settings: DecodeConfig,
    commit: str,
    config_sha: str,
) -> tuple[str, ...]:
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise RunnerError(f"Inference record missing fields: {sorted(missing)}")
    prompt = prompts.get(row["prompt_id"])
    if (
        prompt is None
        or row["scenario_id"] != prompt["scenario_id"]
        or row["prompt_sha256"] != prompt["prompt_sha256"]
    ):
        raise RunnerError("Inference record does not match prompt inventory")
    expected = _key(row["prompt_id"], adapter.spec.revision, spec.condition, settings)
    actual = (
        row["prompt_id"],
        row["model_revision"],
        row["condition"],
        str(row["seed"]),
        row["decoding_config_hash"],
    )
    if (
        actual != expected
        or row["model_id"] != spec.model_id
        or row["run_id"] != spec.run_id
    ):
        raise RunnerError("Inference record key or run metadata mismatch")
    if row["code_commit"] != commit or row["config_sha256"] != config_sha:
        raise RunnerError("Inference record code/config mismatch")
    if row["tokenizer_revision"] != adapter.spec.tokenizer_revision:
        raise RunnerError("Inference tokenizer revision mismatch")
    expected_version = adapter.config["conditions"][spec.condition]["version"]
    if (
        row["system_prompt_version"] != expected_version
        or row["precision"] != adapter.config["precision"]
        or row["seed"] != settings.seed
        or row["do_sample"] != settings.do_sample
        or row["temperature"] != settings.temperature
        or row["top_p"] != settings.top_p
        or row["max_new_tokens"] != settings.max_new_tokens
        or row["retry_count"] not in (0, 1)
    ):
        raise RunnerError("Inference decoding or instruction metadata mismatch")
    if row["status"] not in {"ok", "error"}:
        raise RunnerError("Invalid inference status")
    if row["status"] == "ok":
        if row["error_type"] is not None or not isinstance(row["raw_output"], str):
            raise RunnerError("Successful inference row has invalid output/error")
        if (
            row["normalized_output"] != row["raw_output"].strip()
            or not isinstance(row["input_token_count"], int)
            or row["input_token_count"] < 1
            or not isinstance(row["output_token_count"], int)
            or row["output_token_count"] < 1
            or row["finish_reason"] not in {"eos", "length"}
        ):
            raise RunnerError(
                "Successful inference row has invalid token/output fields"
            )
        if _hash(row["raw_output"].encode()) != row["output_sha256"]:
            raise RunnerError("Inference output hash mismatch")
    elif (
        row["raw_output"] is not None
        or row["normalized_output"] is not None
        or row["input_token_count"] is not None
        or row["output_token_count"] is not None
        or row["finish_reason"] is not None
        or not row["error_type"]
        or row["output_sha256"] is not None
    ):
        raise RunnerError("Failed inference row has invalid output/error")
    return expected


def _infrastructure_error(error: Exception) -> bool:
    return isinstance(error, (OSError, TimeoutError, ConnectionError)) or type(
        error
    ).__name__ in {
        "OutOfMemoryError",
        "CUDAError",
    }


def _error_record(
    adapter: ModelAdapter, condition: str, settings: DecodeConfig, error: Exception
) -> dict[str, Any]:
    return {
        "model_id": adapter.spec.key,
        "model_revision": adapter.spec.revision,
        "tokenizer_revision": adapter.spec.tokenizer_revision,
        "condition": condition,
        "system_prompt_version": adapter.config["conditions"][condition]["version"],
        "precision": adapter.config["precision"],
        "device": str(adapter.model.device),
        "seed": settings.seed,
        "do_sample": settings.do_sample,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_new_tokens": settings.max_new_tokens,
        "input_token_count": None,
        "output_token_count": None,
        "raw_output": None,
        "normalized_output": None,
        "finish_reason": None,
        "runtime_seconds": None,
        "error_type": type(error).__name__,
        "output_sha256": None,
    }


def _produce_row(
    prompt: dict[str, Any],
    adapter: ModelAdapter,
    spec: RunSpec,
    settings: DecodeConfig,
    commit: str,
    config_sha: str,
) -> dict[str, Any]:
    retries = 0
    try:
        result = adapter.generate(
            prompt["prompt_text"], spec.condition, decoding=settings
        )
    except Exception as error:
        if _infrastructure_error(error):
            retries = 1
            try:
                result = adapter.generate(
                    prompt["prompt_text"], spec.condition, decoding=settings
                )
            except Exception as retry_error:
                result = _error_record(adapter, spec.condition, settings, retry_error)
        else:
            result = _error_record(adapter, spec.condition, settings, error)
    if result.get("raw_output") is not None:
        result = {
            **result,
            "error_type": None,
            "output_sha256": _hash(result["raw_output"].encode()),
        }
        status = "ok"
    else:
        status = "error"
    return {
        **result,
        "run_id": spec.run_id,
        "prompt_id": prompt["prompt_id"],
        "scenario_id": prompt["scenario_id"],
        "prompt_sha256": prompt["prompt_sha256"],
        "decoding_config_hash": _key(
            prompt["prompt_id"], adapter.spec.revision, spec.condition, settings
        )[4],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "config_sha256": config_sha,
        "status": status,
        "retry_count": retries,
    }


def run_inference(
    prompts: list[dict[str, Any]],
    adapter: ModelAdapter,
    spec: RunSpec,
    shared_root: Path,
    *,
    settings: DecodeConfig | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    state_provider: Callable[[], tuple[str, bool]] = git_state,
    environment_provider: Callable[[], dict[str, Any]] = collect_environment,
    gate_checker: Callable[[Path], None] | None = None,
) -> Path:
    """Resume a run from verified partitions; return its private run manifest."""
    if checkpoint_interval < 1:
        raise RunnerError("Checkpoint interval must be positive")
    if not RUN_ID_PATTERN.fullmatch(spec.run_id):
        raise RunnerError("Unsafe inference run ID")
    if (
        spec.model_id != adapter.spec.key
        or spec.condition not in adapter.config["conditions"]
    ):
        raise RunnerError("Run spec does not match loaded model or condition")
    settings = settings or decode_config(config=adapter.config)
    by_id = _prompt_inventory(prompts)
    commit, dirty = state_provider()
    if spec.confirmatory and dirty:
        raise RunnerError("Confirmatory inference requires a clean Git worktree")
    config_sha = _hash(CONFIG_PATH.read_bytes())
    lock_sha = _hash(LOCK_PATH.read_bytes())
    input_sha = _hash(_jsonl(prompts))
    root = shared_root.expanduser().resolve()
    if root == ROOT:
        raise RunnerError("Private inference output must use shared storage")
    if spec.confirmatory:
        (gate_checker or _check_test_gate)(root)
    output_path = _safe_output(root, spec.output_path)
    partition_dir = output_path.with_suffix("") / "parts"
    manifest_path = root / "manifests" / "inference" / f"{spec.run_id}.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    run_identity = {
        "run_id": spec.run_id,
        "model_id": spec.model_id,
        "model_revision": adapter.spec.revision,
        "tokenizer_revision": adapter.spec.tokenizer_revision,
        "condition": spec.condition,
        "output_path": spec.output_path,
        "confirmatory": spec.confirmatory,
        "code_commit": commit,
        "dirty_worktree": dirty,
        "config_sha256": config_sha,
        "package_lock_sha256": lock_sha,
        "input_sha256": input_sha,
        "decoding_config_hash": _key(
            "", adapter.spec.revision, spec.condition, settings
        )[4],
        "seed": settings.seed,
        "checkpoint_interval": checkpoint_interval,
    }
    if existing_manifest and any(
        existing_manifest.get(key) != value for key, value in run_identity.items()
    ):
        raise RunnerError("Existing run manifest identity differs")
    parts = sorted(partition_dir.glob("part-*.jsonl")) if partition_dir.exists() else []
    completed: dict[tuple[str, ...], dict[str, Any]] = {}
    partition_entries = []
    for index, path in enumerate(parts):
        if path.name != f"part-{index:06d}.jsonl":
            raise RunnerError("Inference partition sequence has a gap")
        rows = _read_jsonl(path)
        if len(rows) > checkpoint_interval or (
            index < len(parts) - 1 and len(rows) != checkpoint_interval
        ):
            raise RunnerError("Inference partition has an invalid row count")
        for row in rows:
            key = _validate_record(
                row, spec, adapter, by_id, settings, commit, config_sha
            )
            if key in completed:
                prior = completed[key]
                if prior["output_sha256"] != row["output_sha256"]:
                    raise RunnerError("Conflicting duplicate inference key")
                raise RunnerError("Duplicate inference key")
            completed[key] = row
        partition_entries.append(
            {
                "path": str(path.relative_to(root)),
                "rows": len(rows),
                "sha256": _hash(path.read_bytes()),
            }
        )
    if existing_manifest:
        previously = existing_manifest.get("partitions", [])
        if partition_entries[: len(previously)] != previously:
            raise RunnerError("Existing manifest and partitions disagree")
    if output_path.exists():
        expected = b"".join(path.read_bytes() for path in parts)
        if len(completed) != len(prompts) or output_path.read_bytes() != expected:
            raise RunnerError("Existing final output differs from partitions")
    start_time = time.perf_counter()
    started = (
        existing_manifest["started_at_utc"]
        if existing_manifest
        else datetime.now(UTC).isoformat()
    )
    accumulated = (
        existing_manifest.get("elapsed_seconds", 0.0) if existing_manifest else 0.0
    )
    environment = (
        existing_manifest.get("environment")
        if existing_manifest
        else environment_provider()
    )

    def checkpoint(*, finished: bool) -> None:
        elapsed = accumulated + time.perf_counter() - start_time
        errors = Counter(
            row["error_type"] for row in completed.values() if row["status"] == "error"
        )
        manifest = {
            "schema_version": 1,
            **run_identity,
            "started_at_utc": started,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "ended_at_utc": datetime.now(UTC).isoformat() if finished else None,
            "elapsed_seconds": elapsed,
            "expected_rows": len(prompts),
            "completed_rows": len(completed),
            "successful_rows": sum(row["status"] == "ok" for row in completed.values()),
            "error_summary": dict(sorted(errors.items())),
            "partitions": partition_entries,
            "output_sha256": _hash(output_path.read_bytes()) if finished else None,
            "environment": environment,
            "status": "complete_with_errors"
            if finished and errors
            else "complete"
            if finished
            else "in_progress",
        }
        _atomic_write(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )

    pending_rows = []
    for prompt in prompts:
        key = _key(prompt["prompt_id"], adapter.spec.revision, spec.condition, settings)
        if key in completed:
            continue
        row = _produce_row(prompt, adapter, spec, settings, commit, config_sha)
        _validate_record(row, spec, adapter, by_id, settings, commit, config_sha)
        pending_rows.append(row)
        if len(pending_rows) == checkpoint_interval:
            path = partition_dir / f"part-{len(partition_entries):06d}.jsonl"
            if path.exists():
                raise RunnerError("Refusing to overwrite an inference partition")
            content = _jsonl(pending_rows)
            _atomic_write(path, content)
            partition_entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": len(pending_rows),
                    "sha256": _hash(content),
                }
            )
            completed.update(
                (
                    _key(
                        row["prompt_id"],
                        adapter.spec.revision,
                        spec.condition,
                        settings,
                    ),
                    row,
                )
                for row in pending_rows
            )
            pending_rows = []
            checkpoint(finished=False)
    if pending_rows:
        path = partition_dir / f"part-{len(partition_entries):06d}.jsonl"
        if path.exists():
            raise RunnerError("Refusing to overwrite an inference partition")
        content = _jsonl(pending_rows)
        _atomic_write(path, content)
        partition_entries.append(
            {
                "path": str(path.relative_to(root)),
                "rows": len(pending_rows),
                "sha256": _hash(content),
            }
        )
        completed.update(
            (
                _key(row["prompt_id"], adapter.spec.revision, spec.condition, settings),
                row,
            )
            for row in pending_rows
        )
        checkpoint(finished=False)
    if len(completed) != len(prompts):
        raise RunnerError("Inference finished with missing prompt keys")
    final_bytes = b"".join(
        (root / entry["path"]).read_bytes() for entry in partition_entries
    )
    if output_path.exists():
        if output_path.read_bytes() != final_bytes:
            raise RunnerError("Existing final output differs")
    else:
        _atomic_write(output_path, final_bytes)
    checkpoint(finished=True)
    if spec.confirmatory and any(
        row["status"] == "error" for row in completed.values()
    ):
        raise RunnerError(
            f"Confirmatory run completed with errors; inspect {manifest_path}"
        )
    return manifest_path


def _check_test_gate(root: Path) -> None:
    """Require structural, tokenizer, quality, and truthful review receipts."""
    validation = root / "manifests/data_validation_manifest.json"
    token_lengths = root / "manifests/token_length_manifest.json"
    quality_path = root / CAB_QUALITY_MANIFEST
    receipt = root / "manifests/data_gate_approval_v2.json"
    if not receipt.is_file():
        receipt = root / "manifests/data_gate_approval.json"
    if (
        not validation.is_file()
        or not token_lengths.is_file()
        or not quality_path.is_file()
        or not receipt.is_file()
    ):
        raise RunnerError(
            "CAB test run requires the private data-gate approval receipt"
        )
    structural = json.loads(validation.read_text(encoding="utf-8"))
    lengths = json.loads(token_lengths.read_text(encoding="utf-8"))
    approval = json.loads(receipt.read_text(encoding="utf-8"))
    try:
        quality = verify_cab_quality_manifest(root)
    except CabQualityError as error:
        raise RunnerError(str(error)) from error
    cab_config = json.loads(CAB_CONFIG_PATH.read_text(encoding="utf-8"))
    bbq_config = json.loads(
        (ROOT / "configs/bbq_build.json").read_text(encoding="utf-8")
    )
    membership_config = json.loads(
        (ROOT / "configs/membership_data.json").read_text(encoding="utf-8")
    )
    expected_inputs = {
        cab_config["output_path"],
        bbq_config["items_path"],
        bbq_config["permutations_path"],
        f"{membership_config['output_directory']}/training_presentations.jsonl",
    }
    input_hashes = lengths.get("inputs_sha256")
    inputs_match = (
        isinstance(input_hashes, dict) and set(input_hashes) == expected_inputs
    )
    if inputs_match:
        inputs_match = all(
            (root / relative).is_file()
            and _hash((root / relative).read_bytes()) == input_hashes[relative]
            for relative in expected_inputs
        )
    common = {
        "schema_version": 1,
        "validation_manifest_sha256": _hash(validation.read_bytes()),
        "token_length_manifest_sha256": _hash(token_lengths.read_bytes()),
        "cab_implicit_quality_manifest_sha256": _hash(quality_path.read_bytes()),
        "governance_lock_sha256": _hash(GOVERNANCE_LOCK_PATH.read_bytes()),
        "token_length_status": "passed",
    }
    human_approval = {
        **common,
        "human_review_status": "approved",
        "approved_by": ["Abhinand"],
    }
    ai_receipt_path = root / (
        "manifests/data_ai_review_receipt_v2.json"
        if receipt.name == "data_gate_approval_v2.json"
        else "manifests/data_ai_review_receipt.json"
    )
    reviewed_packet_path = root / "private_outputs/data_qa/review_packet.json"
    ai_approval = {
        **common,
        "review_status": "approved_with_documented_implicit_exclusions",
        "review_method": "ai_assisted_no_human_review",
        "reviewed_by": "Codex AI-assisted audit",
        "review_packet_sha256": structural.get("review_packet_sha256"),
        "review_evidence_sha256": _hash(
            (ROOT / "docs/data_qa_ai_review.md").read_bytes()
        ),
    }
    ai_receipt_matches = (
        ai_receipt_path.is_file()
        and reviewed_packet_path.is_file()
        and structural.get("review_packet_sha256")
        == _hash(reviewed_packet_path.read_bytes())
        and _hash(ai_receipt_path.read_bytes())
        == approval.get("ai_review_receipt_sha256")
    )
    if ai_receipt_matches:
        ai_receipt = json.loads(ai_receipt_path.read_text(encoding="utf-8"))
        ai_receipt_matches = ai_receipt == ai_approval
    expected_ai_gate = {
        **ai_approval,
        "ai_review_receipt_sha256": _hash(ai_receipt_path.read_bytes())
        if ai_receipt_path.is_file()
        else None,
    }
    if (
        structural.get("structural_checks") != "passed"
        or lengths.get("status") != "passed"
        or quality.get("status") != "passed_with_implicit_exclusions"
        or lengths.get("validation_manifest_sha256") != _hash(validation.read_bytes())
        or not inputs_match
        or not (
            approval == human_approval
            or (approval == expected_ai_gate and ai_receipt_matches)
        )
    ):
        raise RunnerError("CAB test data-gate receipt is incomplete or stale")


def run_registered_cab(shared_root: Path, run_id: str, split: str) -> Path:
    """Run one registered CAB model/condition on development or approved test data."""
    if split not in {"development", "test"}:
        raise RunnerError("CAB split must be development or test")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    matches = [row for row in registry["runs"] if row["id"] == run_id]
    if len(matches) != 1:
        raise RunnerError(f"Unknown experiment run ID: {run_id}")
    entry = matches[0]
    if run_id.startswith("CAB-EXP-") and entry["dataset"] == "D1":
        mode = "explicit"
    elif run_id.startswith("CAB-IMP-") and entry["dataset"] == "D1-implicit":
        mode = "implicit"
    else:
        raise RunnerError("Only registered CAB explicit/implicit runs are supported")
    condition = {"B0": "baseline", "F1": "mitigation"}.get(entry["condition"])
    if condition is None or entry["model"] not in {"M1", "M2"}:
        raise RunnerError("Unsupported registered model or condition")
    root = shared_root.expanduser().resolve()
    if root == ROOT:
        raise RunnerError("Private inference output must use shared storage")
    commit, dirty = git_state()
    if split == "test":
        if dirty:
            raise RunnerError("Confirmatory inference requires a clean Git worktree")
        _check_test_gate(root)
    if mode == "implicit":
        try:
            quality = verify_cab_quality_manifest(root)
        except CabQualityError as error:
            raise RunnerError(str(error)) from error
        excluded = set(quality["excluded_implicit_scenario_ids"])
    else:
        excluded = set()
    cab_config = json.loads(CAB_CONFIG_PATH.read_text(encoding="utf-8"))
    prompt_path = root / cab_config["output_path"]
    if not prompt_path.is_file():
        raise RunnerError("Missing built CAB prompt manifest")
    prompts = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        row
        for row in prompts
        if row["split"] == split
        and row["explicit_implicit"] == mode
        and row["scenario_id"] not in excluded
    ]
    if not selected:
        raise RunnerError("Registered CAB selection is empty")
    if split == "development":
        basename = PurePosixPath(entry["output"]).stem
        output = f"private_outputs/pilot/{basename}_development.jsonl"
        effective_id = f"{run_id}-DEV"
    else:
        output = entry["output"]
        effective_id = run_id
    spec = RunSpec(
        run_id=effective_id,
        model_id=entry["model"],
        condition=condition,
        output_path=output,
        confirmatory=split == "test",
    )
    model_adapter = load_model(entry["model"])
    return run_inference(selected, model_adapter, spec, root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args(argv)
    print(run_registered_cab(args.root, args.run_id, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
