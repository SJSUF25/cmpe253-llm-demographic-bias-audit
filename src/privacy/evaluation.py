"""Run registered synthetic-PII controls and preserve private evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inference import runner
from inference.adapter import load_model
from privacy import controls
from privacy.synthetic_data import DEFAULT_CONFIG as DATA_CONFIG

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/experiment_registry.json"


class PrivacyEvaluationError(ValueError):
    """A registered PII run or its private artifacts are invalid."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise PrivacyEvaluationError(f"Missing private input: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_private_corpus(root: Path, data_config: dict, split: str) -> Path:
    input_path = root / data_config["output_directory"] / f"{split}.jsonl"
    manifest_path = root / data_config["manifest_path"]
    if not input_path.is_file() or not manifest_path.is_file():
        raise PrivacyEvaluationError("Missing private PII dataset or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(input_path.relative_to(root))
    expected = manifest.get("files", {}).get(relative)
    if (
        manifest.get("config_sha256") != _sha(DATA_CONFIG.read_bytes())
        or not isinstance(expected, dict)
        or expected.get("rows") != 200
        or expected.get("sha256") != _sha(input_path.read_bytes())
    ):
        raise PrivacyEvaluationError("Private PII dataset hash or count differs")
    return input_path


def _registered(run_id: str) -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [row for row in registry["runs"] if row["id"] == run_id]
    if len(matches) != 1 or not run_id.startswith("PII-"):
        raise PrivacyEvaluationError("Unknown registered PII run")
    row = matches[0]
    if row["dataset"] != "synthetic-pii-test200" or row["condition"] not in {
        "P0",
        "P1",
        "P2",
    }:
        raise PrivacyEvaluationError("Invalid PII registry entry")
    if row["model"] not in {"M1", "M2"}:
        raise PrivacyEvaluationError("Invalid PII model")
    return row


def make_prompts(
    records: list[dict], condition: str, *, analyzer: Any | None = None
) -> tuple[list[dict], dict]:
    """Build content-hashed inputs and an input-redaction audit from one split."""
    if len(records) != 200 or len({r["record_id"] for r in records}) != 200:
        raise PrivacyEvaluationError("PII split must contain 200 unique records")
    config = controls.load_config()
    prompts = []
    mismatch = []
    counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda row: row["record_id"]):
        text, audit = controls.prepare_input(
            record, condition, analyzer=analyzer, config=config
        )
        if condition != "P0":
            if text != record["reference"]["redacted_input"]:
                mismatch.append(record["record_id"])
            counts.update(audit["replacement_counts"])
        prompts.append(
            {
                "prompt_id": record["record_id"],
                "scenario_id": record["record_id"],
                "dataset": "synthetic-pii",
                "prompt_text": text,
                "prompt_sha256": _sha(text.encode()),
                "control_config_sha256": controls.config_sha256(),
            }
        )
    if mismatch:
        raise PrivacyEvaluationError(
            f"Input redaction differs from reference for {len(mismatch)} records"
        )
    if condition != "P0" and any(
        counts[entity] != 200 for entity in config["placeholders"]
    ):
        raise PrivacyEvaluationError(
            "PII replacement counts do not match split inventory"
        )
    return prompts, {
        "records": len(records),
        "redaction_reference_match_count": 0 if condition == "P0" else len(records),
        "input_replacement_counts": dict(sorted(counts.items())),
        "control_config_sha256": controls.config_sha256(),
    }


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise PrivacyEvaluationError(f"Existing private artifact differs: {path}")
    else:
        runner._atomic_write(path, content)


def finalize_p2(
    root: Path,
    run_id: str,
    records: list[dict],
    *,
    analyzer: Any,
    output_path: Path,
    run_manifest: Path,
) -> Path:
    """Filter P2 raw generations and keep a hash-linked private filtered file."""
    raw_rows = _rows(output_path)
    by_id = {row["record_id"]: row for row in records}
    if (
        len(raw_rows) != 200
        or {row["prompt_id"] for row in raw_rows} != set(by_id)
        or any(row["status"] != "ok" for row in raw_rows)
    ):
        raise PrivacyEvaluationError(
            "P2 raw outputs are missing, duplicated, or failed"
        )
    filtered = []
    replacement_counts: Counter[str] = Counter()
    for row in sorted(raw_rows, key=lambda item: item["prompt_id"]):
        record = by_id[row["prompt_id"]]
        text, audit = controls.filter_output(
            row["raw_output"], record, "P2", analyzer=analyzer
        )
        replacement_counts.update(audit["replacement_counts"])
        filtered.append(
            {
                "record_id": row["prompt_id"],
                "run_id": run_id,
                "raw_output_sha256": row["output_sha256"],
                "filtered_output": text,
                "filtered_output_sha256": _sha(text.encode()),
                "filter_audit": audit,
            }
        )
    filtered_path = output_path.with_name(output_path.stem + "_filtered.jsonl")
    content = (
        "\n".join(json.dumps(row, sort_keys=True) for row in filtered) + "\n"
    ).encode()
    _write_once(filtered_path, content)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_manifest_sha256": _sha(run_manifest.read_bytes()),
        "raw_output_sha256": _sha(output_path.read_bytes()),
        "filtered_output_sha256": _sha(content),
        "filtered_output_path": str(filtered_path.relative_to(root)),
        "records": len(filtered),
        "replacement_counts": dict(sorted(replacement_counts.items())),
        "control_config_sha256": controls.config_sha256(),
    }
    manifest_path = root / "manifests/privacy" / f"{run_id}_filter.json"
    _write_once(
        manifest_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    )
    return manifest_path


def run_registered_privacy(shared_root: Path, run_id: str, split: str) -> Path:
    """Run one pinned model, condition, and split; never inspect test before gate."""
    if split not in {"development", "test"}:
        raise PrivacyEvaluationError("Split must be development or test")
    entry = _registered(run_id)
    root = shared_root.expanduser().resolve()
    if root == ROOT:
        raise PrivacyEvaluationError("Privacy outputs require shared storage")
    commit, dirty = runner.git_state()
    if split == "test":
        if dirty:
            raise PrivacyEvaluationError("Test inference requires a clean Git worktree")
        runner._check_test_gate(root)
    data_config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    input_path = _verify_private_corpus(root, data_config, split)
    records = _rows(input_path)
    if any(row["split"] != split for row in records):
        raise PrivacyEvaluationError("PII input contains another split")
    analyzer = controls.make_analyzer() if entry["condition"] != "P0" else None
    prompts, input_audit = make_prompts(records, entry["condition"], analyzer=analyzer)
    output = entry["output"]
    effective_id = run_id
    if split == "development":
        output = f"private_outputs/pilot/{Path(output).name}"
        effective_id = f"{run_id}-DEV"
    spec = runner.RunSpec(
        run_id=effective_id,
        model_id=entry["model"],
        condition=entry["condition"],
        output_path=output,
        confirmatory=split == "test",
    )
    adapter = load_model(entry["model"])
    manifest = runner.run_inference(prompts, adapter, spec, root)
    audit_path = root / "manifests/privacy" / f"{effective_id}_input.json"
    audit = {
        "schema_version": 1,
        "run_id": effective_id,
        "split": split,
        "input_file_sha256": _sha(input_path.read_bytes()),
        "run_manifest_sha256": _sha(manifest.read_bytes()),
        **input_audit,
    }
    _write_once(
        audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    )
    if entry["condition"] == "P2":
        if analyzer is None:
            raise PrivacyEvaluationError("P2 analyzer was not loaded")
        return finalize_p2(
            root,
            effective_id,
            records,
            analyzer=analyzer,
            output_path=root / output,
            run_manifest=manifest,
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args(argv)
    print(run_registered_privacy(args.root, args.run_id, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
