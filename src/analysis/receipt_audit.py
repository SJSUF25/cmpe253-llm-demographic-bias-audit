"""Reconcile the measured privacy and controlled-memorization artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from privacy.canary_attack import summarize
from privacy.membership_data import DEFAULT_CONFIG, _jsonl, generate_corpus
from privacy.metrics import known_exposure
from privacy.synthetic_data import DEFAULT_CONFIG as PII_CONFIG
from privacy.synthetic_data import _jsonl as pii_jsonl
from privacy.synthetic_data import generate_records


class ReceiptAuditError(ValueError):
    """A stored output, receipt, or aggregate does not match its source."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReceiptAuditError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path, expected: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected or any(not isinstance(row, dict) for row in rows):
        raise ReceiptAuditError(f"Expected {expected} rows: {path}")
    return rows


def _equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ReceiptAuditError(f"{label} differs")


def audit(root: Path) -> dict:
    """Verify hashes and independently recompute canary and ordinary-task totals."""
    root = root.expanduser().resolve()
    sources: dict[str, str] = {}

    def checked(relative: str, digest: str | None = None) -> Path:
        path = root / relative
        actual = _sha(path)
        if digest is not None:
            _equal(actual, digest, relative + " SHA-256")
        sources[relative] = actual
        return path

    config = _load(DEFAULT_CONFIG)
    corpus = generate_corpus(config)
    membership = _load(checked("manifests/membership_data_manifest.json"))
    _equal(membership["config_sha256"], _sha(DEFAULT_CONFIG), "membership config")
    for partition in (
        "member_canaries",
        "nonmember_calibration",
        "nonmember_evaluation",
        "ordinary_validation",
    ):
        relative = f"private_outputs/membership/{partition}.jsonl"
        expected = membership["files"][relative]
        _equal(expected["rows"], len(corpus[partition]), partition + " count")
        _equal(
            hashlib.sha256(_jsonl(corpus[partition])).hexdigest(),
            expected["sha256"],
            partition + " regenerated corpus",
        )
        checked(relative, expected["sha256"])

    plan = _load(checked("manifests/canary_training/plan.json"))
    training = []
    for epoch in (1, 2, 3):
        receipt = _load(checked(f"manifests/canary_training/epoch_{epoch}.json"))
        _equal(receipt["context"], plan["context"], f"epoch {epoch} context")
        _equal(receipt["epoch"], epoch, f"epoch {epoch} number")
        _equal(receipt["presentations"], 3200, f"epoch {epoch} presentations")
        _equal(receipt["optimizer_steps"], 200, f"epoch {epoch} steps")
        training.append(receipt["mean_microbatch_loss"])

    attacks = {}
    attack_sources = {}
    for mode in ("base", "adapter"):
        manifest_path = checked(f"manifests/canary_attack/{mode}.json")
        receipt = _load(manifest_path)
        _equal(receipt["mode"], mode, mode + " attack mode")
        for partition in (
            "member_canaries",
            "nonmember_calibration",
            "nonmember_evaluation",
        ):
            relative = f"private_outputs/membership/{partition}.jsonl"
            _equal(receipt["source_hashes"][partition], sources[relative], relative)
        if mode == "adapter":
            _equal(
                receipt["training_receipt_sha256"],
                sources["manifests/canary_training/epoch_3.json"],
                "adapter training link",
            )
        output = checked(
            f"private_outputs/canary_attack/{mode}.jsonl",
            receipt["private_output_sha256"],
        )
        aggregate = _load(checked(f"aggregate_results/canary_attack_{mode}.json"))
        _equal(
            aggregate["source_manifest_sha256"],
            _sha(manifest_path),
            mode + " receipt link",
        )
        _equal(aggregate["mode"], mode, mode + " aggregate mode")
        recomputed = summarize(_rows(output, 1300))
        for key, value in recomputed.items():
            _equal(aggregate[key], value, mode + " recomputed " + key)
        attacks[mode] = aggregate
        attack_sources[mode] = receipt["code_commit"]
    _equal(attack_sources["base"], attack_sources["adapter"], "attack code commit")

    comparison = _load(checked("aggregate_results/adapter_utility_comparison.json"))
    _equal(
        comparison["source_validation_sha256"],
        sources["private_outputs/membership/ordinary_validation.jsonl"],
        "ordinary validation source",
    )
    utility = {}
    for mode in ("base", "adapter"):
        summary_path = checked(f"aggregate_results/adapter_utility_{mode}.json")
        summary = _load(summary_path)
        _equal(summary["mode"], mode, mode + " ordinary utility mode")
        _equal(
            comparison[f"{mode}_summary_sha256"],
            _sha(summary_path),
            mode + " ordinary utility link",
        )
        output = checked(
            f"private_outputs/adapter_utility/{mode}.jsonl",
            summary["private_scores_sha256"],
        )
        rows = _rows(output, 400)
        by_task: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row)
        for task, group in (("overall", rows), *sorted(by_task.items())):
            expected = (
                summary["overall"] if task == "overall" else summary["tasks"][task]
            )
            _equal(len(group), expected["records"], mode + " " + task + " rows")
            if not math.isclose(
                sum(row["target_loss"] for row in group) / len(group),
                expected["mean_target_loss"],
                rel_tol=1e-13,
                abs_tol=1e-13,
            ):
                raise ReceiptAuditError(mode + " " + task + " loss differs")
            _equal(
                sum(bool(row["exact_accuracy"]) for row in group) / len(group),
                expected["exact_accuracy"],
                mode + " " + task + " accuracy",
            )
        utility[mode] = summary
    _equal(
        utility["base"]["source_validation_sha256"],
        utility["adapter"]["source_validation_sha256"],
        "ordinary-task paired source",
    )

    exposure_path = checked("aggregate_results/privacy_exposure_summary.json")
    exposure = _load(exposure_path)
    h5 = _load(checked("aggregate_results/privacy_h5.json"))
    pii_config = _load(PII_CONFIG)
    pii_manifest = _load(checked("manifests/synthetic_pii_manifest.json"))
    _equal(pii_manifest["config_sha256"], _sha(PII_CONFIG), "PII config")
    test_records = [
        row for row in generate_records(pii_config) if row["split"] == "test"
    ]
    pii_relative = "private_outputs/synthetic_pii/test.jsonl"
    pii_sha = pii_manifest["files"][pii_relative]["sha256"]
    _equal(pii_manifest["files"][pii_relative]["rows"], 200, "PII record count")
    _equal(
        hashlib.sha256(pii_jsonl(test_records)).hexdigest(), pii_sha, "PII regeneration"
    )
    checked(pii_relative, pii_sha)
    _equal(exposure["test_input_sha256"], pii_sha, "PII exposure source")
    records = {row["record_id"]: row for row in test_records}
    exposure_cells = {
        (row["model"], row["condition"], row["task"]): row for row in exposure["cells"]
    }
    if len(exposure_cells) != 30:
        raise ReceiptAuditError("PII exposure cell inventory differs")
    _equal(h5["exposure_summary_sha256"], _sha(exposure_path), "H5 exposure link")
    _equal(
        h5["source_output_hashes"],
        exposure["source_output_hashes"],
        "H5 output inventory",
    )
    for run_id, digest in sorted(h5["source_output_hashes"].items()):
        model, condition = run_id.split("-")[1:]
        receipt = _load(checked(f"manifests/inference/{run_id}.json"))
        _equal(receipt["status"], "complete", run_id + " status")
        _equal(receipt["completed_rows"], 200, run_id + " rows")
        _equal(receipt["output_sha256"], digest, run_id + " output link")
        input_receipt = _load(checked(f"manifests/privacy/{run_id}_input.json"))
        _equal(input_receipt["input_file_sha256"], pii_sha, run_id + " input")
        _equal(
            input_receipt["run_manifest_sha256"],
            sources[f"manifests/inference/{run_id}.json"],
            run_id + " input link",
        )
        output = checked(
            f"private_outputs/pii_{model.lower()}_{condition.lower()}.jsonl",
            digest,
        )
        raw_rows = _rows(output, 200)
        _equal({row["prompt_id"] for row in raw_rows}, set(records), run_id + " keys")
        if condition == "P2":
            filter_receipt = _load(checked(f"manifests/privacy/{run_id}_filter.json"))
            _equal(
                filter_receipt["raw_output_sha256"], digest, run_id + " filter input"
            )
            _equal(
                filter_receipt["source_run_manifest_sha256"],
                sources[f"manifests/inference/{run_id}.json"],
                run_id + " filter receipt link",
            )
            _equal(
                filter_receipt["filtered_output_sha256"],
                h5["filtered_output_hashes"][run_id],
                run_id + " filtered H5 link",
            )
            filtered_path = checked(
                f"private_outputs/pii_{model.lower()}_p2_filtered.jsonl",
                filter_receipt["filtered_output_sha256"],
            )
            filtered = {row["record_id"]: row for row in _rows(filtered_path, 200)}
            _equal(set(filtered), set(records), run_id + " filtered keys")
        else:
            filtered = {}
        by_id = {row["prompt_id"]: row for row in raw_rows}
        for task in {row["task"] for row in test_records}:
            task_ids = [
                record_id for record_id, row in records.items() if row["task"] == task
            ]
            if len(task_ids) != 40:
                raise ReceiptAuditError("PII task count differs")
            exact = normalized = 0
            exact_by_type: Counter[str] = Counter()
            normalized_by_type: Counter[str] = Counter()
            for record_id in task_ids:
                response = (
                    filtered[record_id]["filtered_output"]
                    if condition == "P2"
                    else by_id[record_id]["raw_output"]
                )
                detected = known_exposure(response, records[record_id]["pii"])
                exact += any(flags["exact"] for flags in detected.values())
                normalized += any(flags["normalized"] for flags in detected.values())
                exact_by_type.update(
                    entity for entity, flags in detected.items() if flags["exact"]
                )
                normalized_by_type.update(
                    entity for entity, flags in detected.items() if flags["normalized"]
                )
            cell = exposure_cells[model, condition, task]
            _equal(cell["records"], 40, run_id + " task records " + task)
            _equal(cell["exact_record_exposures"], exact, run_id + " exact " + task)
            _equal(
                cell["normalized_record_exposures"],
                normalized,
                run_id + " normalized " + task,
            )
            _equal(
                cell["exact_by_type"],
                dict(sorted(exact_by_type.items())),
                run_id + " exact types " + task,
            )
            _equal(
                cell["normalized_by_type"],
                dict(sorted(normalized_by_type.items())),
                run_id + " normalized types " + task,
            )

    return {
        "schema_version": 1,
        "status": "verified_source_hashes_and_recomputed_canary_utility_pii_exposure",
        "source_sha256": dict(sorted(sources.items())),
        "source_count": len(sources),
        "training_epoch_mean_losses": training,
        "canary_auc": {
            mode: {
                name: attack["score_attacks"][name]["auc"]
                for name in ("negative_loss", "min_k20")
            }
            for mode, attack in attacks.items()
        },
        "ordinary_exact_accuracy": {
            mode: summary["overall"]["exact_accuracy"]
            for mode, summary in utility.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
