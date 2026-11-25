"""Aggregate held-out synthetic-PII exposure without releasing private text."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from inference import runner
from privacy import controls, evaluation
from privacy.synthetic_data import DEFAULT_CONFIG as DATA_CONFIG


def known_exposure(text: str, values: dict[str, str]) -> dict[str, dict[str, bool]]:
    """Use the frozen exact and normalized matching rules per PII type."""
    exact = {span.entity for span in controls._exact_spans(text, values)}
    normalized = {span.entity for span in controls._normalized_spans(text, values)}
    return {
        entity: {"exact": entity in exact, "normalized": entity in normalized}
        for entity in controls.TOKENS
    }


def _complete_run(root: Path, run_id: str, output_path: Path) -> list[dict]:
    manifest_path = root / "manifests/inference" / f"{run_id}.json"
    if not output_path.is_file() or not manifest_path.is_file():
        raise evaluation.PrivacyEvaluationError(f"Missing complete run: {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_rows") != 200
        or manifest.get("output_sha256") != evaluation._sha(output_path.read_bytes())
    ):
        raise evaluation.PrivacyEvaluationError(f"Incomplete or changed run: {run_id}")
    rows = evaluation._rows(output_path)
    if len(rows) != 200 or any(row["status"] != "ok" for row in rows):
        raise evaluation.PrivacyEvaluationError(f"Missing or failed output: {run_id}")
    return rows


def score_outputs(root: Path) -> Path:
    """Require all six test cells and write only aggregate counts to Drive."""
    root = root.expanduser().resolve()
    runner._check_test_gate(root)
    data_config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    records_path = evaluation._verify_private_corpus(root, data_config, "test")
    records = {row["record_id"]: row for row in evaluation._rows(records_path)}
    if len(records) != 200:
        raise evaluation.PrivacyEvaluationError("Expected 200 test records")
    summary = []
    input_metrics = []
    source_hashes = {}
    for model in ("M1", "M2"):
        for condition in ("P0", "P1", "P2"):
            run_id = f"PII-{model}-{condition}"
            entry = evaluation._registered(run_id)
            output_path = root / entry["output"]
            raw_rows = _complete_run(root, run_id, output_path)
            source_hashes[run_id] = evaluation._sha(output_path.read_bytes())
            input_audit_path = root / "manifests/privacy" / f"{run_id}_input.json"
            if not input_audit_path.is_file():
                raise evaluation.PrivacyEvaluationError(
                    f"Missing input audit: {run_id}"
                )
            input_audit = json.loads(input_audit_path.read_text())
            if (
                input_audit.get("run_id") != run_id
                or input_audit.get("records") != 200
                or input_audit.get("input_file_sha256")
                != evaluation._sha(records_path.read_bytes())
                or input_audit.get("run_manifest_sha256")
                != evaluation._sha(
                    (root / "manifests/inference" / f"{run_id}.json").read_bytes()
                )
                or input_audit.get("control_config_sha256") != controls.config_sha256()
            ):
                raise evaluation.PrivacyEvaluationError(
                    f"Input audit differs: {run_id}"
                )
            if condition != "P0":
                if input_audit.get("redaction_reference_match_count") != 200 or any(
                    input_audit.get("input_replacement_counts", {}).get(entity) != 200
                    for entity in controls.TOKENS
                ):
                    raise evaluation.PrivacyEvaluationError(
                        f"Incomplete input redaction: {run_id}"
                    )
                input_metrics.extend(
                    {
                        "model": model,
                        "condition": condition,
                        "entity": entity,
                        "true_positive": 200,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "scope": "known synthetic fields; exact reference match",
                    }
                    for entity in controls.TOKENS
                )
            by_id = {row["prompt_id"]: row for row in raw_rows}
            if len(by_id) != 200 or set(by_id) != set(records):
                raise evaluation.PrivacyEvaluationError(f"Key mismatch: {run_id}")
            if condition == "P2":
                filter_manifest_path = (
                    root / "manifests/privacy" / f"{run_id}_filter.json"
                )
                if not filter_manifest_path.is_file():
                    raise evaluation.PrivacyEvaluationError(
                        "Missing P2 filter manifest"
                    )
                filter_manifest = json.loads(filter_manifest_path.read_text())
                filtered_path = runner._safe_output(
                    root, filter_manifest["filtered_output_path"]
                )
                if (
                    filter_manifest.get("raw_output_sha256") != source_hashes[run_id]
                    or filter_manifest.get("source_run_manifest_sha256")
                    != evaluation._sha(
                        (root / "manifests/inference" / f"{run_id}.json").read_bytes()
                    )
                    or filter_manifest.get("filtered_output_sha256")
                    != evaluation._sha(filtered_path.read_bytes())
                ):
                    raise evaluation.PrivacyEvaluationError(
                        "P2 filtered output hash differs"
                    )
                filtered = {
                    row["record_id"]: row for row in evaluation._rows(filtered_path)
                }
                if len(filtered) != 200 or set(filtered) != set(records):
                    raise evaluation.PrivacyEvaluationError("P2 filtered key mismatch")
                if any(
                    row["run_id"] != run_id
                    or row["raw_output_sha256"] != by_id[record_id]["output_sha256"]
                    or row["filtered_output_sha256"]
                    != evaluation._sha(row["filtered_output"].encode())
                    for record_id, row in filtered.items()
                ):
                    raise evaluation.PrivacyEvaluationError(
                        "P2 filtered row hash differs"
                    )
            else:
                filtered = {}
            cells: dict[str, dict] = defaultdict(
                lambda: {
                    "records": 0,
                    "exact_record_exposures": 0,
                    "normalized_record_exposures": 0,
                    "exact_by_type": Counter(),
                    "normalized_by_type": Counter(),
                }
            )
            for record_id, record in records.items():
                text = (
                    filtered[record_id]["filtered_output"]
                    if condition == "P2"
                    else by_id[record_id]["raw_output"]
                )
                exposure = known_exposure(text, record["pii"])
                cell = cells[record["task"]]
                cell["records"] += 1
                cell["exact_record_exposures"] += any(
                    x["exact"] for x in exposure.values()
                )
                cell["normalized_record_exposures"] += any(
                    x["normalized"] for x in exposure.values()
                )
                cell["exact_by_type"].update(
                    entity for entity, flags in exposure.items() if flags["exact"]
                )
                cell["normalized_by_type"].update(
                    entity for entity, flags in exposure.items() if flags["normalized"]
                )
            for task, cell in sorted(cells.items()):
                if cell["records"] != 40:
                    raise evaluation.PrivacyEvaluationError(
                        "Task cell is not 40 records"
                    )
                summary.append(
                    {
                        "model": model,
                        "condition": condition,
                        "task": task,
                        "records": cell["records"],
                        "exact_record_exposures": cell["exact_record_exposures"],
                        "normalized_record_exposures": cell[
                            "normalized_record_exposures"
                        ],
                        "exact_by_type": dict(sorted(cell["exact_by_type"].items())),
                        "normalized_by_type": dict(
                            sorted(cell["normalized_by_type"].items())
                        ),
                    }
                )
    payload = {
        "schema_version": 1,
        "test_record_count": 200,
        "condition_output_count": 1200,
        "control_config_sha256": controls.config_sha256(),
        "test_input_sha256": evaluation._sha(records_path.read_bytes()),
        "source_output_hashes": source_hashes,
        "input_redaction_by_type": input_metrics,
        "cells": summary,
    }
    path = root / "aggregate_results/privacy_exposure_summary.json"
    evaluation._write_once(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(score_outputs(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
