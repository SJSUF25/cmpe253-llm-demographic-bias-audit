"""Prepare blinded privacy helpfulness review and ordinary adapter utility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from inference import runner
from privacy import canary_attack, evaluation, h5
from privacy.membership_data import DEFAULT_CONFIG as MEMBERSHIP_CONFIG
from privacy.membership_data import generate_corpus
from privacy.synthetic_data import DEFAULT_CONFIG as PII_CONFIG


class UtilityError(ValueError):
    """A utility source, blinded packet, or model result is incomplete."""


def audit_sample(records: list[dict]) -> list[dict]:
    """Select exactly ten stable-hash test records from each of five tasks."""
    tasks: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["split"] != "test":
            raise UtilityError("Helpfulness packet received non-test record")
        tasks[record["task"]].append(record)
    if len(tasks) != 5 or any(len(rows) != 40 for rows in tasks.values()):
        raise UtilityError("Helpfulness sample needs 40 test records per task")
    selected = []
    for _task, rows in sorted(tasks.items()):
        selected.extend(
            sorted(
                rows,
                key=lambda row: (
                    evaluation._sha(f"253|helpfulness|{row['record_id']}".encode()),
                    row["record_id"],
                ),
            )[:10]
        )
    return selected


def prepare_helpfulness_packet(root: Path) -> Path:
    """Write 100 blinded model-record rows and a separate private alias key."""
    root = root.expanduser().resolve()
    runner._check_test_gate(root)
    config = json.loads(PII_CONFIG.read_text(encoding="utf-8"))
    data_path = evaluation._verify_private_corpus(root, config, "test")
    selected = audit_sample(evaluation._rows(data_path))
    packet = []
    key = []
    all_source_hashes = {}
    for model in ("M1", "M2"):
        outputs = {}
        for condition in ("P0", "P1", "P2"):
            rows, raw_hash, filtered_hash = h5._load_outputs(root, model, condition)
            outputs[condition] = rows
            all_source_hashes[f"{model}:{condition}"] = {
                "raw": raw_hash,
                "filtered": filtered_hash,
            }
        for record in selected:
            record_id = record["record_id"]
            order = sorted(
                ("P0", "P1", "P2"),
                key=lambda condition: evaluation._sha(
                    f"253|blind|{model}|{record_id}|{condition}".encode()
                ),
            )
            aliases = dict(zip(("A", "B", "C"), order, strict=True))
            packet.append(
                {
                    "audit_id": evaluation._sha(f"{model}|{record_id}".encode())[:16],
                    "task": record["task"],
                    "case_facts": h5._reference_text(record),
                    "responses": {
                        alias: h5.normalize_known(
                            outputs[condition][record_id], record["pii"]
                        )
                        for alias, condition in aliases.items()
                    },
                    "rating_scale": "helpfulness_1_to_5",
                }
            )
            key.append(
                {
                    "audit_id": packet[-1]["audit_id"],
                    "model": model,
                    "record_id": record_id,
                    "aliases": aliases,
                }
            )
    directory = root / "private_outputs/privacy_h5"
    packet_path = directory / "helpfulness_packet.jsonl"
    key_path = directory / "helpfulness_alias_key.jsonl"
    manifest_path = root / "manifests/privacy/helpfulness_packet.json"
    packet_bytes = (
        "\n".join(json.dumps(row, sort_keys=True) for row in packet) + "\n"
    ).encode()
    key_bytes = (
        "\n".join(json.dumps(row, sort_keys=True) for row in key) + "\n"
    ).encode()
    evaluation._write_once(packet_path, packet_bytes)
    evaluation._write_once(key_path, key_bytes)
    source = {
        "schema_version": 1,
        "test_input_sha256": evaluation._sha(data_path.read_bytes()),
        "source_hashes": all_source_hashes,
        "packet_sha256": evaluation._sha(packet_bytes),
        "alias_key_sha256": evaluation._sha(key_bytes),
        "selected_records": 50,
        "model_record_rows": 100,
        "ratings_expected": 300,
    }
    evaluation._write_once(
        manifest_path, (json.dumps(source, indent=2, sort_keys=True) + "\n").encode()
    )
    return packet_path


def _validation_records(root: Path) -> tuple[list[dict], str]:
    config = json.loads(MEMBERSHIP_CONFIG.read_text(encoding="utf-8"))
    manifest_path = root / config["manifest_path"]
    path = root / config["output_directory"] / "ordinary_validation.jsonl"
    if not manifest_path.is_file() or not path.is_file():
        raise UtilityError("Missing ordinary validation corpus")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = path.read_bytes()
    expected = generate_corpus(config)["ordinary_validation"]
    if manifest.get("config_sha256") != evaluation._sha(
        MEMBERSHIP_CONFIG.read_bytes()
    ) or manifest.get("files", {}).get(str(path.relative_to(root))) != {
        "rows": 400,
        "sha256": evaluation._sha(content),
    }:
        raise UtilityError("Ordinary validation hash differs")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    if rows != expected:
        raise UtilityError("Ordinary validation records differ from frozen generator")
    return rows, evaluation._sha(content)


def score_adapter_utility(root: Path, mode: str) -> Path:
    """Score exact ordinary tasks and target-only loss before/after adapter."""
    if mode not in canary_attack.MODES:
        raise UtilityError("Mode must be base or adapter")
    root = root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise UtilityError("Adapter utility requires a clean Git worktree")
    records, source_hash = _validation_records(root)
    model, tokenizer, context = canary_attack._load_model(root, mode)
    scored = []
    for record in records:
        result = canary_attack.score_one(
            model,
            tokenizer,
            {**record, "repetition": 0},
        )
        scored.append(
            {
                "record_id": result["record_id"],
                "task": record["task"],
                "target_loss": -result["negative_loss"],
                "exact_accuracy": result["extracted_exact"],
            }
        )
    private_path = root / f"private_outputs/adapter_utility/{mode}.jsonl"
    content = (
        "\n".join(json.dumps(row, sort_keys=True) for row in scored) + "\n"
    ).encode()
    evaluation._write_once(private_path, content)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_task[row["task"]].append(row)
    if len(by_task) != 4 or any(len(rows) != 100 for rows in by_task.values()):
        raise UtilityError("Ordinary validation task cells differ")
    summary = {
        "schema_version": 1,
        "mode": mode,
        "code_commit": commit,
        "source_validation_sha256": source_hash,
        "private_scores_sha256": evaluation._sha(content),
        **context,
        "overall": {
            "records": 400,
            "mean_target_loss": sum(row["target_loss"] for row in scored) / 400,
            "exact_accuracy": sum(row["exact_accuracy"] for row in scored) / 400,
        },
        "tasks": {
            task: {
                "records": 100,
                "mean_target_loss": sum(row["target_loss"] for row in rows) / 100,
                "exact_accuracy": sum(row["exact_accuracy"] for row in rows) / 100,
            }
            for task, rows in sorted(by_task.items())
        },
    }
    output = root / f"aggregate_results/adapter_utility_{mode}.json"
    evaluation._write_once(
        output, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    )
    return output


def compare_adapter_utility(root: Path) -> Path:
    """Compare the same 400 ordinary records before and after adapter attachment."""
    root = root.expanduser().resolve()
    summaries = {}
    for mode in canary_attack.MODES:
        path = root / f"aggregate_results/adapter_utility_{mode}.json"
        if not path.is_file():
            raise UtilityError(f"Missing ordinary utility results: {mode}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        private_path = root / f"private_outputs/adapter_utility/{mode}.jsonl"
        if (
            not private_path.is_file()
            or summary.get("mode") != mode
            or summary.get("private_scores_sha256")
            != evaluation._sha(private_path.read_bytes())
        ):
            raise UtilityError(f"Changed ordinary utility results: {mode}")
        summaries[mode] = summary
    base, adapter = summaries["base"], summaries["adapter"]
    if (
        base["source_validation_sha256"] != adapter["source_validation_sha256"]
        or base["model_revision"] != adapter["model_revision"]
        or set(base["tasks"]) != set(adapter["tasks"])
    ):
        raise UtilityError("Base and adapter utility sources differ")

    def difference(left: dict, right: dict) -> dict:
        return {
            "records": left["records"],
            "base_mean_target_loss": left["mean_target_loss"],
            "adapter_mean_target_loss": right["mean_target_loss"],
            "adapter_minus_base_target_loss": (
                right["mean_target_loss"] - left["mean_target_loss"]
            ),
            "base_exact_accuracy": left["exact_accuracy"],
            "adapter_exact_accuracy": right["exact_accuracy"],
            "adapter_minus_base_exact_accuracy": (
                right["exact_accuracy"] - left["exact_accuracy"]
            ),
        }

    result = {
        "schema_version": 1,
        "source_validation_sha256": base["source_validation_sha256"],
        "base_summary_sha256": evaluation._sha(
            (root / "aggregate_results/adapter_utility_base.json").read_bytes()
        ),
        "adapter_summary_sha256": evaluation._sha(
            (root / "aggregate_results/adapter_utility_adapter.json").read_bytes()
        ),
        "overall": difference(base["overall"], adapter["overall"]),
        "tasks": {
            task: difference(base["tasks"][task], adapter["tasks"][task])
            for task in sorted(base["tasks"])
        },
    }
    output = root / "aggregate_results/adapter_utility_comparison.json"
    evaluation._write_once(
        output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    )
    return output


def score_helpfulness_ratings(root: Path) -> Path:
    """Unblind exactly 300 locked 1–5 ratings and write aggregate means only."""
    root = root.expanduser().resolve()
    directory = root / "private_outputs/privacy_h5"
    packet = directory / "helpfulness_packet.jsonl"
    key_path = directory / "helpfulness_alias_key.jsonl"
    ratings_path = directory / "helpfulness_ratings.jsonl"
    manifest_path = root / "manifests/privacy/helpfulness_packet.json"
    if any(
        not path.is_file() for path in (packet, key_path, ratings_path, manifest_path)
    ):
        raise UtilityError("Helpfulness packet, key, or ratings are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("packet_sha256") != evaluation._sha(
        packet.read_bytes()
    ) or manifest.get("alias_key_sha256") != evaluation._sha(key_path.read_bytes()):
        raise UtilityError("Blinded packet hash differs")
    keys = evaluation._rows(key_path)
    aliases = {
        (row["audit_id"], alias): (row["model"], condition)
        for row in keys
        for alias, condition in row["aliases"].items()
    }
    ratings = evaluation._rows(ratings_path)
    if len(keys) != 100 or len(aliases) != 300 or len(ratings) != 300:
        raise UtilityError("Expected 300 blinded ratings")
    observed = {}
    for row in ratings:
        key = (row.get("audit_id"), row.get("alias"))
        if (
            key not in aliases
            or key in observed
            or type(row.get("rating")) is not int
            or row["rating"] not in (1, 2, 3, 4, 5)
            or row.get("reviewer") != "Vineet Malewar"
        ):
            raise UtilityError("Invalid or duplicate blinded rating")
        observed[key] = row["rating"]
    if set(observed) != set(aliases):
        raise UtilityError("Blinded rating keys are incomplete")
    grouped: dict[str, list[int]] = defaultdict(list)
    for key, rating in observed.items():
        model, condition = aliases[key]
        grouped[f"{model}:{condition}"].append(rating)
    if len(grouped) != 6 or any(len(values) != 50 for values in grouped.values()):
        raise UtilityError("Helpfulness condition cells differ")
    result = {
        "schema_version": 1,
        "packet_manifest_sha256": evaluation._sha(manifest_path.read_bytes()),
        "private_ratings_sha256": evaluation._sha(ratings_path.read_bytes()),
        "scope": "50 records per model; blinded construct-validity check",
        "cells": {
            key: {"ratings": 50, "mean_helpfulness": sum(values) / 50}
            for key, values in sorted(grouped.items())
        },
    }
    output = root / "aggregate_results/privacy_helpfulness.json"
    evaluation._write_once(
        output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-packet")
    sub.add_parser("score-helpfulness")
    sub.add_parser("compare-adapter-utility")
    score = sub.add_parser("adapter-utility")
    score.add_argument("--mode", choices=canary_attack.MODES, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit-packet":
        result = prepare_helpfulness_packet(args.root)
    elif args.command == "score-helpfulness":
        result = score_helpfulness_ratings(args.root)
    elif args.command == "compare-adapter-utility":
        result = compare_adapter_utility(args.root)
    else:
        result = score_adapter_utility(args.root, args.mode)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
