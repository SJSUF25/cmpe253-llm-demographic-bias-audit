"""Frozen E2-T04 pilot selection, execution, reconciliation, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from environment_tools import collect_environment, comparable_environment
from inference import runner
from inference.adapter import load_model
from scoring.bbq import format_question, score_item

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/pilot.json"
CAB_CONFIG = ROOT / "configs/cab_build.json"
BBQ_CONFIG = ROOT / "configs/bbq_build.json"
RUN_LABEL = "PILOT2"
RUN_DIRECTORY = "pilot_v2"


class PilotError(ValueError):
    """The pilot plan, evidence, or output violates its frozen contract."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + "\n"
    ).encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise PilotError(f"Existing pilot artifact differs: {path}")
        return
    runner._atomic_write(path, content)


def _ordered(
    rows: list[dict[str, Any]], namespace: str, key: str
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{namespace}{row[key]}".encode()).hexdigest(),
            str(row[key]),
        ),
    )


def _allocate(
    total: int, availability: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """Hamilton allocation proportional to available rows with lexical tie breaks."""
    available = sum(availability.values())
    if total < 1 or available < total or not availability:
        raise PilotError("Pilot stratum inventory cannot satisfy the requested total")
    exact = {key: total * count / available for key, count in availability.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(availability, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    if any(result[key] > availability[key] for key in result):
        raise PilotError("Pilot allocation exceeds a stratum inventory")
    return result


def build_selection(shared_root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Freeze the 50 CAB families and stratified 100 BBQ items before model use."""
    root = shared_root.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if (
        config.get("schema_version") != 1
        or config.get("models") != ["M1", "M2"]
        or config.get("conditions") != ["baseline", "mitigation"]
    ):
        raise PilotError("Unsupported pilot configuration")
    cab_config = json.loads(CAB_CONFIG.read_text())
    bbq_config = json.loads(BBQ_CONFIG.read_text())
    cab_source = root / cab_config["output_path"]
    bbq_source = root / bbq_config["items_path"]
    permutation_source = root / bbq_config["permutations_path"]
    for path in (cab_source, bbq_source, permutation_source):
        if not path.is_file():
            raise PilotError(f"Missing built pilot input: {path}")

    cab_rows = _read_jsonl(cab_source)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cab_rows:
        if (
            row.get("split") == config["cab"]["split"]
            and row.get("explicit_implicit") == config["cab"]["mode"]
        ):
            by_family[row["scenario_id"]].append(row)
    selected_family_ids: list[str] = []
    for attribute, count in config["cab"]["families_per_attribute"].items():
        families = [
            {"scenario_id": scenario_id}
            for scenario_id, rows in by_family.items()
            if {row["attribute"] for row in rows} == {attribute}
        ]
        chosen = _ordered(
            families, config["selection_namespace"] + f"cab|{attribute}|", "scenario_id"
        )[:count]
        if len(chosen) != count:
            raise PilotError(f"Insufficient development CAB families for {attribute}")
        selected_family_ids.extend(row["scenario_id"] for row in chosen)
    selected_set = set(selected_family_ids)
    selected_cab = [
        row
        for row in cab_rows
        if row["scenario_id"] in selected_set and row["explicit_implicit"] == "explicit"
    ]
    if len(selected_set) != sum(config["cab"]["families_per_attribute"].values()):
        raise PilotError("Duplicate CAB family in pilot selection")

    bbq_rows = _read_jsonl(bbq_source)
    strata_fields = config["bbq"]["strata"]
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in bbq_rows:
        by_stratum[tuple(str(row[field]) for field in strata_fields)].append(row)
    allocation = _allocate(
        config["bbq"]["items"], {key: len(rows) for key, rows in by_stratum.items()}
    )
    selected_bbq: list[dict[str, Any]] = []
    for key in sorted(by_stratum):
        selected_bbq.extend(
            _ordered(
                by_stratum[key],
                config["selection_namespace"] + "bbq|" + "|".join(key) + "|",
                "item_id",
            )[: allocation[key]]
        )
    selected_item_ids = {row["item_id"] for row in selected_bbq}
    selected_permutations = [
        row
        for row in _read_jsonl(permutation_source)
        if row["item_id"] in selected_item_ids
    ]
    if len(selected_bbq) != config["bbq"]["items"] or len(
        selected_permutations
    ) != 3 * len(selected_bbq):
        raise PilotError("Pilot BBQ selection or permutation inventory is incomplete")

    directory = root / "private_outputs/pilot/selection"
    cab_path = directory / "cab_prompts.jsonl"
    bbq_path = directory / "bbq_items.jsonl"
    permutations_path = directory / "bbq_permutations.jsonl"
    artifacts = (
        (cab_path, _jsonl_bytes(selected_cab)),
        (bbq_path, _jsonl_bytes(selected_bbq)),
        (permutations_path, _jsonl_bytes(selected_permutations)),
    )
    for path, content in artifacts:
        _immutable(path, content)
    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": _sha(config_bytes),
        "source_sha256": {
            str(cab_source.relative_to(root)): _sha(cab_source.read_bytes()),
            str(bbq_source.relative_to(root)): _sha(bbq_source.read_bytes()),
            str(permutation_source.relative_to(root)): _sha(
                permutation_source.read_bytes()
            ),
        },
        "cab": {
            "families": len(selected_set),
            "prompts": len(selected_cab),
            "family_ids": sorted(selected_set),
            "attribute_counts": dict(
                sorted(
                    Counter(
                        by_family[scenario_id][0]["attribute"]
                        for scenario_id in selected_set
                    ).items()
                )
            ),
            "path": str(cab_path.relative_to(root)),
            "sha256": _sha(cab_path.read_bytes()),
        },
        "bbq": {
            "items": len(selected_bbq),
            "permutations": len(selected_permutations),
            "allocation": {
                "|".join(key): allocation[key] for key in sorted(allocation)
            },
            "item_ids": sorted(selected_item_ids),
            "items_path": str(bbq_path.relative_to(root)),
            "items_sha256": _sha(bbq_path.read_bytes()),
            "permutations_path": str(permutations_path.relative_to(root)),
            "permutations_sha256": _sha(permutations_path.read_bytes()),
        },
    }
    manifest_path = root / "manifests/pilot/selection.json"
    _immutable(manifest_path, _json_bytes(manifest))
    return manifest_path


def _verify_selection(
    root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    path = root / "manifests/pilot/selection.json"
    if not path.is_file():
        raise PilotError("Missing pilot selection; run the plan action first")
    manifest = json.loads(path.read_text())
    cab_path = root / manifest["cab"]["path"]
    bbq_path = root / manifest["bbq"]["items_path"]
    permutations_path = root / manifest["bbq"]["permutations_path"]
    if (
        _sha(cab_path.read_bytes()) != manifest["cab"]["sha256"]
        or _sha(bbq_path.read_bytes()) != manifest["bbq"]["items_sha256"]
        or _sha(permutations_path.read_bytes())
        != manifest["bbq"]["permutations_sha256"]
    ):
        raise PilotError("Pilot selection artifact changed after freezing")
    return (
        manifest,
        _read_jsonl(cab_path),
        _read_jsonl(bbq_path),
        _read_jsonl(permutations_path),
    )


def _cuda_peak() -> int | None:
    try:
        import torch

        return (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        )
    except ImportError:
        return None


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _preflight(
    adapter: Any,
    cab: list[dict[str, Any]],
    bbq: list[dict[str, Any]],
    permutations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check greedy order invariance and repeated likelihoods on pilot inputs."""
    prompts = _ordered(cab, "253|pilot-preflight|", "prompt_id")[:2]
    if len(prompts) != 2:
        raise PilotError("Pilot preflight requires two CAB prompts")
    output_hashes: dict[str, str] = {}
    for ordered in (prompts, list(reversed(prompts))):
        for row in ordered:
            result = adapter.generate(row["prompt_text"], "baseline")
            digest = _sha(result["raw_output"].encode())
            previous = output_hashes.setdefault(row["prompt_id"], digest)
            if previous != digest:
                raise PilotError("Greedy output changed with prompt execution order")
    item = _ordered(bbq, "253|pilot-preflight-bbq|", "item_id")[0]
    variants = sorted(
        (row for row in permutations if row["item_id"] == item["item_id"]),
        key=lambda row: row["order"],
    )
    if len(variants) != 3:
        raise PilotError("Pilot preflight BBQ item lacks three option orders")
    prompt = format_question(item, variants[0]["answers"])
    answer = variants[0]["answers"][0]
    first = adapter.conditional_log_likelihood(prompt, answer, "baseline")
    second = adapter.conditional_log_likelihood(prompt, answer, "baseline")
    delta = abs(first - second)
    if delta > 1e-4:
        raise PilotError("Repeated pilot likelihood exceeds the 1e-4 tolerance")
    return {
        "status": "passed",
        "greedy_order_invariant_prompt_ids": sorted(output_hashes),
        "greedy_output_sha256": dict(sorted(output_hashes.items())),
        "likelihood_item_id": item["item_id"],
        "likelihood_max_absolute_delta": delta,
    }


def _run_bbq(
    root: Path,
    adapter: Any,
    condition: str,
    items: list[dict[str, Any]],
    permutations: list[dict[str, Any]],
    *,
    checkpoint: int,
    commit: str,
    registered_run_id: str | None = None,
    registered_output: str | None = None,
    registered_manifest: str | None = None,
    environment: dict[str, Any] | None = None,
) -> Path:
    if checkpoint < 1:
        raise PilotError("BBQ pilot checkpoint must be positive")
    label = "B0" if condition == "baseline" else "F1"
    custom = (registered_run_id, registered_output, registered_manifest)
    if any(value is not None for value in custom) and not all(
        value is not None for value in custom
    ):
        raise PilotError("Registered BBQ run requires ID, output, and manifest")
    run_id = registered_run_id or f"{RUN_LABEL}-BBQ-{adapter.spec.key}-{label}"
    output = runner._safe_output(
        root,
        registered_output or f"private_outputs/{RUN_DIRECTORY}/{run_id.lower()}.jsonl",
    )
    parts = output.with_suffix("") / "parts"
    manifest_relative = (
        registered_manifest or f"manifests/{RUN_DIRECTORY}/{run_id}.json"
    )
    manifest_pure = PurePosixPath(manifest_relative)
    if (
        manifest_pure.is_absolute()
        or ".." in manifest_pure.parts
        or not manifest_pure.parts
        or manifest_pure.parts[0] != "manifests"
    ):
        raise PilotError("Unsafe BBQ manifest path")
    manifest_path = root / Path(*manifest_pure.parts)
    input_sha = _sha(_jsonl_bytes(items) + _jsonl_bytes(permutations))
    environment_sha = (
        _sha(_json_bytes(comparable_environment(environment)))
        if environment is not None
        else None
    )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        partitions_match = all(
            (root / entry["path"]).is_file()
            and _sha((root / entry["path"]).read_bytes()) == entry["sha256"]
            for entry in existing.get("partitions", [])
        )
        if (
            existing.get("run_id") == run_id
            and existing.get("code_commit") == commit
            and existing.get("input_sha256") == input_sha
            and (
                environment_sha is None
                or existing.get("environment_sha256") == environment_sha
            )
            and existing.get("status") == "complete"
            and existing.get("completed_rows") == len(items)
            and partitions_match
            and output.is_file()
            and existing.get("output_sha256") == _sha(output.read_bytes())
            and output.read_bytes()
            == b"".join(
                (root / entry["path"]).read_bytes() for entry in existing["partitions"]
            )
        ):
            return manifest_path
        raise PilotError(f"Existing BBQ pilot manifest differs: {manifest_path}")
    by_item = defaultdict(list)
    for row in permutations:
        by_item[row["item_id"]].append(row)
    completed: dict[str, dict[str, Any]] = {}
    expected_items = {row["item_id"] for row in items}
    entries = []
    for index, path in enumerate(
        sorted(parts.glob("part-*.jsonl")) if parts.exists() else []
    ):
        if path.name != f"part-{index:06d}.jsonl":
            raise PilotError("BBQ pilot partition sequence has a gap")
        rows = _read_jsonl(path)
        for row in rows:
            if (
                row.get("run_id") != run_id
                or row.get("code_commit") != commit
                or row.get("item_id") not in expected_items
                or row.get("model_id") != adapter.spec.key
                or row.get("model_revision") != adapter.spec.revision
                or row.get("tokenizer_revision") != adapter.spec.tokenizer_revision
                or row.get("condition") != condition
                or row.get("system_prompt_version")
                != adapter.config["conditions"][condition]["version"]
                or (
                    environment_sha is not None
                    and row.get("environment_sha256") != environment_sha
                )
            ):
                raise PilotError("BBQ pilot partition identity changed")
            if row["item_id"] in completed:
                raise PilotError("Duplicate BBQ pilot item")
            completed[row["item_id"]] = row
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "rows": len(rows),
                "sha256": _sha(path.read_bytes()),
            }
        )
    started = time.perf_counter()
    pending = []
    for item in items:
        if item["item_id"] in completed:
            continue
        item_start = time.perf_counter()
        result = score_item(
            item,
            by_item[item["item_id"]],
            lambda prompt, answer: adapter.conditional_log_likelihood(
                prompt, answer, condition
            ),
            model_id=adapter.spec.key,
            condition=condition,
        )
        pending.append(
            {
                **result,
                "run_id": run_id,
                "model_revision": adapter.spec.revision,
                "tokenizer_revision": adapter.spec.tokenizer_revision,
                "system_prompt_version": adapter.config["conditions"][condition][
                    "version"
                ],
                "runtime_seconds": time.perf_counter() - item_start,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "code_commit": commit,
                **(
                    {"environment_sha256": environment_sha}
                    if environment_sha is not None
                    else {}
                ),
            }
        )
        if len(pending) == checkpoint:
            path = parts / f"part-{len(entries):06d}.jsonl"
            content = _jsonl_bytes(pending)
            _immutable(path, content)
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": len(pending),
                    "sha256": _sha(content),
                }
            )
            completed.update((row["item_id"], row) for row in pending)
            pending = []
    if pending:
        path = parts / f"part-{len(entries):06d}.jsonl"
        content = _jsonl_bytes(pending)
        _immutable(path, content)
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "rows": len(pending),
                "sha256": _sha(content),
            }
        )
        completed.update((row["item_id"], row) for row in pending)
    if set(completed) != expected_items:
        raise PilotError("BBQ pilot did not reconcile every selected item")
    final = b"".join((root / entry["path"]).read_bytes() for entry in entries)
    _immutable(output, final)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "BBQ",
        "model_id": adapter.spec.key,
        "condition": condition,
        "code_commit": commit,
        "input_sha256": input_sha,
        "model_revision": adapter.spec.revision,
        "tokenizer_revision": adapter.spec.tokenizer_revision,
        "status": "complete",
        "expected_rows": len(items),
        "completed_rows": len(completed),
        "successful_rows": len(completed),
        "error_summary": {},
        "elapsed_seconds": sum(row["runtime_seconds"] for row in completed.values()),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "peak_gpu_memory_bytes": _cuda_peak(),
        "partitions": entries,
        "output_path": str(output.relative_to(root)),
        "output_sha256": _sha(output.read_bytes()),
    }
    if environment is not None:
        manifest["environment"] = environment
        manifest["environment_sha256"] = environment_sha
    _immutable(manifest_path, _json_bytes(manifest))
    return manifest_path


def run_model(
    shared_root: Path, model_key: str, config_path: Path = DEFAULT_CONFIG
) -> Path:
    """Run both pilot conditions for one model while keeping only that model loaded."""
    if model_key not in {"M1", "M2"}:
        raise PilotError("Pilot model must be M1 or M2")
    root = shared_root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise PilotError("Pilot requires a clean Git worktree")
    runner._check_test_gate(root)
    selection, cab, bbq, permutations = _verify_selection(root)
    config = json.loads(config_path.read_text())
    receipt_path = root / f"manifests/{RUN_DIRECTORY}/model-{model_key}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") == "complete"
            and receipt.get("model_id") == model_key
            and receipt.get("code_commit") == commit
            and receipt.get("selection_manifest_sha256")
            == _sha((root / "manifests/pilot/selection.json").read_bytes())
            and receipt.get("preflight", {}).get("status") == "passed"
            and receipt["preflight"].get("likelihood_max_absolute_delta", 1.0) <= 1e-4
            and all(
                (root / run["manifest"]).is_file()
                and _sha((root / run["manifest"]).read_bytes()) == run["sha256"]
                for run in receipt.get("runs", [])
            )
            and len(receipt.get("runs", [])) == 4
        ):
            return receipt_path
        raise PilotError(f"Existing model pilot receipt differs: {receipt_path}")
    adapter = load_model(model_key)
    receipts = []
    environment = collect_environment()
    preflight = _preflight(adapter, cab, bbq, permutations)
    for condition in config["conditions"]:
        label = "B0" if condition == "baseline" else "F1"
        _reset_cuda_peak()
        cab_manifest = runner.run_inference(
            cab,
            adapter,
            runner.RunSpec(
                run_id=f"{RUN_LABEL}-CAB-{model_key}-{label}",
                model_id=model_key,
                condition=condition,
                output_path=f"private_outputs/{RUN_DIRECTORY}/pilot2-cab-{model_key.lower()}-{label.lower()}.jsonl",
                confirmatory=False,
            ),
            root,
        )
        receipts.append(
            {
                "dataset": "CAB",
                "condition": condition,
                "manifest": str(cab_manifest.relative_to(root)),
                "sha256": _sha(cab_manifest.read_bytes()),
                "peak_gpu_memory_bytes": _cuda_peak(),
            }
        )
        _reset_cuda_peak()
        bbq_manifest = _run_bbq(
            root,
            adapter,
            condition,
            bbq,
            permutations,
            checkpoint=config["bbq_checkpoint_items"],
            commit=commit,
        )
        receipts.append(
            {
                "dataset": "BBQ",
                "condition": condition,
                "manifest": str(bbq_manifest.relative_to(root)),
                "sha256": _sha(bbq_manifest.read_bytes()),
                "peak_gpu_memory_bytes": _cuda_peak(),
            }
        )
    receipt = {
        "schema_version": 1,
        "model_id": model_key,
        "code_commit": commit,
        "selection_manifest_sha256": _sha(
            (root / "manifests/pilot/selection.json").read_bytes()
        ),
        "environment": environment,
        "preflight": preflight,
        "runs": receipts,
        "status": "complete",
    }
    _immutable(receipt_path, _json_bytes(receipt))
    return receipt_path


def build_report(shared_root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Reconcile all eight pilot cells and produce projections and review material."""
    root = shared_root.expanduser().resolve()
    selection, _, _, _ = _verify_selection(root)
    config = json.loads(config_path.read_text())
    summaries = []
    review_rows = []
    gpu_by_model: dict[str, set[str]] = defaultdict(set)
    for model in config["models"]:
        receipt_path = root / f"manifests/{RUN_DIRECTORY}/model-{model}.json"
        if not receipt_path.is_file():
            raise PilotError(f"Missing completed model pilot receipt: {model}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete" or len(receipt.get("runs", [])) != 4:
            raise PilotError(f"Incomplete model pilot receipt: {model}")
        if (
            receipt.get("preflight", {}).get("status") != "passed"
            or receipt["preflight"].get("likelihood_max_absolute_delta", 1.0) > 1e-4
        ):
            raise PilotError(f"Model pilot preflight failed: {model}")
        gpu_by_model[model].add(str(receipt["environment"]["gpu"].get("name")))
        for entry in receipt["runs"]:
            manifest_path = root / entry["manifest"]
            if _sha(manifest_path.read_bytes()) != entry["sha256"]:
                raise PilotError("Pilot run manifest changed after model receipt")
            manifest = json.loads(manifest_path.read_text())
            expected = (
                selection["cab"]["prompts"]
                if entry["dataset"] == "CAB"
                else selection["bbq"]["items"]
            )
            if (
                manifest.get("status") != "complete"
                or manifest.get("completed_rows") != expected
                or manifest.get("successful_rows") != expected
                or manifest.get("error_summary") != {}
            ):
                raise PilotError("Pilot run has missing rows or errors")
            summaries.append(
                {
                    "model_id": model,
                    "dataset": entry["dataset"],
                    "condition": entry["condition"],
                    "rows": expected,
                    "elapsed_seconds": manifest["elapsed_seconds"],
                    "seconds_per_row": manifest["elapsed_seconds"] / expected,
                    "peak_gpu_memory_bytes": entry["peak_gpu_memory_bytes"],
                }
            )
            if entry["dataset"] == "CAB":
                output = root / manifest["output_path"]
                rows = _read_jsonl(output)
                summaries[-1].update(
                    max_input_tokens=max(row["input_token_count"] for row in rows),
                    max_output_tokens=max(row["output_token_count"] for row in rows),
                    length_finish_count=sum(
                        row["finish_reason"] == "length" for row in rows
                    ),
                    empty_normalized_output_count=sum(
                        not row["normalized_output"] for row in rows
                    ),
                )
                review_rows.extend(
                    {
                        "run_id": manifest["run_id"],
                        "prompt_id": row["prompt_id"],
                        "scenario_id": row["scenario_id"],
                        "normalized_output": row["normalized_output"],
                        "finish_reason": row["finish_reason"],
                    }
                    for row in _ordered(
                        rows, f"253|pilot-review|{manifest['run_id']}|", "prompt_id"
                    )[: config["review_cab_rows_per_run"]]
                )
            else:
                output = root / manifest["output_path"]
                rows = _read_jsonl(output)
                summaries[-1].update(
                    order_disagreement_count=sum(
                        row["order_disagreement"] for row in rows
                    ),
                    accuracy=sum(row["correct"] for row in rows) / len(rows),
                )
    if any(len(names) != 1 or "None" in names for names in gpu_by_model.values()):
        raise PilotError("A model's pilot conditions did not record one GPU class")
    cab_config = json.loads(CAB_CONFIG.read_text())
    cab_rows = _read_jsonl(root / cab_config["output_path"])
    quality = runner.verify_cab_quality_manifest(root)
    excluded = set(quality["excluded_implicit_scenario_ids"])
    full_cab_per_cell = sum(
        row["split"] == "test"
        and (
            row["explicit_implicit"] == "explicit" or row["scenario_id"] not in excluded
        )
        for row in cab_rows
    )
    for row in summaries:
        projected_rows = full_cab_per_cell if row["dataset"] == "CAB" else 1500
        row["projected_full_rows"] = projected_rows
        row["projected_full_seconds"] = row["seconds_per_row"] * projected_rows
    projected_by_model = {
        model: sum(
            row["projected_full_seconds"]
            for row in summaries
            if row["model_id"] == model
        )
        for model in config["models"]
    }
    report = {
        "schema_version": 1,
        "status": "ready_for_operator_review",
        "selection_manifest_sha256": _sha(
            (root / "manifests/pilot/selection.json").read_bytes()
        ),
        "run_count": len(summaries),
        "runs": summaries,
        "projected_full_seconds_by_model": projected_by_model,
        "checks": {
            "all_eight_cells_complete": len(summaries) == 8,
            "zero_errors": True,
            "same_gpu_class_within_model": True,
            "checkpoint_resume_covered_by_integration_test": True,
            "model_preflights_passed": True,
        },
        "review_packet_rows": len(review_rows),
        "decision": "pending_operator_output_review",
    }
    review_path = root / f"private_outputs/{RUN_DIRECTORY}/pilot_output_review.jsonl"
    _immutable(review_path, _jsonl_bytes(review_rows))
    report["review_packet_path"] = str(review_path.relative_to(root))
    report["review_packet_sha256"] = _sha(review_path.read_bytes())
    path = root / f"manifests/{RUN_DIRECTORY}/pilot_report.json"
    _immutable(path, _json_bytes(report))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run-model", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=("M1", "M2"))
    args = parser.parse_args(argv)
    if args.action == "plan":
        if args.model:
            raise PilotError("The plan action does not accept a model")
        result = build_selection(args.root)
    elif args.action == "run-model":
        if not args.model:
            raise PilotError("The run-model action requires --model M1 or M2")
        result = run_model(args.root, args.model)
    else:
        if args.model:
            raise PilotError("The report action does not accept a model")
        result = build_report(args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
