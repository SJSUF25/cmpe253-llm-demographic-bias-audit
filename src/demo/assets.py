"""Export an allowlisted, text-free cached bundle from verified aggregate results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class DemoAssetError(ValueError):
    """An aggregate artifact is missing, malformed, or unsafe for the demo."""


SOURCES = {
    "fairness": (
        "aggregate_results/fairness_descriptive.json",
        {
            "bbq_cells": (
                "model",
                "condition",
                "category",
                "context",
                "items",
                "templates",
                "accuracy",
                "accuracy_ci95",
                "bias_score_points",
                "bias_ci95",
                "accuracy_change",
                "accuracy_change_ci95",
                "absolute_bias_reduction",
                "absolute_bias_reduction_ci95",
            ),
            "bbq_accuracy_gate_components": (
                "model",
                "items",
                "templates",
                "accuracy_change",
                "one_sided_lower95",
                "margin",
                "accuracy_component_passes",
                "joint_h3_status",
            ),
            "cab_refusal_cells": (
                "model",
                "condition",
                "mode",
                "attribute",
                "contrast",
                "families",
                "refusal_gap",
                "absolute_refusal_gap_reduction",
                "ci95",
                "rule_status",
            ),
            "cab_identity_signal_shift": (
                "model",
                "condition",
                "attribute",
                "contrast",
                "common_families",
                "explicit_gap",
                "implicit_gap",
                "implicit_minus_explicit",
                "change_ci95",
                "direction_agreement_when_both_nonzero",
                "rule_status",
            ),
        },
    ),
    "cab_secondary": (
        "aggregate_results/cab_secondary_summary.json",
        {
            "group_cells": (
                "model",
                "condition",
                "mode",
                "attribute",
                "group",
                "rows",
                "mean_toxicity_probability",
                "mean_identity_attack_probability",
                "mean_sentiment_positive_probability",
                "mean_sentiment_negative_probability",
                "mean_response_words",
            ),
            "pair_cells": (
                "model",
                "condition",
                "mode",
                "attribute",
                "contrast",
                "pairs",
                "mean_semantic_similarity",
                "mean_response_word_difference",
            ),
            "truncation": (
                "model",
                "condition",
                "mode",
                "rows",
                "truncated_scorer_inputs",
            ),
        },
    ),
    "explanation": (
        "aggregate_results/explain_summary.json",
        {
            "cells": (
                "model",
                "condition",
                "category",
                "feature",
                "items",
                "measured_items",
                "not_present_items",
                "mean_margin_drop",
                "ci95",
            ),
            "answer_order_stability": (
                "model",
                "condition",
                "category",
                "order_pair",
                "items",
                "spearman",
                "top2_overlap",
            ),
        },
    ),
    "robustness": (
        "aggregate_results/robustness_refusal.json",
        {
            "cells": (
                "model",
                "condition",
                "kind",
                "attribute",
                "contrast",
                "families",
                "original_refusal_gap",
                "perturbed_refusal_gap",
                "gap_change",
                "gap_change_ci95",
            ),
        },
    ),
    "privacy": (
        "aggregate_results/privacy_exposure_summary.json",
        {
            "cells": (
                "model",
                "condition",
                "task",
                "records",
                "exact_record_exposures",
                "normalized_record_exposures",
                "exact_by_type",
                "normalized_by_type",
            ),
        },
    ),
    "privacy_utility": (
        "aggregate_results/privacy_h5.json",
        {
            "cells": (
                "baseline_exposed",
                "control_exposed",
                "improved_pairs",
                "worsened_pairs",
                "exposure_rate_reduction",
                "exposure_p_one_sided",
                "mean_cosine_to_normalized_p0",
                "mean_utility_delta",
                "utility_delta_lower95",
                "exposure_q_bh4",
                "exposure_pass",
                "utility_pass",
                "joint_h5_pass",
            ),
        },
    ),
    "canary_base": ("aggregate_results/canary_attack_base.json", {}),
    "canary_adapter": ("aggregate_results/canary_attack_adapter.json", {}),
    "adapter_utility": ("aggregate_results/adapter_utility_comparison.json", {}),
}
MIXED_REQUIRED = {
    ("fairness", "bbq_cells"): (
        "model",
        "condition",
        "category",
        "context",
        "items",
        "templates",
    ),
    ("fairness", "cab_refusal_cells"): (
        "model",
        "condition",
        "mode",
        "attribute",
        "contrast",
        "families",
        "ci95",
    ),
}


def _safe_label(value: str) -> str:
    if len(value) > 80 or any(char in value for char in ("\n", "@", "/", "\\")):
        raise DemoAssetError("Aggregate label is unsuitable for a public bundle")
    return value


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DemoAssetError("Non-finite aggregate metric")
        return value
    if isinstance(value, str):
        return _safe_label(value)
    if isinstance(value, list):
        if len(value) > 8:
            raise DemoAssetError("Aggregate list is too large")
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        if len(value) > 20:
            raise DemoAssetError("Aggregate mapping is too large")
        return {_safe_label(str(key)): _safe_value(item) for key, item in value.items()}
    raise DemoAssetError("Unsupported aggregate field type")


def _project_rows(
    rows: Any, fields: tuple[str, ...], *, required: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise DemoAssetError("Aggregate table is empty or malformed")
    projected = []
    for row in rows:
        if not isinstance(row, dict):
            raise DemoAssetError("Aggregate row is malformed")
        missing = set(required if required is not None else fields) - set(row)
        if missing:
            raise DemoAssetError(f"Aggregate row lacks {sorted(missing)}")
        projected.append({field: _safe_value(row.get(field)) for field in fields})
    return projected


def _canary_summary(artifact: dict, mode: str) -> dict:
    if artifact.get("mode") != mode:
        raise DemoAssetError(f"Canary attack mode differs: {mode}")
    scores = artifact.get("score_attacks")
    extraction = artifact.get("member_extraction_by_repetition")
    if not isinstance(scores, dict) or set(scores) != {"negative_loss", "min_k20"}:
        raise DemoAssetError("Canary score inventory differs")
    if not isinstance(extraction, dict) or set(extraction) != {"1", "5", "10"}:
        raise DemoAssetError("Canary repetition inventory differs")
    score_rows = []
    for name in ("negative_loss", "min_k20"):
        score = scores[name]
        if score.get("calibration_false_positives", 501) > 5:
            raise DemoAssetError("Canary calibration exceeds the fixed bound")
        score_rows.append(
            {
                "score": name,
                **_project_rows(
                    [score],
                    (
                        "auc",
                        "tpr_at_calibrated_1pct_fpr",
                        "evaluation_fpr",
                        "calibration_false_positives",
                    ),
                )[0],
                "auc_ci95": _safe_value(score["ci95"]["auc"]),
                "tpr_ci95": _safe_value(score["ci95"]["tpr"]),
                "fpr_ci95": _safe_value(score["ci95"]["fpr"]),
            }
        )
    extraction_rows = [
        {
            "presentations": int(repetition),
            **_project_rows([extraction[repetition]], ("records", "exact_rate"))[0],
            "exact_rate_ci95": _safe_value(extraction[repetition]["ci95"]),
        }
        for repetition in ("1", "5", "10")
    ]
    if any(row["records"] != 100 for row in extraction_rows):
        raise DemoAssetError("Canary repetition sample size differs")
    return {"mode": mode, "scores": score_rows, "extraction": extraction_rows}


def _adapter_utility_summary(artifact: dict) -> dict:
    overall = artifact.get("overall")
    tasks = artifact.get("tasks")
    if not isinstance(overall, dict) or overall.get("records") != 400:
        raise DemoAssetError("Adapter utility inventory differs")
    if not isinstance(tasks, dict) or set(tasks) != {
        "addition",
        "copy_code",
        "json_field",
        "uppercase",
    }:
        raise DemoAssetError("Adapter utility task inventory differs")
    fields = (
        "records",
        "base_exact_accuracy",
        "adapter_exact_accuracy",
        "adapter_minus_base_exact_accuracy",
        "base_mean_target_loss",
        "adapter_mean_target_loss",
        "adapter_minus_base_target_loss",
    )
    rows = [{"task": "overall", **_project_rows([overall], fields)[0]}]
    rows.extend(
        {"task": task, **_project_rows([tasks[task]], fields)[0]}
        for task in sorted(tasks)
    )
    if any(row["records"] != 100 for row in rows[1:]):
        raise DemoAssetError("Adapter utility task sample size differs")
    return {"cells": rows}


def build_bundle(root: Path) -> Path:
    """Read only aggregate JSON, copy named metrics, and exclude every raw output."""
    root = root.expanduser().resolve()
    sections: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for section, (relative, tables) in SOURCES.items():
        path = root / relative
        if not path.is_file():
            if section == "fairness":
                raise DemoAssetError(
                    "Fairness aggregate is required before demo export"
                )
            continue
        raw = path.read_bytes()
        artifact = json.loads(raw)
        if artifact.get("schema_version") != 1:
            raise DemoAssetError(f"Unsupported aggregate schema: {section}")
        if section in ("canary_base", "canary_adapter"):
            sections[section] = _canary_summary(
                artifact, section.removeprefix("canary_")
            )
        elif section == "adapter_utility":
            sections[section] = _adapter_utility_summary(artifact)
        elif section == "privacy_utility":
            cells = artifact.get("cells")
            expected = {
                f"{model}:{condition}"
                for model in ("M1", "M2")
                for condition in ("P1", "P2")
            }
            if not isinstance(cells, dict) or set(cells) != expected:
                raise DemoAssetError("Privacy utility inventory differs")
            sections[section] = {
                "cells": [
                    {"cell": key, **_project_rows([cells[key]], tables["cells"])[0]}
                    for key in sorted(cells)
                ]
            }
        else:
            sections[section] = {
                table: _project_rows(
                    artifact.get(table),
                    fields,
                    required=MIXED_REQUIRED.get((section, table)),
                )
                for table, fields in tables.items()
            }
        sources[section] = hashlib.sha256(raw).hexdigest()
    if len(sections["fairness"]["bbq_cells"]) != 36:
        raise DemoAssetError("Fairness BBQ cell inventory differs")
    bundle = {
        "schema_version": 1,
        "mode": "cached_offline",
        "source_sha256": sources,
        "sections": sections,
        "limitations": [
            "Human differential-treatment and helpfulness ratings are pending.",
            "The anchored refusal rule has not been calibrated against human labels.",
            "Missing views indicate experiments that have not completed.",
        ],
    }
    output = root / "demo_assets/bundle.json"
    content = (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    receipt = {
        "schema_version": 1,
        "status": "partial_aggregate_only",
        "bundle_sha256": hashlib.sha256(content).hexdigest(),
        "source_sha256": sources,
        "included_sections": sorted(sections),
        "pending_sections": sorted(set(SOURCES) - set(sections)),
        "reviewed_examples_included": False,
    }
    (root / "demo_assets/manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_bundle(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
