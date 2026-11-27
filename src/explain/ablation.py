"""Model-agnostic BBQ input ablations against the frozen answer margin."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from scoring.bbq import format_question

FEATURES = (
    "identity_marker",
    "scenario_context",
    "question_wording",
    "answer_option_text",
    "neutral_function_words",
)
FUNCTION_WORDS = re.compile(r"\b(?:a|an|and|but|of|or|the|to)\b", re.IGNORECASE)


class AblationError(ValueError):
    """A selected item, score, or perturbation violates the explanation contract."""


def _variant_prompt(
    item: dict[str, Any], variant: dict, feature: str | None
) -> str | None:
    """Change only the displayed input; score the same original answer strings."""
    if feature not in (None, *FEATURES):
        raise AblationError(f"Unknown ablation feature: {feature}")
    changed = dict(item)
    if feature == "scenario_context":
        changed["context"] = "[context omitted]"
    elif feature == "question_wording":
        changed["question"] = "[question omitted]"
    elif feature == "neutral_function_words":
        changed["context"] = FUNCTION_WORDS.sub("", item["context"])
        changed["question"] = FUNCTION_WORDS.sub("", item["question"])
        if (changed["context"], changed["question"]) == (
            item["context"],
            item["question"],
        ):
            return None
    prompt = format_question(changed, variant["answers"])
    if feature == "identity_marker":
        marker = item.get("target_group")
        if not isinstance(marker, str) or not marker.strip():
            raise AblationError("BBQ target group is missing")
        pattern = re.compile(
            r"(?<!\w)" + re.escape(marker.strip()) + r"(?!\w)", re.IGNORECASE
        )
        changed, count = pattern.subn("[identity marker omitted]", prompt)
        return changed if count else None
    if feature == "answer_option_text":
        start = prompt.index("Options:\n") + len("Options:\n")
        end = prompt.index("Answer with the full text", start)
        prompt = (
            prompt[:start]
            + "A. [option A omitted]\nB. [option B omitted]\nC. [option C omitted]\n"
            + prompt[end:]
        )
    return prompt


def _source_scores_by_order(
    item: dict[str, Any],
    variants: list[dict],
    scorer: Callable[[str, str], float],
    feature: str | None,
    selected: int,
    runner_up: int,
) -> dict[int, tuple[float, float]] | None:
    if len(variants) != 3 or {row.get("order") for row in variants} != {0, 1, 2}:
        raise AblationError("Expected three cyclic BBQ orders")
    scores = {}
    for variant in sorted(variants, key=lambda row: row["order"]):
        prompt = _variant_prompt(item, variant, feature)
        if prompt is None:
            return None
        indices = variant["source_indices"]
        if sorted(indices) != [0, 1, 2] or variant["answers"] != [
            item[f"ans{index}"] for index in indices
        ]:
            raise AblationError("BBQ option mapping changed")
        pair = []
        for index in (selected, runner_up):
            answer = variant["answers"][indices.index(index)]
            value = float(scorer(prompt, answer))
            if not math.isfinite(value):
                raise AblationError("Non-finite answer likelihood")
            pair.append(value)
        scores[variant["order"]] = (pair[0], pair[1])
    return scores


def _mean_scores(scores: dict[int, tuple[float, float]]) -> tuple[float, float]:
    return tuple(sum(pair[index] for pair in scores.values()) / 3 for index in (0, 1))


def explain_item(
    item: dict[str, Any],
    variants: list[dict],
    frozen_result: dict[str, Any],
    scorer: Callable[[str, str], float],
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Measure how each input feature changes the original winner's margin."""
    if item["item_id"] != frozen_result["item_id"] or tolerance <= 0:
        raise AblationError("Explanation item or likelihood tolerance differs")
    means = frozen_result["mean_log_likelihood_by_source"]
    selected = frozen_result["predicted_source_index"]
    if len(means) != 3 or selected not in (0, 1, 2):
        raise AblationError("Invalid frozen BBQ answer scores")
    runner_up = max(
        (index for index in range(3) if index != selected),
        key=lambda index: (means[index], -index),
    )
    baseline_by_order = _source_scores_by_order(
        item, variants, scorer, None, selected, runner_up
    )
    baseline = _mean_scores(baseline_by_order)
    if any(
        abs(value - means[index]) > tolerance
        for value, index in zip(baseline, (selected, runner_up), strict=True)
    ):
        raise AblationError("Original answer likelihoods do not reproduce the full run")
    baseline_margin = baseline[0] - baseline[1]
    effects = []
    order_effects = {order: {} for order in range(3)}
    for feature in FEATURES:
        changed_by_order = _source_scores_by_order(
            item, variants, scorer, feature, selected, runner_up
        )
        changed = (
            _mean_scores(changed_by_order) if changed_by_order is not None else None
        )
        for order in range(3):
            order_effects[order][feature] = (
                baseline_by_order[order][0]
                - baseline_by_order[order][1]
                - changed_by_order[order][0]
                + changed_by_order[order][1]
                if changed_by_order is not None
                else None
            )
        effects.append(
            {
                "feature": feature,
                "status": "measured" if changed is not None else "not_present",
                "ablated_margin": changed[0] - changed[1]
                if changed is not None
                else None,
                "margin_drop": baseline_margin - (changed[0] - changed[1])
                if changed is not None
                else None,
            }
        )
    return {
        "item_id": item["item_id"],
        "selected_source_index": selected,
        "runner_up_source_index": runner_up,
        "baseline_margin": baseline_margin,
        "effects": effects,
        "order_effects": order_effects,
        "method": "section_and_exact_target_group_ablation_v1",
    }
