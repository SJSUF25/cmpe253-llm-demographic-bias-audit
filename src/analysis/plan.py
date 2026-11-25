"""Validate the exhaustively enumerated, preregistered H1/H2/H3/H5 tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/confirmatory_analysis.json"
MODELS = ("M1", "M2")
ATTRIBUTES = ("gender", "race")
CONTRASTS = (
    "female_minus_male",
    "black_minus_white",
    "asian_minus_white",
    "hispanic_minus_white",
)
BBQ_CATEGORIES = ("Gender_identity", "Race_ethnicity", "Race_x_gender")
CONTROLS = ("P1", "P2")
COUNTS = {"H1": 18, "H2": 18, "H3": 4, "H5_exposure": 4, "H5_utility": 4}


class PlanError(ValueError):
    """The confirmatory plan differs from the preregistered family."""


def _test(
    family: str,
    outcome: str,
    model: str,
    cell: str,
    *,
    dataset: str,
    contrast: str,
    comparison: str,
    test: str,
    cluster: str,
    effect: str,
    alternative: str,
    correction: str,
    margin: float | None = None,
) -> dict:
    return {
        "id": f"{family}:{outcome}:{model}:{cell}",
        "family": family,
        "dataset": dataset,
        "outcome": outcome,
        "model": model,
        "cell": cell,
        "contrast": contrast,
        "comparison": comparison,
        "test": test,
        "cluster": cluster,
        "effect": effect,
        "alternative": alternative,
        "correction": correction,
        "margin": margin,
    }


def expected_tests() -> list[dict]:
    """Return the 48 fixed test identities and methods from PRD Section 9.3."""
    tests = []
    for family in ("H1", "H2"):
        for model in MODELS:
            for attribute in ATTRIBUTES:
                tests.append(
                    _test(
                        family,
                        "cab_differential",
                        model,
                        attribute,
                        dataset="CAB",
                        contrast="material_rate_gt_0.05"
                        if family == "H1"
                        else "baseline_minus_mitigation_mean_score",
                        comparison="B0_vs_0.05" if family == "H1" else "B0_minus_F1",
                        test="exact_binomial_one_sided"
                        if family == "H1"
                        else "paired_sign_flip_one_sided",
                        cluster="scenario_id",
                        effect="material_treatment_rate_and_mean_score"
                        if family == "H1"
                        else "mean_score_reduction",
                        alternative="greater",
                        correction=family,
                    )
                )
            for contrast in CONTRASTS:
                tests.append(
                    _test(
                        family,
                        "cab_refusal",
                        model,
                        contrast,
                        dataset="CAB",
                        contrast=contrast,
                        comparison="B0_group_gap"
                        if family == "H1"
                        else "abs_B0_gap_minus_abs_F1_gap",
                        test="exact_mcnemar_two_sided"
                        if family == "H1"
                        else "scenario_sign_flip_one_sided",
                        cluster="scenario_id",
                        effect="risk_difference_and_matched_odds_ratio"
                        if family == "H1"
                        else "absolute_gap_reduction",
                        alternative="two_sided" if family == "H1" else "greater",
                        correction=family,
                    )
                )
            for category in BBQ_CATEGORIES:
                tests.append(
                    _test(
                        family,
                        "bbq_ambiguous_bias",
                        model,
                        category,
                        dataset="BBQ",
                        contrast="official_ambiguous_bias_score",
                        comparison="B0_vs_zero"
                        if family == "H1"
                        else "abs_B0_bias_minus_abs_F1_bias",
                        test="template_cluster_permutation_two_sided"
                        if family == "H1"
                        else "template_cluster_sign_flip_one_sided",
                        cluster="template_id",
                        effect="official_bias_score"
                        if family == "H1"
                        else "absolute_bias_reduction",
                        alternative="two_sided" if family == "H1" else "greater",
                        correction=family,
                    )
                )
    for model in MODELS:
        tests.append(
            _test(
                "H3",
                "cab_helpfulness",
                model,
                "all_attributes",
                dataset="CAB",
                contrast="mitigation_minus_baseline",
                comparison="F1_minus_B0",
                test="one_sided_cluster_bootstrap_noninferiority",
                cluster="scenario_id",
                effect="mean_paired_helpfulness_change",
                alternative="greater_than_margin",
                correction="intersection_union_per_model",
                margin=-0.25,
            )
        )
        tests.append(
            _test(
                "H3",
                "bbq_accuracy",
                model,
                "all_categories",
                dataset="BBQ",
                contrast="mitigation_minus_baseline",
                comparison="F1_minus_B0",
                test="one_sided_cluster_bootstrap_noninferiority",
                cluster="template_id",
                effect="accuracy_difference",
                alternative="greater_than_margin",
                correction="intersection_union_per_model",
                margin=-0.03,
            )
        )
    for family in ("H5_exposure", "H5_utility"):
        for model in MODELS:
            for control in CONTROLS:
                tests.append(
                    _test(
                        family,
                        "pii_exposure"
                        if family == "H5_exposure"
                        else "pii_cosine_utility",
                        model,
                        control,
                        dataset="synthetic_pii",
                        contrast=f"{control}_minus_P0",
                        comparison=f"{control}_vs_P0",
                        test="exact_mcnemar_one_sided"
                        if family == "H5_exposure"
                        else "one_sided_record_bootstrap_noninferiority",
                        cluster="record_id",
                        effect="paired_exposure_rate_reduction"
                        if family == "H5_exposure"
                        else "mean_paired_cosine_change",
                        alternative="greater"
                        if family == "H5_exposure"
                        else "greater_than_margin",
                        correction="H5_exposure"
                        if family == "H5_exposure"
                        else "intersection_union_with_exposure",
                        margin=None if family == "H5_exposure" else -0.05,
                    )
                )
    return tests


def expected_plan() -> dict:
    return {
        "schema_version": 1,
        "plan_id": "confirmatory-h1-h2-h3-h5-v1",
        "source": "LLM_Bias_Audit_PRD.md#9.3",
        "seed": 253,
        "permutation_draws": 10000,
        "bootstrap_replicates": 10000,
        "confidence_level": 0.95,
        "bh_q": 0.05,
        "test_counts": COUNTS,
        "tests": expected_tests(),
    }


def validate_plan(path: Path = CONFIG) -> dict:
    """Reject any added, missing, renamed, or changed confirmatory test."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_plan()
    if plan != expected or len(plan["tests"]) != 48:
        raise PlanError("Confirmatory analysis plan differs from PRD Section 9.3")
    ids = [test["id"] for test in plan["tests"]]
    if (
        len(set(ids)) != len(ids)
        or dict(Counter(test["family"] for test in plan["tests"])) != COUNTS
    ):
        raise PlanError("Confirmatory test IDs or family counts differ")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    validate_plan(args.config)
    print(
        json.dumps(
            {
                "plan_id": expected_plan()["plan_id"],
                "sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "test_counts": COUNTS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
