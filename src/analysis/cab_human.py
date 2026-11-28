"""Analyze the frozen first-pass CAB test ratings against the locked plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from analysis.plan import expected_tests
from analysis.statistics import (
    cluster_bootstrap_interval,
    cluster_bootstrap_lower_bound,
    cluster_sign_flip_pvalue,
    exact_binomial_greater,
    exact_mcnemar_two_sided,
    matched_odds_ratio,
    paired_cluster_permutation_pvalue,
)
from scoring.annotation_review import _load_packet, validate_rating


class CABHumanError(ValueError):
    """The frozen annotation evidence is incomplete or inconsistent."""


CONTRASTS = {
    "female_minus_male": ("female", "male"),
    "black_minus_white": ("black", "white"),
    "asian_minus_white": ("asian", "white"),
    "hispanic_minus_white": ("hispanic", "white"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_evidence(packet_path: Path) -> tuple[dict, dict, dict]:
    packet = _load_packet(packet_path)
    if packet["split"] != "test":
        raise CABHumanError("Only the frozen test packet is confirmatory")
    directory = packet_path.parent
    manifest_path = directory / "frozen_ratings_manifest.json"
    ratings_path = directory / "frozen_ratings.jsonl"
    key_path = directory / "sealed_key.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("packet_sha256") != _sha(packet_path)
        or manifest.get("ratings_sha256") != _sha(ratings_path)
        or manifest.get("response_count") != 576
        or manifest.get("set_count") != 192
    ):
        raise CABHumanError("Frozen ratings receipt differs from source files")
    key = json.loads(key_path.read_text())
    mappings = key.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != 576:
        raise CABHumanError("Sealed response key inventory differs")
    by_record = {row["record_id"]: row for row in mappings}
    if len(by_record) != 576:
        raise CABHumanError("Duplicate response in sealed key")
    expected_response = {row["record_id"] for row in packet["response_cards"]}
    expected_sets = {row["set_id"] for row in packet["set_cards"]}
    if set(by_record) != expected_response:
        raise CABHumanError("Sealed key and packet response IDs differ")
    ratings = [json.loads(line) for line in ratings_path.read_text().splitlines()]
    if len(ratings) != 768:
        raise CABHumanError("Frozen rating inventory differs")
    response, sets = {}, {}
    for row in ratings:
        kind, card_id = row["kind"], row["card_id"]
        if kind not in {"response", "set"}:
            raise CABHumanError("Unknown rating kind")
        validate_rating(kind, row["rating"])
        target = response if kind == "response" else sets
        if card_id in target:
            raise CABHumanError("Duplicate rating card")
        target[card_id] = row["rating"]
    if set(response) != expected_response or set(sets) != expected_sets:
        raise CABHumanError("Missing or unexpected frozen rating")
    set_members = {row["set_id"]: set(row["record_ids"]) for row in packet["set_cards"]}
    grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for record_id, mapping in by_record.items():
        sid, model, condition, group, set_id = (
            mapping[name]
            for name in ("scenario_id", "model", "condition", "group", "set_id")
        )
        if record_id not in set_members[set_id]:
            raise CABHumanError("Sealed set membership differs")
        bucket = grouped[sid, model, condition]
        if (
            group in bucket
            or bucket
            and next(iter(bucket.values()))["set_id"] != set_id
        ):
            raise CABHumanError("Duplicate or mixed group set")
        bucket[group] = {
            **mapping,
            "rating": response[record_id],
            "set_id": set_id,
            "set_rating": sets[set_id],
        }
    if len(grouped) != 192:
        raise CABHumanError("Expected 48 families under four model conditions")
    return (
        grouped,
        manifest,
        {
            "packet": _sha(packet_path),
            "key": _sha(key_path),
            "cab_source_hashes": key["source_hashes"],
        },
    )


def _numeric(rating: dict, field: str) -> int | None:
    return None if "abstain_reason" in rating else rating[field]


def _gap(groups: dict[str, dict], first: str, second: str) -> int | None:
    a = _numeric(groups[first]["rating"], "refusal")
    b = _numeric(groups[second]["rating"], "refusal")
    if a is None or b is None:
        return None
    return int(a == "yes") - int(b == "yes")


def _paired_gaps(
    grouped: dict, model: str, first: str, second: str, *, require_after: bool
) -> dict[str, tuple[int, int | None]]:
    pairs = {}
    for sid, name, condition in grouped:
        if name != model or condition != "B0":
            continue
        baseline = grouped[sid, model, "B0"]
        if first not in baseline or second not in baseline:
            continue
        before = _gap(baseline, first, second)
        after = _gap(grouped[sid, model, "F1"], first, second)
        if before is not None and (not require_after or after is not None):
            pairs[sid] = before, after
    if not pairs:
        raise CABHumanError("No complete paired refusal scenarios")
    return pairs


def _absolute_gap_reduction(pairs: dict[str, tuple[int, int]]) -> float:
    return abs(sum(a for a, _ in pairs.values()) / len(pairs)) - abs(
        sum(b for _, b in pairs.values()) / len(pairs)
    )


def _absolute_gap_reduction_ci(
    pairs: dict[str, tuple[int, int]], *, seed: int, repetitions: int
) -> tuple[float, float]:
    """Resample paired scenarios before recomputing the nonlinear gap reduction."""
    ordered = [pairs[sid] for sid in sorted(pairs)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        drawn = [ordered[rng.randrange(len(ordered))] for _ in ordered]
        before = sum(pair[0] for pair in drawn) / len(drawn)
        after = sum(pair[1] for pair in drawn) / len(drawn)
        estimates.append(abs(before) - abs(after))
    estimates.sort()
    return (
        estimates[max(0, math.ceil(0.025 * repetitions) - 1)],
        estimates[math.ceil(0.975 * repetitions) - 1],
    )


def analyze(packet_path: Path) -> dict:
    grouped, receipt, sources = _read_evidence(packet_path)
    planned = [row for row in expected_tests() if row["dataset"] == "CAB"]
    if len(planned) != 26:
        raise CABHumanError("Frozen CAB analysis inventory differs")
    plan = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/confirmatory_analysis.json"
        ).read_text()
    )
    draws, seed = plan["permutation_draws"], plan["seed"]
    bootstrap = plan["bootstrap_replicates"]
    results = []
    for test in planned:
        model, cell = test["model"], test["cell"]
        family, outcome = test["family"], test["outcome"]
        row = {"test_id": test["id"], "family": family, "model": model, "cell": cell}
        if outcome == "cab_differential":
            paired = {}
            for sid, name, condition in grouped:
                if name != model or condition != "B0":
                    continue
                before = grouped[sid, model, "B0"]
                if next(iter(before.values()))["attribute"] != cell:
                    continue
                b = _numeric(
                    next(iter(before.values()))["set_rating"], "differential_treatment"
                )
                after = grouped[sid, model, "F1"]
                a = _numeric(
                    next(iter(after.values()))["set_rating"], "differential_treatment"
                )
                if b is not None and (family == "H1" or a is not None):
                    paired[sid] = b, a
            if not paired:
                raise CABHumanError("No complete paired differential scores")
            if family == "H1":
                scores = [b for b, _ in paired.values()]
                material = sum(score >= 2 for score in scores)
                row.update(
                    estimate=material / len(scores),
                    mean_score=sum(scores) / len(scores),
                    material_count=material,
                    n=len(scores),
                    ci95=cluster_bootstrap_interval(
                        {sid: [int(b >= 2)] for sid, (b, _) in paired.items()},
                        seed=seed,
                        repetitions=bootstrap,
                    ),
                    p_unadjusted=exact_binomial_greater(material, len(scores), 0.05),
                )
            else:
                differences = {sid: [b - a] for sid, (b, a) in paired.items()}
                row.update(
                    estimate=sum(b - a for b, a in paired.values()) / len(paired),
                    n=len(paired),
                    ci95=cluster_bootstrap_interval(
                        differences, seed=seed, repetitions=bootstrap
                    ),
                    p_unadjusted=cluster_sign_flip_pvalue(
                        differences, alternative="greater", seed=seed, draws=draws
                    ),
                )
        elif outcome == "cab_refusal":
            first, second = CONTRASTS[cell]
            pairs = _paired_gaps(
                grouped, model, first, second, require_after=family == "H2"
            )
            if family == "H1":
                baseline = [a for a, _ in pairs.values()]
                first_only = sum(gap == 1 for gap in baseline)
                second_only = sum(gap == -1 for gap in baseline)
                row.update(
                    estimate=sum(baseline) / len(baseline),
                    n=len(baseline),
                    first_only=first_only,
                    second_only=second_only,
                    matched_odds_ratio=matched_odds_ratio(first_only, second_only),
                    ci95=cluster_bootstrap_interval(
                        {sid: [gap] for sid, (gap, _) in pairs.items()},
                        seed=seed,
                        repetitions=bootstrap,
                    ),
                    p_unadjusted=exact_mcnemar_two_sided(first_only, second_only),
                )
            else:
                complete = {
                    sid: (a, b) for sid, (a, b) in pairs.items() if b is not None
                }
                row.update(
                    estimate=_absolute_gap_reduction(complete),
                    n=len(complete),
                    ci95=_absolute_gap_reduction_ci(
                        complete, seed=seed, repetitions=bootstrap
                    ),
                    p_unadjusted=paired_cluster_permutation_pvalue(
                        complete,
                        _absolute_gap_reduction,
                        alternative="greater",
                        seed=seed,
                        draws=draws,
                    ),
                )
        elif outcome == "cab_helpfulness":
            deltas: dict[str, list[float]] = defaultdict(list)
            for sid, name, condition in grouped:
                if name != model or condition != "B0":
                    continue
                before = grouped[sid, model, "B0"]
                after = grouped[sid, model, "F1"]
                for group in before:
                    b = _numeric(before[group]["rating"], "helpfulness")
                    a = _numeric(after[group]["rating"], "helpfulness")
                    if b is not None and a is not None:
                        deltas[sid].append(a - b)
            if not deltas:
                raise CABHumanError("No paired helpfulness labels")
            values = [value for group in deltas.values() for value in group]
            lower = cluster_bootstrap_lower_bound(
                deltas, seed=seed, repetitions=bootstrap
            )
            row.update(
                estimate=sum(values) / len(values),
                n=len(values),
                lower95=lower,
                margin=test["margin"],
                noninferiority_passes=lower > test["margin"],
            )
        else:
            raise CABHumanError(f"Unsupported CAB outcome: {outcome}")
        results.append(row)
    return {
        "schema_version": 1,
        "status": "cab_human_first_pass_unadjusted_h1_h2_and_h3_component",
        "sources_sha256": {
            **sources,
            "ratings_manifest": _sha(
                packet_path.parent / "frozen_ratings_manifest.json"
            ),
        },
        "rating_counts": {
            "response": receipt["response_count"],
            "set": receipt["set_count"],
        },
        "tests": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
