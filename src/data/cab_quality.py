"""Freeze a source-preserving CAB implicit-prompt exclusion before inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from data.cab_builder import DEFAULT_CONFIG, build_cab

MANIFEST = Path("manifests/cab_implicit_quality_manifest.json")
RULE = "implicit_has_newline_or_length_gt_135_percent_of_explicit_v1"
EXPECTED_EXCLUDED_FAMILIES = 23
EXPECTED_ELIGIBLE_IMPLICIT_PROMPTS = 736


class CabQualityError(ValueError):
    """The implicit-prompt quality policy cannot be verified."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_manifest(prompt_bytes: bytes, source_sha256: str) -> dict:
    """Apply the rule learned from development data to every family, without outputs."""
    rows = [json.loads(line) for line in prompt_bytes.splitlines() if line]
    families: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[row["scenario_id"]][row["explicit_implicit"]].append(row)
    excluded = []
    counts: Counter[str] = Counter()
    for scenario_id, modes in sorted(families.items()):
        explicit = modes["explicit"]
        implicit = modes["implicit"]
        if not explicit or not implicit or len(explicit) != len(implicit):
            raise CabQualityError(f"Incomplete CAB family: {scenario_id}")
        reference = explicit[0]
        candidate = implicit[0]
        explicit_by_group = {row["group"]: row for row in explicit}
        if len(explicit_by_group) != len(explicit):
            raise CabQualityError(f"Duplicate demographic variant: {scenario_id}")
        flagged = "\n" in candidate["prompt_text"] or (
            len(candidate["prompt_text"]) * 100
            > len(explicit_by_group[candidate["group"]]["prompt_text"]) * 135
        )
        # Each demographic variant must have the same quality disposition.
        if any(
            (
                "\n" in row["prompt_text"]
                or len(row["prompt_text"]) * 100
                > len(explicit_by_group[row["group"]]["prompt_text"]) * 135
            )
            != flagged
            for row in implicit
        ):
            raise CabQualityError(f"Inconsistent variants: {scenario_id}")
        if flagged:
            excluded.append(scenario_id)
            counts[f"{reference['split']}|{reference['attribute']}"] += 1
    excluded_set = set(excluded)
    eligible = sum(
        len(modes["implicit"])
        for scenario_id, modes in families.items()
        if scenario_id not in excluded_set
    )
    if (len(excluded), eligible) != (
        EXPECTED_EXCLUDED_FAMILIES,
        EXPECTED_ELIGIBLE_IMPLICIT_PROMPTS,
    ):
        raise CabQualityError("Pinned CAB quality counts changed; review source/rule")
    return {
        "schema_version": 1,
        "status": "passed_with_implicit_exclusions",
        "rule": RULE,
        "source_sha256": source_sha256,
        "prompt_manifest_sha256": _sha(prompt_bytes),
        "excluded_implicit_scenario_ids": excluded,
        "excluded_family_counts": dict(sorted(counts.items())),
        "eligible_implicit_prompt_count": eligible,
        "explicit_prompt_count": sum(len(m["explicit"]) for m in families.values()),
    }


def verify_manifest(root: Path) -> dict:
    """Reject a missing, stale, or edited quality manifest."""
    root = root.expanduser().resolve()
    path = root / MANIFEST
    if not path.is_file():
        raise CabQualityError("Missing CAB implicit quality manifest")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source = root / config["input_path"]
    prompts = root / config["output_path"]
    if not source.is_file() or not prompts.is_file():
        raise CabQualityError("Missing CAB source or prompt manifest")
    source_sha = _sha(source.read_bytes())
    if source_sha != config["input_sha256"]:
        raise CabQualityError("Pinned CAB source hash changed")
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = make_manifest(prompts.read_bytes(), source_sha)
    if actual != expected:
        raise CabQualityError("CAB implicit quality manifest is stale or altered")
    return actual


def write_manifest(root: Path) -> Path:
    """Write once, then verify the immutable quality decision."""
    root = root.expanduser().resolve()
    build_cab(root)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    source_sha = _sha((root / config["input_path"]).read_bytes())
    manifest = make_manifest((root / config["output_path"]).read_bytes(), source_sha)
    path = root / MANIFEST
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CabQualityError(f"Existing quality manifest differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_manifest(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
