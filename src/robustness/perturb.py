"""Build and verify the frozen neutral-format and adversarial CAB perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from data.cab_builder import DEFAULT_CONFIG
from inference import runner
from robustness.sample import ADVERSARIAL_SUFFIX, RobustnessSampleError, freeze


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(text: str) -> str:
    return " ".join(text.split())


def variants(prompt: dict) -> list[dict]:
    """Keep original identity text; change only whitespace or append frozen attack."""
    text = prompt["prompt_text"]
    if not isinstance(text, str) or not text.strip():
        raise RobustnessSampleError("Empty CAB robustness prompt")
    whitespace = "\n" + re.sub(r" +", "  ", text) + "\n"
    if _canonical(whitespace) != _canonical(text):
        raise RobustnessSampleError("Formatting changed CAB prompt tokens")
    attack = text.rstrip() + "\n\n" + ADVERSARIAL_SUFFIX
    if not attack.startswith(text.rstrip() + "\n\n"):
        raise RobustnessSampleError("Adversarial prompt changed original text")
    result = []
    for kind, value in (
        ("original", text),
        ("whitespace", whitespace),
        ("adversarial", attack),
    ):
        result.append(
            {
                "scenario_id": prompt["scenario_id"],
                "source_prompt_id": prompt["prompt_id"],
                "attribute": prompt["attribute"],
                "group": prompt["group"],
                "superdomain": prompt["superdomain"],
                "kind": kind,
                "prompt_id": _sha(
                    f"253|robustness60|{kind}|{prompt['prompt_id']}".encode()
                )[:20],
                "prompt_text": value,
                "prompt_sha256": _sha(value.encode()),
                "source_prompt_sha256": prompt["prompt_sha256"],
            }
        )
    return result


def build(root: Path) -> Path:
    root = root.expanduser().resolve()
    sample_path = freeze(root)
    sample = json.loads(sample_path.read_text())
    selected = {entry["scenario_id"] for entry in sample["selected_families"]}
    config = json.loads(DEFAULT_CONFIG.read_text())
    prompt_path = root / config["output_path"]
    prompts = [
        row
        for row in runner._read_jsonl(prompt_path)
        if row["scenario_id"] in selected
        and row["split"] == "test"
        and row["explicit_implicit"] == "explicit"
    ]
    if len({row["scenario_id"] for row in prompts}) != 60 or len(prompts) != 180:
        raise RobustnessSampleError("Expected 180 explicit prompts in 60 families")
    output_rows = [row for prompt in prompts for row in variants(prompt)]
    if len(output_rows) != 540 or len({row["prompt_id"] for row in output_rows}) != 540:
        raise RobustnessSampleError("Expected 540 distinct robustness variants")
    output_path = root / "private_outputs/processed_data/robustness/perturbations.jsonl"
    content = (
        b"\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            for row in output_rows
        )
        + b"\n"
    )
    if output_path.exists():
        if output_path.read_bytes() != content:
            raise RobustnessSampleError("Existing robustness prompts differ")
    else:
        runner._atomic_write(output_path, content)
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_robustness_outputs",
        "sample_sha256": _sha(sample_path.read_bytes()),
        "cab_prompt_sha256": _sha(prompt_path.read_bytes()),
        "output_path": str(output_path.relative_to(root)),
        "output_sha256": _sha(content),
        "families": 60,
        "source_prompts": 180,
        "variant_rows": 540,
        "kinds": ["original", "whitespace", "adversarial"],
        "paraphrase_status": "pending_semantic_validation",
        "identity_clause_position_status": "pending_source_specific_validation",
    }
    manifest_path = root / "manifests/robustness/perturbations.json"
    manifest_content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_content:
            raise RobustnessSampleError("Existing robustness manifest differs")
    else:
        runner._atomic_write(manifest_path, manifest_content)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
