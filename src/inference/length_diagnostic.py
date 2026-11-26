"""Development-only check of a concise response instruction after the pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from inference.adapter import ModelAdapter, load_model
from inference.runner import git_state

INSTRUCTION = (
    " Answer in at most 150 words. Address the main request directly and include "
    "the most useful details within that limit."
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sample(rows: list[dict]) -> list[dict]:
    result = []
    for attribute in ("gender", "race"):
        options = [row for row in rows if row["attribute"] == attribute]
        options.sort(
            key=lambda row: (
                _sha(f"253|length-diagnostic|{row['prompt_id']}".encode()),
                row["prompt_id"],
            )
        )
        result.extend(options[:5])
    if len(result) != 10 or len({row["prompt_id"] for row in result}) != 10:
        raise ValueError("Diagnostic requires ten distinct CAB development prompts")
    return result


def run(root: Path, model_key: str) -> Path:
    if model_key not in {"M1", "M2"}:
        raise ValueError("Expected model M1 or M2")
    commit, dirty = git_state()
    if dirty:
        raise ValueError("Commit the diagnostic code before running it in Colab")
    selection_path = root / "manifests/pilot/selection.json"
    selection = json.loads(selection_path.read_text())
    prompt_path = root / selection["cab"]["path"]
    if _sha(prompt_path.read_bytes()) != selection["cab"]["sha256"]:
        raise ValueError("Frozen pilot CAB prompts changed")
    prompts = _sample(_rows(prompt_path))
    output = (
        root
        / "private_outputs/pilot_diagnostic"
        / f"concise150-{model_key.lower()}-{commit[:12]}.jsonl"
    )
    report = (
        root
        / "manifests/pilot_diagnostic"
        / f"concise150-{model_key.lower()}-{commit[:12]}.json"
    )
    if output.exists() or report.exists():
        if not (output.is_file() and report.is_file()):
            raise ValueError("Incomplete existing diagnostic artifacts")
        saved = json.loads(report.read_text())
        if (
            saved.get("code_commit") != commit
            or saved.get("output_sha256") != _sha(output.read_bytes())
            or saved.get("prompt_ids") != [row["prompt_id"] for row in prompts]
        ):
            raise ValueError("Existing diagnostic artifacts differ")
        return report

    loaded = load_model(model_key)
    diagnostic_config = copy.deepcopy(loaded.config)
    for condition in ("baseline", "mitigation"):
        diagnostic_config["conditions"][condition]["system_instruction"] += INSTRUCTION
    diagnostic = ModelAdapter(
        loaded.spec, loaded.tokenizer, loaded.model, diagnostic_config
    )
    results = []
    for condition in ("baseline", "mitigation"):
        for prompt in prompts:
            result = diagnostic.generate(prompt["prompt_text"], condition)
            results.append(
                {
                    "model_id": model_key,
                    "condition": condition,
                    "prompt_id": prompt["prompt_id"],
                    "scenario_id": prompt["scenario_id"],
                    "finish_reason": result["finish_reason"],
                    "output_token_count": result["output_token_count"],
                    "normalized_output": result["normalized_output"],
                    "code_commit": commit,
                    "purpose": "development_only_length_diagnostic_not_confirmatory",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results
        )
        + "\n"
    )
    summary = {
        "schema_version": 1,
        "purpose": "development_only_length_diagnostic_not_confirmatory",
        "model_id": model_key,
        "code_commit": commit,
        "instruction_suffix": INSTRUCTION,
        "selection_manifest_sha256": _sha(selection_path.read_bytes()),
        "prompt_ids": [row["prompt_id"] for row in prompts],
        "rows": len(results),
        "length_finishes": sum(row["finish_reason"] == "length" for row in results),
        "eos_finishes": sum(row["finish_reason"] == "eos" for row in results),
        "output_path": str(output.relative_to(root)),
        "output_sha256": _sha(output.read_bytes()),
    }
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("M1", "M2"))
    args = parser.parse_args(argv)
    path = run(args.root.expanduser().resolve(), args.model)
    print(path)
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
