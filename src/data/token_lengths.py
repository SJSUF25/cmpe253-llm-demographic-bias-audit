"""Check pinned chat-template token lengths before opening confirmatory data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from inference.adapter import load_config, resolve_model
from scoring.bbq import format_question

ROOT = Path(__file__).resolve().parents[2]
CAB_CONFIG = ROOT / "configs/cab_build.json"
BBQ_CONFIG = ROOT / "configs/bbq_build.json"
MEMBERSHIP_CONFIG = ROOT / "configs/membership_data.json"
INFERENCE_CONFIG = ROOT / "configs/inference.json"
VALIDATION_PATH = Path("manifests/data_validation_manifest.json")
OUTPUT_PATH = Path("manifests/token_length_manifest.json")


class TokenLengthError(ValueError):
    """A pinned tokenizer or formatted sequence violates the locked data gate."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TokenLengthError(f"Missing required data artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _token_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, generate: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generate,
        truncation=False,
        padding=False,
        return_dict=True,
    )
    if not isinstance(encoded, Mapping):
        raise TokenLengthError("Pinned chat template did not return input_ids")
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, int) for i in ids):
        raise TokenLengthError("Pinned chat template did not return token IDs")
    return ids


def _input_length(tokenizer: Any, system: str, user: str) -> int:
    return len(
        _token_ids(
            tokenizer,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            generate=True,
        )
    )


def _training_length(tokenizer: Any, prompt: str, response: str) -> int:
    user = {"role": "user", "content": prompt}
    prefix = _token_ids(tokenizer, [user], generate=True)
    full = _token_ids(
        tokenizer, [user, {"role": "assistant", "content": response}], generate=False
    )
    if len(full) <= len(prefix) or full[: len(prefix)] != prefix:
        raise TokenLengthError("Phi training target does not follow the masked prompt")
    return len(full)


def _default_loader(repo_id: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        repo_id, revision=revision, trust_remote_code=False
    )


def check_token_lengths(
    root: Path,
    *,
    tokenizer_loader: Callable[[str, str], Any] = _default_loader,
) -> Path:
    """Write a deterministic receipt; fail if any sequence exceeds its locked cap."""
    root = root.expanduser().resolve()
    validation_path = root / VALIDATION_PATH
    if not validation_path.is_file():
        raise TokenLengthError("Run make validate-data before the tokenizer gate")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("structural_checks") != "passed":
        raise TokenLengthError("Structural data validation has not passed")
    config = load_config()
    cab_path = root / json.loads(CAB_CONFIG.read_text())["output_path"]
    bbq_config = json.loads(BBQ_CONFIG.read_text())
    bbq_items_path = root / bbq_config["items_path"]
    bbq_permutations_path = root / bbq_config["permutations_path"]
    membership_path = (
        root
        / json.loads(MEMBERSHIP_CONFIG.read_text())["output_directory"]
        / "training_presentations.jsonl"
    )
    cab = _rows(cab_path)
    bbq_items = _rows(bbq_items_path)
    bbq_permutations = _rows(bbq_permutations_path)
    training = _rows(membership_path)
    if (len(cab), len(bbq_items), len(bbq_permutations), len(training)) != (
        1604,
        1500,
        4500,
        3200,
    ):
        raise TokenLengthError("Data row counts differ from the frozen plan")
    items = {row["item_id"]: row for row in bbq_items}
    if len(items) != len(bbq_items):
        raise TokenLengthError("Duplicate BBQ item ID")
    failures: list[dict[str, Any]] = []
    maxima: dict[str, Any] = {}
    for model_key in config["model_keys"]:
        spec = resolve_model(model_key)
        tokenizer = tokenizer_loader(spec.repo_id, spec.tokenizer_revision)
        if not getattr(tokenizer, "chat_template", None):
            raise TokenLengthError(f"{model_key} has no official chat template")
        maxima[model_key] = {}
        for condition, settings in config["conditions"].items():
            system = settings["system_instruction"]
            cab_max = 0
            for row in cab:
                length = _input_length(tokenizer, system, row["prompt_text"])
                cab_max = max(cab_max, length)
                if length > 4096:
                    failures.append(
                        {
                            "model": model_key,
                            "condition": condition,
                            "dataset": "CAB",
                            "id": row["prompt_id"],
                            "tokens": length,
                        }
                    )
            bbq_max = 0
            for row in bbq_permutations:
                item = items.get(row["item_id"])
                if item is None:
                    raise TokenLengthError("BBQ permutation has no source item")
                question = format_question(item, row["answers"])
                length = _input_length(tokenizer, system, question)
                bbq_max = max(bbq_max, length)
                if length > 4096:
                    failures.append(
                        {
                            "model": model_key,
                            "condition": condition,
                            "dataset": "BBQ",
                            "id": f"{row['item_id']}|{row['order']}",
                            "tokens": length,
                        }
                    )
            maxima[model_key][condition] = {"cab": cab_max, "bbq": bbq_max}
        if model_key == "M2":
            training_max = 0
            for row in training:
                length = _training_length(tokenizer, row["prompt"], row["response"])
                training_max = max(training_max, length)
                if length > 512:
                    failures.append(
                        {
                            "model": model_key,
                            "condition": "qlora",
                            "dataset": "training",
                            "id": row["record_id"],
                            "tokens": length,
                        }
                    )
            maxima[model_key]["qlora_training"] = training_max
    manifest = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "limits": {"inference_input": 4096, "phi_training_sequence": 512},
        "counts": {
            "cab": len(cab),
            "bbq_items": len(bbq_items),
            "bbq_permutations": len(bbq_permutations),
            "training_presentations": len(training),
        },
        "max_tokens": maxima,
        "failures": failures,
        "validation_manifest_sha256": _sha(validation_path.read_bytes()),
        "inputs_sha256": {
            str(path.relative_to(root)): _sha(path.read_bytes())
            for path in (
                cab_path,
                bbq_items_path,
                bbq_permutations_path,
                membership_path,
            )
        },
        "inference_config_sha256": _sha(INFERENCE_CONFIG.read_bytes()),
        "tokenizer_revisions": {
            key: resolve_model(key).tokenizer_revision for key in config["model_keys"]
        },
    }
    output = root / OUTPUT_PATH
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() and output.read_bytes() != payload:
        raise TokenLengthError("Existing token-length manifest differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(payload)
    if failures:
        raise TokenLengthError(f"{len(failures)} sequences exceed the locked limits")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(check_token_lengths(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
