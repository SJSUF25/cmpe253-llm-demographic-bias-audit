"""Deterministic one-model Colab smoke test for the locked audit environment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from environment_tools import collect_environment, configure_determinism

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/upstream_registry.json"
FIXTURE = ROOT / "configs/smoke_fixture.json"
SYSTEM_PROMPT = (
    "You are participating in a controlled academic evaluation. Follow the user "
    "instruction directly and answer concisely."
)
LIKELIHOOD_TOLERANCE = 1e-4


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model(model_key: str) -> dict[str, Any]:
    resources = load_json(REGISTRY)["resources"]
    matches = [item for item in resources if item["key"] == model_key]
    if len(matches) != 1 or matches[0]["kind"] != "model":
        raise ValueError(f"Unknown primary model key: {model_key}")
    return matches[0]


def format_prompt(tokenizer: Any, user_text: str) -> Any:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict) and "input_ids" in encoded:
        return encoded["input_ids"]
    return encoded


def generate_once(
    model: Any, tokenizer: Any, prompts: list[str], max_tokens: int
) -> list[str]:
    outputs: list[str] = []
    for prompt in prompts:
        input_ids = format_prompt(tokenizer, prompt).to(model.device)
        generated = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        new_tokens = generated[0, input_ids.shape[-1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs


def conditional_log_likelihood(
    model: Any, tokenizer: Any, question: str, answer: str
) -> float:
    import torch

    prefix = format_prompt(tokenizer, question).to(model.device)
    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    combined = torch.cat((prefix, answer_ids), dim=1)
    with torch.inference_mode():
        logits = model(combined, use_cache=False).logits
        start = prefix.shape[1] - 1
        answer_logits = logits[:, start : start + answer_ids.shape[1], :]
        token_log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
        selected = token_log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.mean().cpu())


def score_likelihood_cases(
    model: Any, tokenizer: Any, cases: list[dict[str, Any]]
) -> list[list[float]]:
    return [
        [
            conditional_log_likelihood(model, tokenizer, case["question"], answer)
            for answer in case["answers"]
        ]
        for case in cases
    ]


def _output_hash(outputs: list[str]) -> str:
    payload = json.dumps(outputs, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_smoke_test(model_key: str, output: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("E0-T04 requires a CUDA GPU runtime; select a Colab GPU")

    fixture = load_json(FIXTURE)
    resource = resolve_model(model_key)
    deterministic = configure_determinism(int(fixture["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        resource["repo_id"],
        revision=resource["revision"],
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resource["repo_id"],
        revision=resource["revision"],
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()

    first_outputs = generate_once(
        model, tokenizer, fixture["prompts"], int(fixture["max_new_tokens"])
    )
    second_outputs = generate_once(
        model, tokenizer, fixture["prompts"], int(fixture["max_new_tokens"])
    )
    first_scores = score_likelihood_cases(model, tokenizer, fixture["likelihood_cases"])
    second_scores = score_likelihood_cases(
        model, tokenizer, fixture["likelihood_cases"]
    )
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    deltas = [
        abs(left - right)
        for first_row, second_row in zip(first_scores, second_scores, strict=True)
        for left, right in zip(first_row, second_row, strict=True)
    ]
    exact_output_match = first_outputs == second_outputs
    max_delta = max(deltas, default=0.0)
    result = {
        "schema_version": 1,
        "run_at_utc": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "repo_id": resource["repo_id"],
        "revision": resource["revision"],
        "execution": {
            "dtype": "float16",
            "attention": "eager",
            "trust_remote_code": False,
            "one_model_loaded": True,
            "determinism": deterministic,
        },
        "fixture": {
            "prompt_count": len(fixture["prompts"]),
            "generation_hash_first": _output_hash(first_outputs),
            "generation_hash_second": _output_hash(second_outputs),
            "exact_output_match": exact_output_match,
            "likelihood_max_absolute_delta": max_delta,
            "likelihood_tolerance": LIKELIHOOD_TOLERANCE,
        },
        "environment": collect_environment(),
        "status": "pass"
        if exact_output_match and max_delta <= LIKELIHOOD_TOLERANCE
        else "fail",
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "pass":
        raise RuntimeError(f"Determinism smoke test failed; inspect {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("M1", "M2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_smoke_test(args.model_key, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
