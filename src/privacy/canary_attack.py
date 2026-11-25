"""Measure controlled canary extraction and member/non-member attack scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from environment_tools import configure_determinism
from inference import runner
from inference.adapter import resolve_model
from privacy import train_canary
from privacy.membership_data import DEFAULT_CONFIG as DATA_CONFIG
from privacy.membership_data import generate_corpus

ROOT = Path(__file__).resolve().parents[2]
MODES = ("base", "adapter")
PARTITIONS = ("member_canaries", "nonmember_calibration", "nonmember_evaluation")
SCORE_NAMES = ("negative_loss", "min_k20")


class CanaryAttackError(ValueError):
    """The attack input, model, result, or threshold violates the frozen plan."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _read_partitions(root: Path) -> tuple[list[dict], dict[str, str]]:
    config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    manifest_path = root / config["manifest_path"]
    if not manifest_path.is_file():
        raise CanaryAttackError("Missing membership corpus manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != _sha(DATA_CONFIG.read_bytes()):
        raise CanaryAttackError("Membership configuration hash differs")
    expected = generate_corpus(config)
    records = []
    hashes = {}
    for partition in PARTITIONS:
        path = root / config["output_directory"] / f"{partition}.jsonl"
        if not path.is_file():
            raise CanaryAttackError(f"Missing private {partition} partition")
        content = path.read_bytes()
        relative = str(path.relative_to(root))
        if manifest.get("files", {}).get(relative) != {
            "rows": len(expected[partition]),
            "sha256": _sha(content),
        }:
            raise CanaryAttackError(f"Changed private {partition} partition")
        rows = [json.loads(line) for line in content.decode().splitlines()]
        if rows != expected[partition]:
            raise CanaryAttackError(f"Changed private {partition} partition")
        hashes[partition] = _sha(content)
        records.extend(rows)
    if len(records) != 1300 or len({row["record_id"] for row in records}) != 1300:
        raise CanaryAttackError("Canary attack inventory is incomplete")
    return records, hashes


def threshold_from_calibration(scores: list[float]) -> float:
    """Sixth highest of 500; ties at threshold classify as non-member."""
    if len(scores) != 500 or any(not math.isfinite(x) for x in scores):
        raise CanaryAttackError("Expected 500 finite calibration scores")
    threshold = sorted(scores, reverse=True)[5]
    if sum(value > threshold for value in scores) > 5:
        raise CanaryAttackError("Calibration exceeded the 1% FPR bound")
    return threshold


def roc_auc(positive: list[float], negative: list[float]) -> float:
    """Tie-aware pairwise AUC from average ranks, without a scorer dependency."""
    if not positive or not negative:
        raise CanaryAttackError("AUC needs both member and non-member scores")
    values = sorted([(x, 1) for x in positive] + [(x, 0) for x in negative])
    if any(not math.isfinite(value) for value, _ in values):
        raise CanaryAttackError("Non-finite attack score")
    rank_sum = 0.0
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[end][0] == values[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in values[cursor:end])
        cursor = end
    return (rank_sum - len(positive) * (len(positive) + 1) / 2) / (
        len(positive) * len(negative)
    )


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise CanaryAttackError("Bootstrap produced no samples")
    last = len(ordered) - 1
    return [ordered[math.floor(0.025 * last)], ordered[math.floor(0.975 * last)]]


def _bootstrap(
    members: list[dict],
    evaluation: list[dict],
    threshold: float,
    score_name: str,
    *,
    seed: int = 253,
    repetitions: int = 10000,
) -> dict[str, list[float]]:
    if repetitions < 1:
        raise CanaryAttackError("Bootstrap repetition count must be positive")
    rng = random.Random(seed)
    strata = {
        repetition: [row for row in members if row["repetition"] == repetition]
        for repetition in (1, 5, 10)
    }
    samples = {name: [] for name in ("auc", "tpr", "fpr")}
    for _ in range(repetitions):
        member_draw = [
            rng.choice(strata[repetition])[score_name]
            for repetition in (1, 5, 10)
            for _ in range(100)
        ]
        eval_draw = [rng.choice(evaluation)[score_name] for _ in range(500)]
        samples["auc"].append(roc_auc(member_draw, eval_draw))
        samples["tpr"].append(sum(x > threshold for x in member_draw) / 300)
        samples["fpr"].append(sum(x > threshold for x in eval_draw) / 500)
    return {name: _interval(values) for name, values in samples.items()}


def summarize(rows: list[dict], *, bootstrap_repetitions: int = 10000) -> dict:
    """Lock thresholds on calibration only and report held-out attack estimates."""
    groups = {
        partition: [row for row in rows if row["split"] == partition]
        for partition in ("member", "calibration", "evaluation")
    }
    if (
        len(rows) != 1300
        or len({row["record_id"] for row in rows}) != 1300
        or [len(groups[key]) for key in groups] != [300, 500, 500]
        or any(
            sum(row["repetition"] == repetition for row in groups["member"]) != 100
            for repetition in (1, 5, 10)
        )
    ):
        raise CanaryAttackError("Attack score partitions do not reconcile")
    members, calibration, evaluation = (
        groups["member"],
        groups["calibration"],
        groups["evaluation"],
    )
    scores = {}
    for name in SCORE_NAMES:
        threshold = threshold_from_calibration([row[name] for row in calibration])
        positive = [row[name] for row in members]
        negative = [row[name] for row in evaluation]
        scores[name] = {
            "threshold": threshold,
            "calibration_false_positives": sum(
                row[name] > threshold for row in calibration
            ),
            "auc": roc_auc(positive, negative),
            "tpr_at_calibrated_1pct_fpr": sum(x > threshold for x in positive) / 300,
            "evaluation_fpr": sum(x > threshold for x in negative) / 500,
            "ci95": _bootstrap(
                members,
                evaluation,
                threshold,
                name,
                repetitions=bootstrap_repetitions,
            ),
        }
    rng = random.Random(253)
    extraction = {}
    for repetition in (1, 5, 10):
        present = [
            int(row["extracted_exact"])
            for row in members
            if row["repetition"] == repetition
        ]
        rates = [
            sum(rng.choice(present) for _ in range(100)) / 100
            for _ in range(bootstrap_repetitions)
        ]
        extraction[str(repetition)] = {
            "records": 100,
            "exact_rate": sum(present) / 100,
            "ci95": _interval(rates),
        }
    return {"score_attacks": scores, "member_extraction_by_repetition": extraction}


def _token_ids(tokenizer: Any, messages: list[dict], generate: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generate,
        truncation=False,
        padding=False,
        return_dict=True,
    )
    if not isinstance(encoded, Mapping):
        raise CanaryAttackError("Chat template did not return token IDs")
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(x, int) for x in ids):
        raise CanaryAttackError("Chat template returned invalid token IDs")
    return ids


def score_one(model: Any, tokenizer: Any, record: dict) -> dict:
    """Greedy extraction plus target-only loss and lowest-20%-token log-probability."""
    import torch

    user = {"role": "user", "content": record["prompt"]}
    prefix = _token_ids(tokenizer, [user], True)
    full = _token_ids(
        tokenizer,
        [user, {"role": "assistant", "content": record["response"]}],
        False,
    )
    if len(full) <= len(prefix) or full[: len(prefix)] != prefix or len(full) > 512:
        raise CanaryAttackError("Attack target boundary or token cap differs")
    device = model.device
    with torch.inference_mode():
        input_ids = torch.tensor([full], device=device)
        logits = model(input_ids, use_cache=False).logits
        target = input_ids[:, len(prefix) :]
        relevant = logits[:, len(prefix) - 1 : -1, :].float()
        token_log_probs = torch.log_softmax(relevant, dim=-1)
        chosen = token_log_probs.gather(2, target.unsqueeze(-1)).squeeze()
        values = chosen.reshape(-1).cpu().tolist()
        prompt_ids = torch.tensor([prefix], device=device)
        generated = model.generate(
            prompt_ids,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if generated.shape[0] != 1 or generated[0, : len(prefix)].tolist() != prefix:
        raise CanaryAttackError("Generation did not preserve the prompt")
    output = tokenizer.decode(
        generated[0, len(prefix) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    k = max(1, math.ceil(len(values) * 0.2))
    negative_loss = sum(values) / len(values)
    min_k20 = sum(sorted(values)[:k]) / k
    if not all(math.isfinite(x) for x in (negative_loss, min_k20)):
        raise CanaryAttackError("Non-finite attack score")
    return {
        "record_id": record["record_id"],
        "split": record["split"],
        "repetition": record["repetition"],
        "negative_loss": negative_loss,
        "min_k20": min_k20,
        "extracted_exact": output.strip() == record["response"],
    }


def _load_model(root: Path, mode: str) -> tuple[Any, Any, dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    spec = resolve_model("M2")
    context = {
        "model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
    }
    checkpoint = root / "adapters_private/canary_m2_qlora/epoch_3"
    if mode == "adapter":
        receipt_path = root / "manifests/canary_training/epoch_3.json"
        if not receipt_path.is_file():
            raise CanaryAttackError("Final training receipt is missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("epoch") != 3
            or any(
                receipt.get("context", {}).get(key) != value
                for key, value in context.items()
            )
            or not train_canary._checkpoint_valid(checkpoint, receipt)
        ):
            raise CanaryAttackError("Final adapter does not match training receipt")
        context["training_receipt_sha256"] = _sha(receipt_path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, revision=spec.tokenizer_revision, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        trust_remote_code=False,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="eager",
    )
    model = (
        PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
        if mode == "adapter"
        else base
    ).eval()
    return model, tokenizer, context


def _score_parts(
    root: Path,
    mode: str,
    records: list[dict],
    source_hashes: dict[str, str],
    context: dict,
    commit: str,
    model: Any,
    tokenizer: Any,
) -> list[dict]:
    """Checkpoint 100 rows at a time and verify every completed part on resume."""
    directory = root / f"private_outputs/canary_attack/{mode}_parts"
    scored = []
    for start in range(0, len(records), 100):
        block = records[start : start + 100]
        number = start // 100
        path = directory / f"{number:04d}.jsonl"
        receipt_path = directory / f"{number:04d}.json"
        expected_context = {
            "schema_version": 1,
            "mode": mode,
            "code_commit": commit,
            "source_hashes": source_hashes,
            "model_context": context,
            "record_ids": [row["record_id"] for row in block],
        }
        if path.exists() or receipt_path.exists():
            if not path.is_file() or not receipt_path.is_file():
                raise CanaryAttackError("Incomplete attack checkpoint part")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if {
                key: receipt.get(key) for key in expected_context
            } != expected_context or receipt.get("sha256") != _sha(path.read_bytes()):
                raise CanaryAttackError("Attack checkpoint part differs")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            rows = [score_one(model, tokenizer, row) for row in block]
            content = (
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
            ).encode()
            runner._atomic_write(path, content)
            runner._atomic_write(
                receipt_path,
                _json_bytes({**expected_context, "sha256": _sha(content)}),
            )
        if (
            len(rows) != len(block)
            or [row.get("record_id") for row in rows] != expected_context["record_ids"]
            or any(
                row.get("split") != source["split"]
                or row.get("repetition") != source["repetition"]
                or not isinstance(row.get("extracted_exact"), bool)
                or any(
                    not isinstance(row.get(name), (int, float))
                    or not math.isfinite(row[name])
                    for name in SCORE_NAMES
                )
                for row, source in zip(rows, block, strict=True)
            )
        ):
            raise CanaryAttackError("Attack checkpoint rows differ")
        scored.extend(rows)
    return scored


def run_attack(root: Path, mode: str) -> Path:
    """Write one private 1,300-row attack partition and aggregate summary."""
    if mode not in MODES:
        raise CanaryAttackError("Mode must be base or adapter")
    root = root.expanduser().resolve()
    commit, dirty = runner.git_state()
    if dirty:
        raise CanaryAttackError("Attack run requires a clean Git worktree")
    runner._check_test_gate(root)
    records, source_hashes = _read_partitions(root)
    configure_determinism(253)
    model, tokenizer, context = _load_model(root, mode)
    output = root / f"private_outputs/canary_attack/{mode}.jsonl"
    manifest = root / f"manifests/canary_attack/{mode}.json"
    aggregate = root / f"aggregate_results/canary_attack_{mode}.json"
    expected_ids = {row["record_id"] for row in records}
    if output.exists():
        if manifest.is_file():
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                prior.get("mode") != mode
                or prior.get("code_commit") != commit
                or prior.get("source_hashes") != source_hashes
                or prior.get("private_output_sha256") != _sha(output.read_bytes())
                or any(prior.get(key) != value for key, value in context.items())
            ):
                raise CanaryAttackError("Existing attack output/manifest differs")
        else:
            parts = _score_parts(
                root, mode, records, source_hashes, context, commit, model, tokenizer
            )
            expected = (
                "\n".join(json.dumps(row, sort_keys=True) for row in parts) + "\n"
            ).encode()
            if output.read_bytes() != expected:
                raise CanaryAttackError("Unreceipted attack output differs from parts")
        scored = [json.loads(line) for line in output.read_text().splitlines()]
        if len(scored) != 1300 or {row["record_id"] for row in scored} != expected_ids:
            raise CanaryAttackError("Existing attack output is incomplete")
    else:
        scored = _score_parts(
            root, mode, records, source_hashes, context, commit, model, tokenizer
        )
        runner._atomic_write(
            output,
            (
                "\n".join(json.dumps(row, sort_keys=True) for row in scored) + "\n"
            ).encode(),
        )
    summary = summarize(scored)
    metadata = {
        "schema_version": 1,
        "mode": mode,
        "code_commit": commit,
        "source_hashes": source_hashes,
        "private_output_sha256": _sha(output.read_bytes()),
        **context,
    }
    if manifest.exists() and manifest.read_bytes() != _json_bytes(metadata):
        raise CanaryAttackError("Existing attack manifest differs")
    if not manifest.exists():
        runner._atomic_write(manifest, _json_bytes(metadata))
    result = {
        "schema_version": 1,
        "mode": mode,
        "source_manifest_sha256": _sha(manifest.read_bytes()),
        **summary,
    }
    if aggregate.exists() and aggregate.read_bytes() != _json_bytes(result):
        raise CanaryAttackError("Existing attack summary differs")
    if not aggregate.exists():
        runner._atomic_write(aggregate, _json_bytes(result))
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)
    print(run_attack(args.root, args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
