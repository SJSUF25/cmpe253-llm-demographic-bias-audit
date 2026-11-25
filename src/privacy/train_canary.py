"""Train and resume the pinned Phi QLoRA canary adapter from private data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from environment_tools import configure_determinism
from inference import runner
from inference.adapter import resolve_model
from privacy.membership_data import DEFAULT_CONFIG as DATA_CONFIG
from privacy.membership_data import generate_corpus

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/canary_training.json"
LOCK = ROOT / "uv.lock"


class CanaryTrainingError(ValueError):
    """A training input, checkpoint, or setting differs from the frozen plan."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "config_id": "phi-canary-qlora-v1",
        "model_key": "M2",
        "seed": 253,
        "epochs": 3,
        "max_sequence_tokens": 512,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "output_directory": "adapters_private/canary_m2_qlora",
        "manifest_directory": "manifests/canary_training",
    }
    if config != expected:
        raise CanaryTrainingError("Training settings differ from frozen plan")
    return config


def _data_rows(root: Path) -> tuple[list[dict], str]:
    data_config = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    manifest_path = root / data_config["manifest_path"]
    path = root / data_config["output_directory"] / "training_presentations.jsonl"
    if not manifest_path.is_file() or not path.is_file():
        raise CanaryTrainingError("Build the private membership corpus first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = path.read_bytes()
    entry = manifest.get("files", {}).get(str(path.relative_to(root)))
    if manifest.get("config_sha256") != _sha(DATA_CONFIG.read_bytes()) or entry != {
        "rows": 3200,
        "sha256": _sha(content),
    }:
        raise CanaryTrainingError("Training corpus hash or count differs")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    expected = generate_corpus(data_config)["training_presentations"]
    if rows != expected:
        raise CanaryTrainingError("Training presentations differ from frozen generator")
    counts = Counter(row["record_id"] for row in rows)
    if len(rows) != 3200 or len(counts) != 1900:
        raise CanaryTrainingError("Unexpected training membership inventory")
    return rows, _sha(content)


def epoch_order(rows: list[dict], epoch: int, seed: int = 253) -> list[int]:
    """Generate the stored, deterministic presentation order for one epoch."""
    if epoch not in (1, 2, 3) or len(rows) != 3200:
        raise CanaryTrainingError("Unexpected epoch or presentation count")
    indices = list(range(len(rows)))
    random.Random(seed + epoch - 1).shuffle(indices)
    return indices


def encode_example(tokenizer: Any, row: dict, limit: int = 512) -> dict:
    """Mask every prompt token and refuse any truncated training sequence."""
    user = {"role": "user", "content": row["prompt"]}
    options = {
        "tokenize": True,
        "truncation": False,
        "padding": False,
        "return_dict": True,
    }
    prefix = tokenizer.apply_chat_template(
        [user], add_generation_prompt=True, **options
    )
    full = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": row["response"]}],
        add_generation_prompt=False,
        **options,
    )
    if not isinstance(prefix, Mapping) or not isinstance(full, Mapping):
        raise CanaryTrainingError("Official chat template did not return token IDs")
    prompt_ids, full_ids = prefix.get("input_ids"), full.get("input_ids")
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(full_ids, list)
        or not prompt_ids
        or any(not isinstance(value, int) for value in full_ids)
        or len(full_ids) <= len(prompt_ids)
        or full_ids[: len(prompt_ids)] != prompt_ids
        or len(full_ids) > limit
    ):
        raise CanaryTrainingError("Prompt/target boundary or 512-token cap failed")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels, "record_id": row["record_id"]}


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): _sha(item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _checkpoint_valid(path: Path, receipt: dict) -> bool:
    return path.is_dir() and _file_hashes(path) == receipt.get("files")


def _resume_epoch(root: Path, config: dict, context: dict) -> int:
    manifest_dir = root / config["manifest_directory"]
    output_dir = root / config["output_directory"]
    completed = 0
    for epoch in range(1, 4):
        receipt_path = manifest_dir / f"epoch_{epoch}.json"
        checkpoint = output_dir / f"epoch_{epoch}"
        if not receipt_path.exists():
            if checkpoint.exists():
                raise CanaryTrainingError("Unreceipted training checkpoint exists")
            break
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            {key: receipt.get("context", {}).get(key) for key in context} != context
            or receipt.get("epoch") != epoch
            or not _checkpoint_valid(checkpoint, receipt)
        ):
            raise CanaryTrainingError("Training checkpoint receipt differs")
        completed = epoch
    if any(
        (manifest_dir / f"epoch_{epoch}.json").exists()
        or (output_dir / f"epoch_{epoch}").exists()
        for epoch in range(completed + 2, 4)
    ):
        raise CanaryTrainingError("Training checkpoint sequence has a gap")
    return completed


def _batch(encoded: list[dict], pad_id: int, device: Any) -> dict:
    import torch

    width = max(len(item["input_ids"]) for item in encoded)
    ids = [
        item["input_ids"] + [pad_id] * (width - len(item["input_ids"]))
        for item in encoded
    ]
    labels = [
        item["labels"] + [-100] * (width - len(item["labels"])) for item in encoded
    ]
    masks = [
        [1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"]))
        for item in encoded
    ]
    return {
        "input_ids": torch.tensor(ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(masks, device=device),
    }


def _save_epoch(
    root: Path,
    config: dict,
    context: dict,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    order: list[int],
    losses: list[float],
) -> Path:
    import torch

    output_dir = root / config["output_directory"]
    manifest_dir = root / config["manifest_directory"]
    checkpoint = output_dir / f"epoch_{epoch}"
    temporary = output_dir / f"epoch_{epoch}.tmp"
    receipt_path = manifest_dir / f"epoch_{epoch}.json"
    if checkpoint.exists() or temporary.exists() or receipt_path.exists():
        raise CanaryTrainingError("Checkpoint destination already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "training_state.pt",
    )
    (temporary / "order.json").write_text(json.dumps(order) + "\n", encoding="utf-8")
    temporary.rename(checkpoint)
    receipt = {
        "schema_version": 1,
        "context": context,
        "epoch": epoch,
        "presentations": len(order),
        "optimizer_steps": len(order)
        // (config["per_device_batch_size"] * config["gradient_accumulation_steps"]),
        "mean_microbatch_loss": sum(losses) / len(losses),
        "order_sha256": _sha((checkpoint / "order.json").read_bytes()),
        "files": _file_hashes(checkpoint),
    }
    runner._atomic_write(
        receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    )
    return receipt_path


def train(root: Path) -> Path:
    """Run three pinned epochs; resume only from a verified epoch boundary."""
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )

    root = root.expanduser().resolve()
    config = load_config()
    commit, dirty = runner.git_state()
    if dirty:
        raise CanaryTrainingError("Training requires a clean Git worktree")
    rows, input_hash = _data_rows(root)
    spec = resolve_model(config["model_key"])
    context = {
        "config_sha256": _sha(CONFIG.read_bytes()),
        "input_sha256": input_hash,
        "model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "uv_lock_sha256": _sha(LOCK.read_bytes()),
        "code_commit": commit,
    }
    completed = _resume_epoch(root, config, context)
    if completed == 3:
        return root / config["manifest_directory"] / "epoch_3.json"
    if not torch.cuda.is_available():
        raise CanaryTrainingError("Phi QLoRA training requires a CUDA GPU")
    configure_determinism(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, revision=spec.tokenizer_revision, trust_remote_code=False
    )
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise CanaryTrainingError("Pinned Phi tokenizer has no chat template/EOS")
    encoded = [encode_example(tokenizer, row) for row in rows]
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
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    if completed:
        checkpoint = root / config["output_directory"] / f"epoch_{completed}"
        model = PeftModel.from_pretrained(
            base, checkpoint / "adapter", is_trainable=True
        )
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise CanaryTrainingError("Non-LoRA base weights would be trained")
    resolved_modules = sorted({name.rsplit(".lora_", 1)[0] for name in trainable})
    if any("lm_head" in name for name in resolved_modules):
        raise CanaryTrainingError("Output head was selected for training")
    context["resolved_modules"] = resolved_modules
    plan_path = root / config["manifest_directory"] / "plan.json"
    plan = (
        json.dumps({"schema_version": 1, "context": context}, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    if plan_path.exists():
        if plan_path.read_bytes() != plan:
            raise CanaryTrainingError("Resolved training plan changed on resume")
    else:
        if completed:
            raise CanaryTrainingError("Training plan missing on resume")
        runner._atomic_write(plan_path, plan)
    if completed:
        prior = json.loads(
            (
                root / config["manifest_directory"] / f"epoch_{completed}.json"
            ).read_text()
        )
        if prior["context"] != context:
            raise CanaryTrainingError("Resolved modules changed on resume")
    optimizer = PagedAdamW8bit(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    steps_per_epoch = 3200 // (2 * 8)
    total_steps = steps_per_epoch * 3
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.03),
        num_training_steps=total_steps,
    )
    if completed:
        checkpoint = root / config["output_directory"] / f"epoch_{completed}"
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    model.train()
    for epoch in range(completed + 1, 4):
        torch.manual_seed(config["seed"] + epoch - 1)
        torch.cuda.manual_seed_all(config["seed"] + epoch - 1)
        order = epoch_order(rows, epoch)
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(order), 2):
            batch = _batch(
                [encoded[index] for index in order[offset : offset + 2]],
                tokenizer.eos_token_id,
                model.device,
            )
            result = model(**batch, use_cache=False)
            if not torch.isfinite(result.loss):
                raise CanaryTrainingError("Non-finite training loss")
            losses.append(float(result.loss.detach().cpu()))
            (result.loss / 8).backward()
            if (offset // 2 + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    config["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        receipt = _save_epoch(
            root, config, context, epoch, model, optimizer, scheduler, order, losses
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(train(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
