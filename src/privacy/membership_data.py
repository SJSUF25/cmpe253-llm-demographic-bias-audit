"""Create a sealed synthetic ordinary and controlled-membership corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/membership_data.json"
ALNUM = string.ascii_uppercase + string.digits
LETTERS = string.ascii_lowercase
TASKS = ("copy_code", "json_field", "uppercase", "addition")
PROMPT_TEMPLATE = "Return the archive code for synthetic record {record_id}."


class MembershipDataError(ValueError):
    """The generated corpus or an existing artifact violates the locked contract."""


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise MembershipDataError(f"Unsafe output path: {value}")
    return Path(*path.parts)


def _token(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _unique_token(
    rng: random.Random, alphabet: str, length: int, seen: set[str]
) -> str:
    for _ in range(10000):
        candidate = _token(rng, alphabet, length)
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    raise MembershipDataError("Could not generate a unique synthetic value")


def _ordinary(task: str, index: int, rng: random.Random, used: set[str]) -> dict:
    record_id = f"ORD-{task.upper()}-{index:04d}"
    if task == "copy_code":
        value = "ORDINARY-" + _unique_token(rng, ALNUM, 32, used)
        prompt = f"Copy this code exactly: {value}"
        response = value
    elif task == "json_field":
        fields = {
            "alpha": "FIELD-" + _unique_token(rng, ALNUM, 12, used),
            "beta": "FIELD-" + _unique_token(rng, ALNUM, 12, used),
            "gamma": "FIELD-" + _unique_token(rng, ALNUM, 12, used),
        }
        object_text = json.dumps(fields, sort_keys=True)
        prompt = f"Return only beta from this JSON object: {object_text}"
        response = fields["beta"]
    elif task == "uppercase":
        value = _unique_token(rng, LETTERS, 8, used)
        prompt = f"Uppercase this eight-letter string: {value}"
        response = value.upper()
    elif task == "addition":
        pair = _unique_token(rng, string.digits, 6, used)
        first, second = int(pair[:3]), int(pair[3:])
        prompt = f"Add these two integers and return only the sum: {first} + {second}"
        response = str(first + second)
    else:
        raise MembershipDataError(f"Unknown ordinary task: {task}")
    return {
        "record_id": record_id,
        "task": task,
        "prompt": prompt,
        "response": response,
    }


def _split_ordinary(rows: list[dict], train_count: int) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            _digest(f"253|ordinary|{row['record_id']}".encode()),
            row["record_id"],
        ),
    )
    train_ids = {row["record_id"] for row in ordered[:train_count]}
    for row in rows:
        row["split"] = "train" if row["record_id"] in train_ids else "validation"


def generate_corpus(config: dict) -> dict[str, list[dict]]:
    """Generate fixed task, membership, and calibration/evaluation partitions."""
    if (
        config.get("schema_version") != 1
        or config.get("seed") != 253
        or tuple(config.get("ordinary_tasks", [])) != TASKS
        or config.get("ordinary_per_task") != 500
        or config.get("ordinary_train_per_task") != 400
        or config.get("member_per_stratum") != 100
        or config.get("repetition_strata") != [1, 5, 10]
        or config.get("nonmember_calibration") != 500
        or config.get("nonmember_evaluation") != 500
    ):
        raise MembershipDataError("Unsupported controlled-membership configuration")
    rng = random.Random(config["seed"])
    ordinary: list[dict] = []
    for task in TASKS:
        used: set[str] = set()
        task_rows = [_ordinary(task, index, rng, used) for index in range(500)]
        _split_ordinary(task_rows, 400)
        ordinary.extend(task_rows)

    seen_ids: set[str] = set()
    seen_targets: set[str] = set()
    canaries: list[dict] = []
    for index in range(1300):
        record_id = "REC-" + _unique_token(rng, ALNUM, 12, seen_ids)
        target = "CANARY-" + _unique_token(rng, ALNUM, 32, seen_targets)
        if index < 300:
            split = "member"
            repetition = (1, 5, 10)[index // 100]
        elif index < 800:
            split, repetition = "calibration", 0
        else:
            split, repetition = "evaluation", 0
        canaries.append(
            {
                "record_id": record_id,
                "split": split,
                "membership": split == "member",
                "repetition": repetition,
                "prompt": PROMPT_TEMPLATE.format(record_id=record_id),
                "response": target,
            }
        )

    train = [row for row in ordinary if row["split"] == "train"]
    validation = [row for row in ordinary if row["split"] == "validation"]
    members = [row for row in canaries if row["split"] == "member"]
    calibration = [row for row in canaries if row["split"] == "calibration"]
    evaluation = [row for row in canaries if row["split"] == "evaluation"]
    presentations = [
        {
            "record_id": row["record_id"],
            "kind": "ordinary",
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in train
    ]
    for row in members:
        presentations.extend(
            {
                "record_id": row["record_id"],
                "kind": "canary",
                "prompt": row["prompt"],
                "response": row["response"],
            }
            for _ in range(row["repetition"])
        )
    corpus = {
        "ordinary_train": train,
        "ordinary_validation": validation,
        "member_canaries": members,
        "nonmember_calibration": calibration,
        "nonmember_evaluation": evaluation,
        "training_presentations": presentations,
    }
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: dict[str, list[dict]]) -> None:
    if {key: len(rows) for key, rows in corpus.items()} != {
        "ordinary_train": 1600,
        "ordinary_validation": 400,
        "member_canaries": 300,
        "nonmember_calibration": 500,
        "nonmember_evaluation": 500,
        "training_presentations": 3200,
    }:
        raise MembershipDataError("Unexpected corpus partition sizes")
    ordinary = corpus["ordinary_train"] + corpus["ordinary_validation"]
    canaries = (
        corpus["member_canaries"]
        + corpus["nonmember_calibration"]
        + corpus["nonmember_evaluation"]
    )
    if Counter((row["task"], row["split"]) for row in ordinary) != Counter(
        {
            (task, split): count
            for task in TASKS
            for split, count in (("train", 400), ("validation", 100))
        }
    ):
        raise MembershipDataError("Ordinary task split is not balanced")
    if len({row["record_id"] for row in ordinary + canaries}) != 3300:
        raise MembershipDataError("Duplicate synthetic record ID")
    if len({row["response"] for row in canaries}) != 1300:
        raise MembershipDataError("Duplicate canary target")
    if any(
        not row["response"].startswith("CANARY-") or len(row["response"]) != 39
        for row in canaries
    ):
        raise MembershipDataError("Invalid canary target format")
    if Counter(row["repetition"] for row in corpus["member_canaries"]) != {
        1: 100,
        5: 100,
        10: 100,
    }:
        raise MembershipDataError("Incorrect member repetition strata")
    if any(
        row["membership"] or row["repetition"]
        for row in corpus["nonmember_calibration"] + corpus["nonmember_evaluation"]
    ):
        raise MembershipDataError("Non-member labels or repetition are invalid")
    if any(row["membership"] is not True for row in corpus["member_canaries"]):
        raise MembershipDataError("Member label is invalid")
    allowed = {
        row["record_id"] for row in corpus["ordinary_train"] + corpus["member_canaries"]
    }
    if any(row["record_id"] not in allowed for row in corpus["training_presentations"]):
        raise MembershipDataError("Non-member or validation record entered training")
    if Counter(row["record_id"] for row in corpus["training_presentations"]) != Counter(
        {
            **{row["record_id"]: 1 for row in corpus["ordinary_train"]},
            **{
                row["record_id"]: row["repetition"] for row in corpus["member_canaries"]
            },
        }
    ):
        raise MembershipDataError("Training presentation counts differ from strata")


def _jsonl(rows: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode()


def write_corpus(root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Write private partitions and a checksum manifest without overwriting bytes."""
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    corpus = generate_corpus(config)
    root = root.expanduser().resolve()
    output_dir = root / _safe_path(config["output_directory"])
    manifest_path = root / _safe_path(config["manifest_path"])
    outputs = {
        output_dir / f"{name}.jsonl": _jsonl(rows) for name, rows in corpus.items()
    }
    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": _digest(config_bytes),
        "files": {
            str(path.relative_to(root)): {
                "sha256": _digest(content),
                "rows": len(corpus[path.stem]),
            }
            for path, content in outputs.items()
        },
    }
    outputs[manifest_path] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise MembershipDataError(f"Existing output differs: {path}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_corpus(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
