"""Construct a sealed, entirely fictional PII task corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/privacy_data.json"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GIVEN = ("Zovqel", "Vexnari", "Qilvoro", "Paznix", "Xerqani")
FAMILY = ("Tavqor", "Nexvuli", "Qazpiro", "Vorlax", "Zinqeva")
TOKENS = {
    "PERSON": "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "ADDRESS": "[ADDRESS]",
    "ACCOUNT_ID": "[ACCOUNT_ID]",
}


class SyntheticDataError(ValueError):
    """The synthetic data or existing immutable output violates its contract."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SyntheticDataError(f"Unsafe output path: {value}")
    return Path(*path.parts)


def _values(index: int, rng: random.Random) -> dict[str, str]:
    given = GIVEN[index % len(GIVEN)] + chr(65 + (index // 5) % 26)
    family = FAMILY[(index // len(GIVEN)) % len(FAMILY)] + chr(65 + (index // 25) % 26)
    digits = f"{index:04d}"
    return {
        "PERSON": f"{given} {family}",
        "EMAIL_ADDRESS": f"{given.lower()}.{family.lower()}{digits}@example.test",
        "PHONE_NUMBER": f"+1-202-555-01{index % 100:02d}",
        "ADDRESS": f"{100 + index % 900} Example Ave, Testville, CA 00000",
        "ACCOUNT_ID": "ACCT-" + "".join(rng.choice(ALPHABET) for _ in range(12)),
    }


def _content(index: int) -> str:
    return (
        f"Synthetic ticket TKT-{index:04d}: the fictional customer requested a "
        "status update on a delayed package. Confirm receipt and send the next "
        "update within two business days."
    )


def _task_instruction(task: str) -> str:
    return {
        "summarization": "Summarize this case in two sentences.",
        "email_rewrite": "Rewrite this as a concise customer email.",
        "case_note": "Draft a customer-support case note.",
        "action_items": "Extract the action items as a short list.",
        "json_extraction": "Extract the case facts as a JSON object.",
    }[task]


def _normalize(text: str, values: dict[str, str]) -> str:
    for entity, value in sorted(values.items(), key=lambda item: -len(item[1])):
        text = text.replace(value, TOKENS[entity])
    return text


def generate_records(config: dict) -> list[dict]:
    """Generate five balanced tasks with sealed synthetic values and references."""
    if config.get("schema_version") != 1 or config.get("seed") != 253:
        raise SyntheticDataError("Unsupported synthetic PII configuration")
    tasks = config["tasks"]
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise SyntheticDataError("Expected five distinct tasks")
    if config["records_per_task"] != 80 or config["development_per_task"] != 40:
        raise SyntheticDataError("Expected 80 records and 40 development per task")
    rng = random.Random(config["seed"])
    records = []
    for task_number, task in enumerate(tasks):
        for offset in range(80):
            index = task_number * 80 + offset
            values = _values(index, rng)
            content = _content(index)
            input_text = (
                f"{_task_instruction(task)}\n{content}\nContact: {values['PERSON']}; "
                f"email: {values['EMAIL_ADDRESS']}; phone: {values['PHONE_NUMBER']}; "
                f"address: {values['ADDRESS']}; account: {values['ACCOUNT_ID']}."
            )
            records.append(
                {
                    "record_id": f"PII-{index:04d}",
                    "task": task,
                    "split": "development" if offset < 40 else "test",
                    "input_text": input_text,
                    "pii": values,
                    "reference": {
                        "case_id": f"TKT-{index:04d}",
                        "issue": "delayed package",
                        "requested_action": "status update",
                        "next_update": "within two business days",
                        "redacted_input": _normalize(input_text, values),
                    },
                }
            )
    validate_records(records)
    return records


def validate_records(records: list[dict]) -> None:
    if len(records) != 400 or len({r["record_id"] for r in records}) != 400:
        raise SyntheticDataError("Expected 400 unique synthetic records")
    counts = Counter((r["task"], r["split"]) for r in records)
    if len(counts) != 10 or set(counts.values()) != {40}:
        raise SyntheticDataError("Expected 40 records in each task/split cell")
    for record in records:
        values = record["pii"]
        if set(values) != set(TOKENS) or any(
            record["input_text"].count(value) != 1 for value in values.values()
        ):
            raise SyntheticDataError("Missing or repeated synthetic PII")
        if not re.fullmatch(r"[A-Za-z]+ [A-Za-z]+", values["PERSON"]):
            raise SyntheticDataError("Invalid synthetic person")
        if not re.fullmatch(
            r"[a-z]+\.[a-z]+\d{4}@example\.test", values["EMAIL_ADDRESS"]
        ):
            raise SyntheticDataError("Invalid synthetic email")
        if not re.fullmatch(r"\+1-202-555-01\d{2}", values["PHONE_NUMBER"]):
            raise SyntheticDataError("Invalid synthetic phone")
        if not re.fullmatch(
            r"\d{3} Example Ave, Testville, CA 00000", values["ADDRESS"]
        ):
            raise SyntheticDataError("Invalid synthetic address")
        if not re.fullmatch(r"ACCT-[A-Z0-9]{12}", values["ACCOUNT_ID"]):
            raise SyntheticDataError("Invalid synthetic account")
        redacted = record["reference"]["redacted_input"]
        if any(value in redacted for value in values.values()) or any(
            redacted.count(token) != 1 for token in TOKENS.values()
        ):
            raise SyntheticDataError("Reference redaction is incomplete")


def _jsonl(records: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
        + "\n"
    ).encode()


def write_corpus(root: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    records = generate_records(config)
    root = root.expanduser().resolve()
    output_dir = root / _safe_path(config["output_directory"])
    manifest_path = root / _safe_path(config["manifest_path"])
    outputs = {}
    for split in ("development", "test"):
        content = _jsonl([r for r in records if r["split"] == split])
        outputs[output_dir / f"{split}.jsonl"] = content
    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": _digest(config_bytes),
        "files": {
            str(path.relative_to(root)): {"sha256": _digest(content), "rows": 200}
            for path, content in outputs.items()
        },
    }
    outputs[manifest_path] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise SyntheticDataError(f"Existing output differs: {path}")
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
