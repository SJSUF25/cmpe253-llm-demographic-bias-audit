import json
from collections import UserDict
from pathlib import Path

import pytest

from data.token_lengths import TokenLengthError, check_token_lengths


class FakeTokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": kwargs["add_generation_prompt"],
            "truncation": False,
            "padding": False,
            "return_dict": True,
        }
        length = 5 + sum(len(message["content"].split()) for message in messages)
        return UserDict({"input_ids": list(range(length))})


def _write_rows(root: Path, relative: str, rows: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _data_root(root: Path, *, overlength: bool = False) -> None:
    _write_rows(
        root,
        "private_outputs/processed_data/cab/prompt_manifest.jsonl",
        [
            {
                "prompt_id": f"CAB-{index}",
                "prompt_text": "long " * 4097 if overlength and index == 0 else "short",
            }
            for index in range(1604)
        ],
    )
    _write_rows(
        root,
        "private_outputs/processed_data/bbq/items.jsonl",
        [
            {
                "item_id": f"BBQ-{index}",
                "context": "context",
                "question": "question",
            }
            for index in range(1500)
        ],
    )
    _write_rows(
        root,
        "private_outputs/processed_data/bbq/permutations.jsonl",
        [
            {
                "item_id": f"BBQ-{index}",
                "order": order,
                "answers": ["first", "second", "third"],
            }
            for index in range(1500)
            for order in range(3)
        ],
    )
    _write_rows(
        root,
        "private_outputs/membership/training_presentations.jsonl",
        [
            {"record_id": f"REC-{index}", "prompt": "prompt", "response": "target"}
            for index in range(3200)
        ],
    )
    validation = root / "manifests/data_validation_manifest.json"
    validation.parent.mkdir(parents=True, exist_ok=True)
    validation.write_text(json.dumps({"structural_checks": "passed"}))


def test_pinned_template_gate_writes_stable_receipt(tmp_path: Path) -> None:
    _data_root(tmp_path)
    loaded = []

    def loader(repo: str, revision: str) -> FakeTokenizer:
        loaded.append((repo, revision))
        return FakeTokenizer()

    output = check_token_lengths(tmp_path, tokenizer_loader=loader)
    first = output.read_bytes()
    assert check_token_lengths(tmp_path, tokenizer_loader=loader).read_bytes() == first
    manifest = json.loads(first)
    assert manifest["status"] == "passed"
    assert manifest["counts"]["bbq_permutations"] == 4500
    assert manifest["counts"]["training_presentations"] == 3200
    assert manifest["max_tokens"]["M2"]["qlora_training"] < 512
    assert len(loaded) == 4


def test_overlength_input_blocks_gate_without_truncation(tmp_path: Path) -> None:
    _data_root(tmp_path, overlength=True)

    with pytest.raises(TokenLengthError, match="exceed the locked limits"):
        check_token_lengths(tmp_path, tokenizer_loader=lambda *_: FakeTokenizer())

    manifest_path = tmp_path / "manifests/token_length_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert {row["id"] for row in manifest["failures"]} == {"CAB-0"}
    assert all("prompt_text" not in row for row in manifest["failures"])
