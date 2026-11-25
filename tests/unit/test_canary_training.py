import json
from collections import UserDict
from pathlib import Path

import pytest

from privacy import train_canary
from privacy.membership_data import write_corpus


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "truncation": False,
            "padding": False,
            "return_dict": True,
        }
        if add_generation_prompt:
            return UserDict(input_ids=[1, 2, 3])
        assert len(messages) == 2
        return UserDict(input_ids=[1, 2, 3, 4, 5])


def test_frozen_membership_input_and_shuffle(tmp_path: Path):
    write_corpus(tmp_path)
    rows, digest = train_canary._data_rows(tmp_path)
    assert len(rows) == 3200
    assert len(digest) == 64
    first = train_canary.epoch_order(rows, 1)
    assert first == train_canary.epoch_order(rows, 1)
    assert first != train_canary.epoch_order(rows, 2)
    assert sorted(first) == list(range(3200))
    path = tmp_path / "private_outputs/membership/training_presentations.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(train_canary.CanaryTrainingError, match="hash or count"):
        train_canary._data_rows(tmp_path)


def test_training_labels_mask_prompt_and_reject_overlength():
    row = {"record_id": "REC-1", "prompt": "Question", "response": "Answer"}
    encoded = train_canary.encode_example(Tokenizer(), row)
    assert encoded["input_ids"] == [1, 2, 3, 4, 5]
    assert encoded["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(train_canary.CanaryTrainingError, match="512-token cap"):
        train_canary.encode_example(Tokenizer(), row, limit=4)


def test_resume_requires_hash_linked_contiguous_epochs(tmp_path: Path):
    config = train_canary.load_config()
    context = {"code_commit": "abc", "input_sha256": "def"}
    assert train_canary._resume_epoch(tmp_path, config, context) == 0
    manifest_dir = tmp_path / config["manifest_directory"]
    output_dir = tmp_path / config["output_directory"]
    checkpoint = output_dir / "epoch_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.bin").write_bytes(b"fixture")
    manifest_dir.mkdir(parents=True)
    receipt = {
        "epoch": 1,
        "context": {**context, "resolved_modules": ["q_proj"]},
        "files": train_canary._file_hashes(checkpoint),
    }
    (manifest_dir / "epoch_1.json").write_text(json.dumps(receipt))
    assert train_canary._resume_epoch(tmp_path, config, context) == 1
    (checkpoint / "adapter.bin").write_bytes(b"changed")
    with pytest.raises(train_canary.CanaryTrainingError, match="receipt differs"):
        train_canary._resume_epoch(tmp_path, config, context)
