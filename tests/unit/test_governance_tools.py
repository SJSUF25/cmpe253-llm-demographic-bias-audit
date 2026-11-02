import json
from pathlib import Path

from governance_tools import LOCKED_FILES, create_lock, sha256_file


def test_governance_lock_contains_every_frozen_input(tmp_path: Path) -> None:
    output = create_lock(tmp_path / "governance.json")
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["approval_status"] == "pending_both_members"
    assert [entry["path"] for entry in manifest["files"]] == list(LOCKED_FILES)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("bias-audit\n", encoding="utf-8")

    assert sha256_file(fixture) == sha256_file(fixture)
