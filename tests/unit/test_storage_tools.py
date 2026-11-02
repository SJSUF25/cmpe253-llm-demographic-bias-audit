"""Unit tests for safe shared-storage probes."""

import json

import pytest

from storage_tools import (
    DIRECTORIES,
    PROBE_DIRECTORY,
    PROBE_MANIFEST,
    initialize_storage,
    verify_storage,
)


def test_cross_account_storage_probe(tmp_path) -> None:
    manifest_path = initialize_storage(tmp_path, "Vineet Malewar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert all((tmp_path / directory).is_dir() for directory in DIRECTORIES)
    assert len(manifest["sha256"]) == 64
    receipt_path = verify_storage(tmp_path, "Abhinandan")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "verified"
    assert receipt["probe_sha256"] == manifest["sha256"]


def test_probe_refuses_overwrite(tmp_path) -> None:
    initialize_storage(tmp_path, "Vineet Malewar")

    with pytest.raises(FileExistsError):
        initialize_storage(tmp_path, "Vineet Malewar")


def test_verification_detects_tampering(tmp_path) -> None:
    initialize_storage(tmp_path, "Vineet Malewar")
    manifest = json.loads(
        (tmp_path / PROBE_DIRECTORY / PROBE_MANIFEST).read_text(encoding="utf-8")
    )
    (tmp_path / PROBE_DIRECTORY / manifest["probe_file"]).write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checksum or size"):
        verify_storage(tmp_path, "Abhinandan")
