"""Record the documented AI data audit without claiming a human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from data.cab_quality import verify_manifest
from inference.runner import (
    GOVERNANCE_LOCK_PATH,
    ROOT,
    _check_test_gate,
)

EVIDENCE = ROOT / "docs/data_qa_ai_review.md"
SECTIONS = {
    "cab_development_families": 50,
    "bbq": 12,
    "pii_development": 10,
    "ordinary_train": 8,
    "member_canaries": 3,
    "nonmember_calibration": 2,
}
CONFIRMED_EXCLUSIONS = {
    "8db2c4861d6abdb07b91",
    "1e88e366bbeccbb628ea",
    "9df5a0510db2b84aa7da",
    "cf9725b5930197bc061c",
    "76a56f28d34e51162319",
    "f7c08f24d43c24e2f81e",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, data: dict) -> None:
    content = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing receipt differs: {path}")
    else:
        path.write_bytes(content)


def approve_ai_review(root: Path) -> Path:
    """Bind the prior 85-item AI audit and verified data inputs to a truthful gate."""
    root = root.expanduser().resolve()
    validation_path = root / "manifests/data_validation_manifest.json"
    lengths_path = root / "manifests/token_length_manifest.json"
    quality_path = root / "manifests/cab_implicit_quality_manifest.json"
    packet_path = root / "private_outputs/data_qa/review_packet.json"
    for path in (validation_path, lengths_path, quality_path, packet_path):
        if not path.is_file():
            raise ValueError(f"Missing data review input: {path}")
    validation = json.loads(validation_path.read_text())
    lengths = json.loads(lengths_path.read_text())
    packet = json.loads(packet_path.read_text())
    quality = verify_manifest(root)
    if validation.get("structural_checks") != "passed":
        raise ValueError("Structural validation did not pass")
    if lengths.get("status") != "passed" or lengths.get("failures"):
        raise ValueError("Pinned token-length check did not pass")
    if lengths.get("validation_manifest_sha256") != digest(validation_path):
        raise ValueError("Token-length receipt does not match structural validation")
    if validation.get("review_packet_sha256") != digest(packet_path):
        raise ValueError("Reviewed packet differs from the validated packet")
    if (
        digest(packet_path)
        != "2dde3d874e16407427f35c2663ba0e234bf108ce20ff1b16b6daae1fd1da7fed"
    ):
        raise ValueError("Packet differs from the 85 items in the documented AI audit")
    if {name: len(packet.get(name, [])) for name in SECTIONS} != SECTIONS:
        raise ValueError("AI review packet inventory differs")
    if packet.get("sealed_test_contents_included") is not False:
        raise ValueError("Review packet exposes sealed test contents")
    flagged = {
        item["scenario_id"] for item in packet["cab_development_families"]
    } & set(quality["excluded_implicit_scenario_ids"])
    if flagged != CONFIRMED_EXCLUSIONS:
        raise ValueError("Documented CAB exclusions differ from the review packet")
    if quality.get("status") != "passed_with_implicit_exclusions":
        raise ValueError("CAB implicit quality check did not pass")
    if len(quality["excluded_implicit_scenario_ids"]) != 23:
        raise ValueError("CAB exclusion count differs from the reviewed rule")
    for relative, expected in lengths["inputs_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Tokenizer input changed: {relative}")
    lock = json.loads(GOVERNANCE_LOCK_PATH.read_text())
    expected_prd = next(
        row["sha256"] for row in lock["files"] if row["path"] == "LLM_Bias_Audit_PRD.md"
    )
    if digest(root / "LLM_Bias_Audit_PRD.md") != expected_prd:
        raise ValueError("Drive PRD differs from the current governance lock")
    common = {
        "schema_version": 1,
        "validation_manifest_sha256": digest(validation_path),
        "token_length_manifest_sha256": digest(lengths_path),
        "cab_implicit_quality_manifest_sha256": digest(quality_path),
        "governance_lock_sha256": digest(GOVERNANCE_LOCK_PATH),
        "token_length_status": "passed",
        "review_status": "approved_with_documented_implicit_exclusions",
        "review_method": "ai_assisted_no_human_review",
        "reviewed_by": "Codex AI-assisted audit",
        "review_packet_sha256": digest(packet_path),
        "review_evidence_sha256": digest(EVIDENCE),
    }
    receipt_path = root / "manifests/data_ai_review_receipt.json"
    gate_path = root / "manifests/data_gate_approval.json"
    _write_once(receipt_path, common)
    _write_once(gate_path, {**common, "ai_review_receipt_sha256": digest(receipt_path)})
    _check_test_gate(root)
    return gate_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(approve_ai_review(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
