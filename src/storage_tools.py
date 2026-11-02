"""Safe shared-storage initialization and cross-account access probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

DIRECTORIES = (
    "raw_immutable",
    "private_outputs",
    "adapters_private",
    "aggregate_results",
    "manifests",
    "backups",
)
PROBE_DIRECTORY = "_access_probe"
PROBE_FILE = "probe.txt"
PROBE_MANIFEST = "probe_manifest.json"
VERIFICATION_RECEIPT = "verification_receipt.json"


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for content."""
    return hashlib.sha256(content).hexdigest()


def initialize_storage(root: Path, actor: str) -> Path:
    """Create the storage layout and a non-sensitive access probe."""
    if not actor.strip():
        raise ValueError("actor must not be blank")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for directory in DIRECTORIES:
        (root / directory).mkdir(exist_ok=True)

    probe_dir = root / PROBE_DIRECTORY
    probe_dir.mkdir(exist_ok=True)
    probe_path = probe_dir / PROBE_FILE
    manifest_path = probe_dir / PROBE_MANIFEST
    if probe_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Access probe already exists at {probe_dir}; "
            "verify it instead of overwriting"
        )

    content = (
        f"CMPE253 harmless storage probe\nnonce={secrets.token_hex(32)}\n".encode()
    )
    probe_path.write_bytes(content)
    manifest = {
        "schema_version": 1,
        "created_by": actor.strip(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "probe_file": PROBE_FILE,
        "size_bytes": len(content),
        "sha256": sha256_bytes(content),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def verify_storage(root: Path, actor: str) -> Path:
    """Verify a probe created through another account and write a receipt."""
    if not actor.strip():
        raise ValueError("actor must not be blank")
    root = root.expanduser().resolve()
    missing = [
        directory for directory in DIRECTORIES if not (root / directory).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing storage directories: {', '.join(missing)}")

    probe_dir = root / PROBE_DIRECTORY
    manifest = json.loads((probe_dir / PROBE_MANIFEST).read_text(encoding="utf-8"))
    content = (probe_dir / manifest["probe_file"]).read_bytes()
    actual_hash = sha256_bytes(content)
    if actual_hash != manifest["sha256"] or len(content) != manifest["size_bytes"]:
        raise ValueError("Shared-storage probe checksum or size does not match")

    receipt = {
        "schema_version": 1,
        "verified_by": actor.strip(),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "probe_sha256": actual_hash,
        "status": "verified",
    }
    receipt_path = probe_dir / VERIFICATION_RECEIPT
    if receipt_path.exists():
        raise FileExistsError(
            f"Verification receipt already exists at {receipt_path}; refusing overwrite"
        )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path


def build_parser() -> argparse.ArgumentParser:
    """Build the shared-storage command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, required=True)
        subparser.add_argument("--actor", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run shared-storage initialization or verification."""
    args = build_parser().parse_args(argv)
    output = (
        initialize_storage(args.root, args.actor)
        if args.command == "init"
        else verify_storage(args.root, args.actor)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
