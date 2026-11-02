"""Command entry points for the reproducible project workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from data.download import DEFAULT_CONFIG, download_datasets

WORKFLOW_COMMANDS = (
    "download-data",
    "build-data",
    "validate-data",
    "pilot",
    "infer-baseline",
    "infer-mitigation",
    "score",
    "annotate-export",
    "analyze",
    "explain",
    "privacy",
    "demo",
    "reproduce-small",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level workflow argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-bias-audit",
        description="Reproducible CMPE 253 LLM assurance pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download-data")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    download.add_argument("--verify-remote", action="store_true")
    for command in WORKFLOW_COMMANDS:
        if command != "download-data":
            subparsers.add_parser(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a workflow command or fail clearly while it is unimplemented."""
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        print(
            download_datasets(
                args.root,
                args.config,
                verify_remote=args.verify_remote,
            )
        )
        return 0
    print(
        f"Workflow command '{args.command}' is registered but not implemented yet. "
        "Consult PROJECT_TASKS.md for its implementation task."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
