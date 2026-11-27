"""Command entry points for the reproducible project workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from data.bbq_sampler import DEFAULT_CONFIG as DEFAULT_BBQ_CONFIG
from data.bbq_sampler import build_bbq
from data.cab_builder import DEFAULT_CONFIG as DEFAULT_CAB_CONFIG
from data.cab_builder import build_cab
from data.download import DEFAULT_CONFIG, download_datasets
from data.validate import validate_data
from reproduction import DEFAULT_OUTPUT as DEFAULT_REPRODUCTION_OUTPUT
from reproduction import reproduce

WORKFLOW_COMMANDS = (
    "download-data",
    "build-data",
    "build-bbq",
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
    build = subparsers.add_parser("build-data")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--config", type=Path, default=DEFAULT_CAB_CONFIG)
    bbq = subparsers.add_parser("build-bbq")
    bbq.add_argument("--root", type=Path, required=True)
    bbq.add_argument("--config", type=Path, default=DEFAULT_BBQ_CONFIG)
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--root", type=Path, required=True)
    reproduction = subparsers.add_parser("reproduce-small")
    reproduction.add_argument(
        "--output", type=Path, default=DEFAULT_REPRODUCTION_OUTPUT
    )
    for command in WORKFLOW_COMMANDS:
        if command not in {
            "download-data",
            "build-data",
            "build-bbq",
            "validate-data",
            "reproduce-small",
        }:
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
    if args.command == "build-data":
        print(build_cab(args.root, args.config))
        return 0
    if args.command == "build-bbq":
        print(build_bbq(args.root, args.config))
        return 0
    if args.command == "validate-data":
        print(validate_data(args.root))
        return 0
    if args.command == "reproduce-small":
        print(reproduce(args.output))
        return 0
    print(
        f"Workflow command '{args.command}' is registered but not implemented yet. "
        "Consult PROJECT_TASKS.md for its implementation task."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
