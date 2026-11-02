"""Command entry points for the reproducible project workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    parser.add_argument("command", choices=WORKFLOW_COMMANDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a workflow command or fail clearly while it is unimplemented."""
    args = build_parser().parse_args(argv)
    print(
        f"Workflow command '{args.command}' is registered but not implemented yet. "
        "Consult PROJECT_TASKS.md for its implementation task."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
