"""Unit tests for the initial workflow command registry."""

from project_cli import WORKFLOW_COMMANDS, main


def test_all_required_workflow_commands_are_registered() -> None:
    expected = {
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
    }

    assert set(WORKFLOW_COMMANDS) == expected


def test_unimplemented_command_fails_clearly(capsys) -> None:
    assert main(["download-data"]) == 2
    captured = capsys.readouterr()
    assert "registered but not implemented yet" in captured.out
