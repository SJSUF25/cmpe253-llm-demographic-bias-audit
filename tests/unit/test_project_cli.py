"""Unit tests for the initial workflow command registry."""

from project_cli import WORKFLOW_COMMANDS, main


def test_all_required_workflow_commands_are_registered() -> None:
    expected = {
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
    }

    assert set(WORKFLOW_COMMANDS) == expected


def test_unimplemented_command_fails_clearly(capsys) -> None:
    assert main(["validate-data"]) == 2
    captured = capsys.readouterr()
    assert "registered but not implemented yet" in captured.out


def test_download_command_dispatches(monkeypatch, tmp_path, capsys) -> None:
    expected = tmp_path / "manifest.json"

    def fake_download(root, config, *, verify_remote):
        assert root == tmp_path
        assert not verify_remote
        return expected

    monkeypatch.setattr("project_cli.download_datasets", fake_download)

    assert main(["download-data", "--root", str(tmp_path)]) == 0
    assert str(expected) in capsys.readouterr().out


def test_build_command_dispatches(monkeypatch, tmp_path, capsys) -> None:
    expected = tmp_path / "cab_manifest.json"

    def fake_build(root, config):
        assert root == tmp_path
        return expected

    monkeypatch.setattr("project_cli.build_cab", fake_build)

    assert main(["build-data", "--root", str(tmp_path)]) == 0
    assert str(expected) in capsys.readouterr().out


def test_build_bbq_command_dispatches(monkeypatch, tmp_path, capsys) -> None:
    expected = tmp_path / "bbq_manifest.json"

    def fake_build(root, config):
        assert root == tmp_path
        return expected

    monkeypatch.setattr("project_cli.build_bbq", fake_build)

    assert main(["build-bbq", "--root", str(tmp_path)]) == 0
    assert str(expected) in capsys.readouterr().out
