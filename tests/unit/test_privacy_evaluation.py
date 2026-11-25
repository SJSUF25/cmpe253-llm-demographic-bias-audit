import hashlib
import json
from pathlib import Path

import pytest

from privacy import evaluation
from privacy.synthetic_data import generate_records


class FakeAnalyzer:
    def analyze(self, **_kwargs):
        return []


@pytest.fixture
def development_records():
    config = json.loads(
        (Path(__file__).parents[2] / "configs/privacy_data.json").read_text()
    )
    return [row for row in generate_records(config) if row["split"] == "development"]


def test_prepare_registered_pii_prompts(development_records):
    raw, raw_audit = evaluation.make_prompts(development_records, "P0")
    redacted, redacted_audit = evaluation.make_prompts(
        development_records, "P1", analyzer=FakeAnalyzer()
    )
    assert len(raw) == len(redacted) == 200
    assert raw[0]["prompt_text"] != redacted[0]["prompt_text"]
    assert raw_audit["input_replacement_counts"] == {}
    assert set(redacted_audit["input_replacement_counts"].values()) == {200}
    assert all(row["control_config_sha256"] for row in redacted)


def test_test_split_gate_runs_before_private_data_read(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation.runner, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(
        evaluation.runner,
        "_check_test_gate",
        lambda _root: (_ for _ in ()).throw(ValueError("gate closed")),
    )
    monkeypatch.setattr(
        evaluation,
        "_rows",
        lambda _path: pytest.fail("test data read before gate"),
    )
    with pytest.raises(ValueError, match="gate closed"):
        evaluation.run_registered_privacy(tmp_path, "PII-M1-P0", "test")


def test_p2_private_filtered_artifact_links_raw_outputs(tmp_path, development_records):
    records = development_records
    output = tmp_path / "private_outputs/pilot/pii_m1_p2.jsonl"
    output.parent.mkdir(parents=True)
    rows = []
    for record in records:
        raw = f"Send to {record['pii']['EMAIL_ADDRESS']}"
        rows.append(
            {
                "prompt_id": record["record_id"],
                "status": "ok",
                "raw_output": raw,
                "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
        )
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    run_manifest = tmp_path / "manifests/inference/PII-M1-P2-DEV.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text("{}\n")
    manifest = evaluation.finalize_p2(
        tmp_path,
        "PII-M1-P2-DEV",
        records,
        analyzer=FakeAnalyzer(),
        output_path=output,
        run_manifest=run_manifest,
    )
    result = json.loads(manifest.read_text())
    filtered = evaluation._rows(tmp_path / result["filtered_output_path"])
    assert result["records"] == len(filtered) == 200
    assert result["replacement_counts"] == {"EMAIL_ADDRESS": 200}
    assert all("@example.test" not in row["filtered_output"] for row in filtered)
    assert manifest == evaluation.finalize_p2(
        tmp_path,
        "PII-M1-P2-DEV",
        records,
        analyzer=FakeAnalyzer(),
        output_path=output,
        run_manifest=run_manifest,
    )
