import hashlib
import json
from pathlib import Path

from privacy import metrics
from privacy.synthetic_data import generate_records


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    ).encode()
    path.write_bytes(content)
    return content


def test_known_exposure_exact_and_normalized():
    values = {"PERSON": "ZovqelA TavqorA"}
    assert metrics.known_exposure("ZovqelA TavqorA", values)["PERSON"] == {
        "exact": True,
        "normalized": True,
    }
    assert metrics.known_exposure("ZOVQELA---TAVQORA", values)["PERSON"] == {
        "exact": False,
        "normalized": True,
    }


def test_aggregate_six_complete_private_cells(monkeypatch, tmp_path):
    config_path = Path(__file__).parents[2] / "configs/privacy_data.json"
    config = json.loads(config_path.read_text())
    source = [
        {**row, "split": "test"}
        for row in generate_records(config)
        if row["split"] == "development"
    ]
    data_path = tmp_path / config["output_directory"] / "test.jsonl"
    data_bytes = _write_jsonl(data_path, source)
    synthetic_manifest = tmp_path / config["manifest_path"]
    synthetic_manifest.parent.mkdir(parents=True)
    synthetic_manifest.write_text(
        json.dumps(
            {
                "config_sha256": _sha(config_path.read_bytes()),
                "files": {
                    str(data_path.relative_to(tmp_path)): {
                        "sha256": _sha(data_bytes),
                        "rows": 200,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(metrics.runner, "_check_test_gate", lambda _root: None)
    for model in ("M1", "M2"):
        for condition in ("P0", "P1", "P2"):
            run_id = f"PII-{model}-{condition}"
            entry = metrics.evaluation._registered(run_id)
            output_path = tmp_path / entry["output"]
            raw_rows = []
            filtered_rows = []
            for index, record in enumerate(source):
                text = "Safe case summary."
                if condition == "P0" and index == 0:
                    text += f" {record['pii']['EMAIL_ADDRESS']}"
                raw_hash = _sha(text.encode())
                raw_rows.append(
                    {
                        "prompt_id": record["record_id"],
                        "status": "ok",
                        "raw_output": text,
                        "output_sha256": raw_hash,
                    }
                )
                filtered_rows.append(
                    {
                        "record_id": record["record_id"],
                        "run_id": run_id,
                        "raw_output_sha256": raw_hash,
                        "filtered_output": text,
                        "filtered_output_sha256": raw_hash,
                    }
                )
            output_bytes = _write_jsonl(output_path, raw_rows)
            run_manifest_path = tmp_path / "manifests/inference" / f"{run_id}.json"
            run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            run_manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_rows": 200,
                        "output_sha256": _sha(output_bytes),
                    }
                )
            )
            input_audit = {
                "run_id": run_id,
                "records": 200,
                "input_file_sha256": _sha(data_bytes),
                "run_manifest_sha256": _sha(run_manifest_path.read_bytes()),
                "control_config_sha256": metrics.controls.config_sha256(),
                "redaction_reference_match_count": 0 if condition == "P0" else 200,
                "input_replacement_counts": (
                    {}
                    if condition == "P0"
                    else dict.fromkeys(metrics.controls.TOKENS, 200)
                ),
            }
            audit_path = tmp_path / "manifests/privacy" / f"{run_id}_input.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(input_audit))
            if condition == "P2":
                filtered_path = output_path.with_name(
                    output_path.stem + "_filtered.jsonl"
                )
                filtered_bytes = _write_jsonl(filtered_path, filtered_rows)
                filter_manifest = {
                    "raw_output_sha256": _sha(output_bytes),
                    "source_run_manifest_sha256": _sha(run_manifest_path.read_bytes()),
                    "filtered_output_sha256": _sha(filtered_bytes),
                    "filtered_output_path": str(filtered_path.relative_to(tmp_path)),
                }
                (tmp_path / "manifests/privacy" / f"{run_id}_filter.json").write_text(
                    json.dumps(filter_manifest)
                )
    result = json.loads(metrics.score_outputs(tmp_path).read_text())
    assert result["condition_output_count"] == 1200
    assert len(result["cells"]) == 30
    assert len(result["input_redaction_by_type"]) == 20
    exposed = [
        cell
        for cell in result["cells"]
        if cell["condition"] == "P0" and cell["task"] == "summarization"
    ]
    assert [cell["normalized_record_exposures"] for cell in exposed] == [1, 1]
    assert all(
        cell["normalized_record_exposures"] == 0
        for cell in result["cells"]
        if cell["condition"] != "P0"
    )
