"""The secondary summary rejects changed private scorer outputs."""

import hashlib
import json

import pytest

from scoring.cab_summary import CabSummaryError, _load_run


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def test_summary_input_requires_exact_complete_pair_inventory(tmp_path) -> None:
    run_id = "CAB-EXP-M1-B0"
    rows_path = tmp_path / "private_outputs/scored_cab" / f"{run_id}.jsonl"
    pairs_path = tmp_path / "private_outputs/scored_cab" / f"{run_id}_pairs.jsonl"
    row_hash = _jsonl(
        rows_path,
        [
            {
                "run_id": run_id,
                "prompt_id": f"p{index}",
                "missingness_reason": None,
            }
            for index in range(640)
        ],
    )
    pair_hash = _jsonl(
        pairs_path,
        [{"run_id": run_id, "missingness_reason": None} for _ in range(724)],
    )
    manifest_path = tmp_path / "manifests/scoring" / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "split": "test",
                "rows": 640,
                "pairs": 724,
                "missing_rows": 0,
                "scored_output_sha256": row_hash,
                "pair_output_sha256": pair_hash,
            }
        )
    )
    _, rows, pairs = _load_run(tmp_path, run_id)
    assert len(rows) == 640 and len(pairs) == 724
    pairs_path.write_text(pairs_path.read_text() + "{}\n")
    with pytest.raises(CabSummaryError, match="hashes differ"):
        _load_run(tmp_path, run_id)
