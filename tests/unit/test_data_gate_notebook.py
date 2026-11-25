"""Exercise the operator notebook's fresh review log on a private fixture."""

import ast
import hashlib
import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "notebooks/e1_t06_data_gate_colab.ipynb"
)


def test_notebook_cells_compile_and_have_no_saved_outputs():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 22
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    approval_source = "".join(notebook["cells"][20]["source"])
    assert "\n# approve_data_gate()" in approval_source
    approval_ast = ast.parse(approval_source)
    gate_assignment = next(
        node
        for node in ast.walk(approval_ast)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gate"
            for target in node.targets
        )
    )
    assert {key.value for key in gate_assignment.value.keys} == {
        "schema_version",
        "validation_manifest_sha256",
        "token_length_manifest_sha256",
        "cab_implicit_quality_manifest_sha256",
        "governance_lock_sha256",
        "token_length_status",
        "human_review_status",
        "approved_by",
    }


def test_review_requires_six_exclusion_decisions_and_keeps_corrections(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    engine = "".join(notebook["cells"][14]["source"])
    sections = {
        "cab_development_families": 50,
        "bbq": 12,
        "pii_development": 10,
        "ordinary_train": 8,
        "member_canaries": 3,
        "nonmember_calibration": 2,
    }
    packet = {
        "cab_development_families": [
            {
                "scenario_id": f"CAB-{index}",
                "attribute": "gender",
                "prompts": [
                    {
                        "explicit_implicit": "explicit",
                        "group": "female",
                        "split": "development",
                        "prompt_text": "A review prompt",
                    }
                ],
            }
            for index in range(50)
        ],
        "bbq": [{"item_id": f"BBQ-{index}"} for index in range(12)],
        **{
            section: [{"record_id": f"{section}-{index}"} for index in range(count)]
            for section, count in sections.items()
            if section not in ("cab_development_families", "bbq")
        },
    }
    packet_path = tmp_path / "private_outputs/data_qa/review_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet))
    quality_path = tmp_path / "manifests/cab_implicit_quality_manifest.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text("{}")
    training_path = tmp_path / "private_outputs/membership/training_presentations.jsonl"
    training_path.parent.mkdir(parents=True)
    training_path.write_text("")
    namespace = {
        "SHARED": tmp_path,
        "REVIEWER": "Abhinand",
        "packet": packet,
        "packet_path": packet_path,
        "quality_path": quality_path,
        "sections": sections,
        "flagged_in_packet": {f"CAB-{index}" for index in range(6)},
        "Counter": Counter,
        "json": json,
        "sha": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with redirect_stdout(io.StringIO()):
        exec(engine, namespace)
        for section, count in sections.items():
            for index in range(count):
                namespace["inspect_item"](section, index)
                decision = (
                    "x" if section == "cab_development_families" and index < 6 else "p"
                )
                namespace["record"](section, index, decision)
        assert namespace["show_progress"]() is True
        namespace["inspect_item"]("bbq", 0)
        namespace["record"]("bbq", 0, "i", "Wrong label")
        assert namespace["show_progress"]() is False
        namespace["inspect_item"]("bbq", 0)
        namespace["record"]("bbq", 0, "p")
        assert namespace["show_progress"]() is True
    log_path = tmp_path / "private_outputs/data_qa/manual_review_abhinand_operator.json"
    log = json.loads(log_path.read_text())
    assert len(log["events"]) == 87
    assert [event["decision"] for event in log["events"][-2:]] == ["i", "p"]

    independent = {**namespace, "REVIEWER": "Vineet Malewar"}
    approval = "".join(notebook["cells"][20]["source"])
    exec(approval, independent)
    with pytest.raises(AssertionError, match="Only Abhinand"):
        independent["approve_data_gate"]()
