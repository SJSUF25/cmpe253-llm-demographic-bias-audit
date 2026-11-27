"""The clean CPU fixture must run through the public project command."""

import json

from project_cli import main


def test_small_reproduction_is_deterministic_and_clearly_scoped(tmp_path) -> None:
    output = tmp_path / "report.json"
    assert main(["reproduce-small", "--output", str(output)]) == 0
    first = output.read_bytes()
    report = json.loads(first)
    assert report["status"] == "passed"
    assert report["scope"] == "synthetic_cpu_contract_fixture_not_model_evidence"
    assert (report["cab_rows"], report["cab_scenarios"]) == (12, 2)
    assert (report["bbq_items"], report["bbq_option_orders"]) == (2, 6)
    assert report["paired_accuracy"] == {
        "baseline_only_correct": 1,
        "mitigation_only_correct": 1,
        "mcnemar_exact_p_two_sided": 1.0,
    }
    assert main(["reproduce-small", "--output", str(output)]) == 0
    assert output.read_bytes() == first
