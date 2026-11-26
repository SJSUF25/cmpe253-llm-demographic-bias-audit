"""Deterministic fixtures for the gated E2-T04 pilot workflow."""

import json
from types import SimpleNamespace

from inference.pilot import _allocate, _run_bbq, build_selection


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_hamilton_allocation_is_complete_and_deterministic():
    availability = {("a",): 7, ("b",): 5, ("c",): 3}
    first = _allocate(8, availability)
    assert first == _allocate(8, availability)
    assert sum(first.values()) == 8
    assert all(first[key] <= availability[key] for key in first)


def test_selection_freezes_50_cab_families_and_100_stratified_bbq(tmp_path):
    cab_rows = []
    for attribute, groups in (
        ("gender", ("male", "female")),
        ("race", ("a", "b", "c", "d")),
    ):
        for family in range(30):
            scenario = f"{attribute}-{family:02d}"
            for group in groups:
                cab_rows.append(
                    {
                        "scenario_id": scenario,
                        "prompt_id": f"{scenario}-{group}",
                        "attribute": attribute,
                        "group": group,
                        "split": "development",
                        "explicit_implicit": "explicit",
                    }
                )
    cab_path = tmp_path / "private_outputs/processed_data/cab/prompt_manifest.jsonl"
    _write_jsonl(cab_path, cab_rows)

    categories = ("Gender_identity", "Race_ethnicity", "Race_x_gender")
    contexts = ("ambig", "disambig")
    polarities = ("neg", "nonneg")
    bbq_rows = []
    permutations = []
    for category in categories:
        for context in contexts:
            for polarity in polarities:
                for index in range(10):
                    item_id = f"{category}|{context}|{polarity}|{index}"
                    bbq_rows.append(
                        {
                            "item_id": item_id,
                            "category": category,
                            "context_condition": context,
                            "question_polarity": polarity,
                        }
                    )
                    permutations.extend(
                        {"item_id": item_id, "order": order} for order in range(3)
                    )
    bbq_path = tmp_path / "private_outputs/processed_data/bbq/items.jsonl"
    permutation_path = (
        tmp_path / "private_outputs/processed_data/bbq/permutations.jsonl"
    )
    _write_jsonl(bbq_path, bbq_rows)
    _write_jsonl(permutation_path, permutations)

    manifest_path = build_selection(tmp_path)
    first = manifest_path.read_bytes()
    manifest = json.loads(first)
    assert manifest["cab"]["families"] == 50
    assert manifest["cab"]["prompts"] == 150
    assert manifest["cab"]["attribute_counts"] == {"gender": 25, "race": 25}
    assert manifest["bbq"]["items"] == 100
    assert manifest["bbq"]["permutations"] == 300
    assert max(manifest["bbq"]["allocation"].values()) == 9
    assert min(manifest["bbq"]["allocation"].values()) == 8
    assert build_selection(tmp_path).read_bytes() == first


class _FakeBBQAdapter:
    spec = SimpleNamespace(key="M1", revision="r1", tokenizer_revision="r1")
    config = {
        "conditions": {
            "baseline": {"version": "baseline-v1"},
            "mitigation": {"version": "fairness-v1"},
        }
    }

    @staticmethod
    def conditional_log_likelihood(_prompt, answer, _condition):
        return {"first": -1.0, "second": -2.0, "unknown": -3.0}[answer]


def test_bbq_pilot_checkpoints_and_reuses_completed_manifest(tmp_path):
    items = []
    permutations = []
    for index in range(3):
        item_id = f"Gender_identity|{index}"
        items.append(
            {
                "item_id": item_id,
                "category": "Gender_identity",
                "template_id": str(index),
                "context": "Context",
                "question": "Question?",
                "ans0": "first",
                "ans1": "second",
                "ans2": "unknown",
                "label": 0,
                "target_loc": 1,
                "context_condition": "ambig",
                "question_polarity": "neg",
                "answer_info": {
                    "ans0": ["first", "non-target"],
                    "ans1": ["second", "target"],
                    "ans2": ["unknown", "unknown"],
                },
            }
        )
        answers = ["first", "second", "unknown"]
        for order in range(3):
            source = [(order + offset) % 3 for offset in range(3)]
            permutations.append(
                {
                    "item_id": item_id,
                    "order": order,
                    "answers": [answers[value] for value in source],
                    "source_indices": source,
                    "correct_index": source.index(0),
                }
            )
    adapter = _FakeBBQAdapter()
    manifest = _run_bbq(
        tmp_path,
        adapter,
        "baseline",
        items,
        permutations,
        checkpoint=2,
        commit="a" * 40,
    )
    first = manifest.read_bytes()
    result = json.loads(first)
    assert "manifests/pilot_v2/" in str(manifest)
    assert result["run_id"].startswith("PILOT2-BBQ-")
    assert result["completed_rows"] == 3
    assert [part["rows"] for part in result["partitions"]] == [2, 1]
    assert (
        _run_bbq(
            tmp_path,
            adapter,
            "baseline",
            items,
            permutations,
            checkpoint=2,
            commit="a" * 40,
        ).read_bytes()
        == first
    )
