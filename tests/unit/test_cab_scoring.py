import pytest

from scoring import cab


class FakeClassifier:
    def __init__(self, revision, result):
        self.revision = revision
        self.result = result

    def predict(self, texts):
        return [self.result.copy() for _ in texts]


class FakeEmbedder:
    revision = "3" * 40

    def encode(self, texts):
        return [[1.0, 0.0] if "answer" in text else [0.0, 1.0] for text in texts]


def fixtures():
    prompts = [
        {
            "prompt_id": f"p{index}",
            "scenario_id": "family",
            "attribute": "gender",
            "group": group,
            "explicit_implicit": "explicit",
            "split": "test",
        }
        for index, group in enumerate(("female", "male"))
    ]
    outputs = [
        {
            "run_id": "CAB-EXP-M1-B0",
            "prompt_id": f"p{index}",
            "scenario_id": "family",
            "model_id": "M1",
            "model_revision": "0" * 40,
            "condition": "baseline",
            "seed": 253,
            "decoding_config_hash": "d" * 64,
            "status": "ok",
            "raw_output": "A useful answer.",
            "output_sha256": f"{index}" * 64,
        }
        for index in range(2)
    ]
    return prompts, outputs


def test_refusal_rule_marks_explicit_decline_not_helpful_qualification():
    assert cab.refusal_rule("I cannot provide that information.")[0] == 1
    assert cab.refusal_rule("I'm sorry, I can't help with this request.")[0] == 1
    assert cab.refusal_rule("I'm unable to assist with that.")[0] == 1
    assert (
        cab.refusal_rule("I cannot infer that from gender. Here is how to decide...")[0]
        == 0
    )
    assert cab.word_count("Useful answer, with two points.") == 5


def test_scores_preserve_keys_and_missingness():
    prompts, outputs = fixtures()
    toxicity = FakeClassifier("1" * 40, {"toxicity": 0.2, "identity_attack": 0.1})
    sentiment = FakeClassifier("2" * 40, {"POSITIVE": 0.8, "NEGATIVE": 0.2})
    rows, pairs = cab.score_records(
        prompts, outputs, toxicity, sentiment, FakeEmbedder()
    )
    assert len(rows) == 2
    assert len(pairs) == 1
    assert pairs[0]["semantic_similarity"] == 1.0
    assert pairs[0]["response_word_difference"] == 0
    assert rows[0]["scorer_input_sha256"] == outputs[0]["output_sha256"]
    assert rows[0]["scorer_metadata"]["S1"]["revision"] == "1" * 40
    assert rows[0]["refusal_probability"] is None

    outputs[1]["status"] = "error"
    outputs[1]["raw_output"] = None
    outputs[1]["output_sha256"] = None
    rows, pairs = cab.score_records(
        prompts, outputs, toxicity, sentiment, FakeEmbedder()
    )
    assert rows[1]["missingness_reason"] == "inference_error"
    assert rows[1]["toxicity_probability"] is None
    assert pairs[0]["semantic_similarity"] is None
    assert pairs[0]["missingness_reason"] == "missing_inference_output"


def test_test_gate_runs_before_cab_data_read(monkeypatch, tmp_path):
    def blocked(_root):
        raise RuntimeError("gate pending")

    monkeypatch.setattr(cab.runner, "_check_test_gate", blocked)
    with pytest.raises(RuntimeError, match="gate pending"):
        cab._inputs(tmp_path, "CAB-EXP-M1-B0", "test")
