import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from inference import runner
from inference.adapter import decode_config, load_config, resolve_model
from inference.runner import RunnerError, RunSpec, run_inference


def prompt(index):
    text = f"Synthetic prompt {index}"
    return {
        "prompt_id": f"P-{index}",
        "scenario_id": f"S-{index}",
        "prompt_text": text,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


class FakeAdapter:
    def __init__(self, *, interrupt_at=None, transient_at=None, fail_at=None):
        self.spec = resolve_model("M1")
        self.config = load_config()
        self.model = SimpleNamespace(device="cuda:0")
        self.calls = Counter()
        self.interrupt_at = interrupt_at
        self.transient_at = transient_at
        self.fail_at = fail_at

    def generate(self, text, condition, *, decoding):
        self.calls[text] += 1
        if text == self.interrupt_at:
            raise KeyboardInterrupt
        if text == self.transient_at and self.calls[text] == 1:
            raise OSError("temporary accelerator failure")
        if text == self.fail_at:
            raise ValueError("invalid synthetic prompt")
        raw = f" response to {text} "
        return {
            "model_id": self.spec.key,
            "model_revision": self.spec.revision,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "condition": condition,
            "system_prompt_version": self.config["conditions"][condition]["version"],
            "precision": "float16",
            "device": "cuda:0",
            "seed": decoding.seed,
            "do_sample": decoding.do_sample,
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_new_tokens": decoding.max_new_tokens,
            "input_token_count": 8,
            "output_token_count": 5,
            "raw_output": raw,
            "normalized_output": raw.strip(),
            "finish_reason": "eos",
            "runtime_seconds": 0.01,
        }


def run(rows, adapter, root, *, interval=2, confirmatory=False, state=None):
    spec = RunSpec(
        run_id="CAB-TEST-M1-B0",
        model_id="M1",
        condition="baseline",
        output_path="private_outputs/cab_test_m1_b0.jsonl",
        confirmatory=confirmatory,
    )
    return run_inference(
        rows,
        adapter,
        spec,
        root,
        settings=decode_config(config=adapter.config),
        checkpoint_interval=interval,
        state_provider=state or (lambda: ("a" * 40, False)),
        environment_provider=lambda: {"gpu": {"name": "fixture GPU"}},
        gate_checker=lambda _root: None,
    )


def test_interruption_resumes_without_omissions_or_duplicates(tmp_path):
    rows = [prompt(index) for index in range(5)]
    interrupted = FakeAdapter(interrupt_at="Synthetic prompt 2")
    with pytest.raises(KeyboardInterrupt):
        run(rows, interrupted, tmp_path)
    parts = sorted((tmp_path / "private_outputs/cab_test_m1_b0/parts").glob("*.jsonl"))
    assert len(parts) == 1
    assert interrupted.calls["Synthetic prompt 0"] == 1
    resumed = FakeAdapter()
    manifest_path = run(rows, resumed, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_rows"] == 5
    assert [entry["rows"] for entry in manifest["partitions"]] == [2, 2, 1]
    assert resumed.calls["Synthetic prompt 0"] == 0
    final = tmp_path / "private_outputs/cab_test_m1_b0.jsonl"
    scored = [json.loads(line) for line in final.read_text().splitlines()]
    assert {row["prompt_id"] for row in scored} == {row["prompt_id"] for row in rows}
    assert len(scored) == 5
    assert run(rows, FakeAdapter(), tmp_path).read_bytes() != b""


def test_one_identical_retry_and_terminal_error_row(tmp_path):
    rows = [prompt(index) for index in range(3)]
    adapter = FakeAdapter(
        transient_at="Synthetic prompt 1", fail_at="Synthetic prompt 2"
    )
    manifest_path = run(rows, adapter, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert adapter.calls["Synthetic prompt 1"] == 2
    assert adapter.calls["Synthetic prompt 2"] == 1
    assert manifest["status"] == "complete_with_errors"
    assert manifest["error_summary"] == {"ValueError": 1}
    scored = [
        json.loads(line)
        for line in (tmp_path / "private_outputs/cab_test_m1_b0.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["retry_count"] for row in scored] == [0, 1, 0]
    assert scored[2]["status"] == "error"
    assert scored[2]["raw_output"] is None


def test_dirty_confirmatory_run_is_rejected_before_outputs(tmp_path):
    with pytest.raises(RunnerError, match="clean Git worktree"):
        run(
            [prompt(0)],
            FakeAdapter(),
            tmp_path,
            confirmatory=True,
            state=lambda: ("a" * 40, True),
        )
    assert list(tmp_path.iterdir()) == []


def test_generic_confirmatory_run_requires_gate_receipt(tmp_path):
    model_adapter = FakeAdapter()
    spec = RunSpec(
        run_id="CAB-TEST-M1-B0",
        model_id="M1",
        condition="baseline",
        output_path="private_outputs/cab_test_m1_b0.jsonl",
        confirmatory=True,
    )
    with pytest.raises(RunnerError, match="data-gate approval"):
        run_inference(
            [prompt(0)],
            model_adapter,
            spec,
            tmp_path,
            state_provider=lambda: ("a" * 40, False),
            environment_provider=lambda: {},
        )
    assert list(tmp_path.iterdir()) == []


def test_conflicting_duplicate_and_tampered_output_fail_on_resume(tmp_path):
    rows = [prompt(index) for index in range(4)]
    run(rows, FakeAdapter(), tmp_path)
    partition = tmp_path / "private_outputs/cab_test_m1_b0/parts/part-000001.jsonl"
    records = [json.loads(line) for line in partition.read_text().splitlines()]
    records[0]["raw_output"] = "changed"
    records[0]["normalized_output"] = "changed"
    partition.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(RunnerError, match="output hash mismatch"):
        run(rows, FakeAdapter(), tmp_path)


def test_conflicting_duplicate_key_fails_on_resume(tmp_path):
    rows = [prompt(index) for index in range(4)]
    run(rows, FakeAdapter(), tmp_path)
    parts = tmp_path / "private_outputs/cab_test_m1_b0/parts"
    first = json.loads((parts / "part-000000.jsonl").read_text().splitlines()[0])
    replacement = {**first, "raw_output": "different", "normalized_output": "different"}
    replacement["output_sha256"] = hashlib.sha256(b"different").hexdigest()
    second_path = parts / "part-000001.jsonl"
    second = [json.loads(line) for line in second_path.read_text().splitlines()]
    second[0] = replacement
    second_path.write_text("\n".join(json.dumps(row) for row in second) + "\n")
    with pytest.raises(RunnerError, match="Conflicting duplicate"):
        run(rows, FakeAdapter(), tmp_path)


def test_confirmatory_error_is_recorded_but_run_fails(tmp_path):
    with pytest.raises(RunnerError, match="completed with errors"):
        run(
            [prompt(0)],
            FakeAdapter(fail_at="Synthetic prompt 0"),
            tmp_path,
            confirmatory=True,
        )
    path = tmp_path / "manifests/inference/CAB-TEST-M1-B0.json"
    assert json.loads(path.read_text())["status"] == "complete_with_errors"


def test_registered_cab_development_filters_before_model_run(monkeypatch, tmp_path):
    config = json.loads(runner.CAB_CONFIG_PATH.read_text())
    path = tmp_path / config["output_path"]
    path.parent.mkdir(parents=True)
    records = [
        {**prompt(0), "split": "development", "explicit_implicit": "explicit"},
        {**prompt(1), "split": "test", "explicit_implicit": "explicit"},
        {**prompt(2), "split": "development", "explicit_implicit": "implicit"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    captured = {}

    def fake_run(selected, adapter, spec, root):
        captured.update(selected=selected, adapter=adapter, spec=spec, root=root)
        return tmp_path / "manifest.json"

    fake_adapter = FakeAdapter()
    monkeypatch.setattr(runner, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner, "load_model", lambda key: fake_adapter)
    monkeypatch.setattr(runner, "run_inference", fake_run)
    result = runner.run_registered_cab(tmp_path, "CAB-EXP-M1-B0", "development")
    assert result == tmp_path / "manifest.json"
    assert [row["prompt_id"] for row in captured["selected"]] == ["P-0"]
    assert captured["spec"].run_id == "CAB-EXP-M1-B0-DEV"
    assert captured["spec"].confirmatory is False


def test_registered_cab_test_requires_gate_before_loading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(
        runner, "load_model", lambda _key: pytest.fail("model loaded before gate")
    )
    with pytest.raises(RunnerError, match="data-gate approval"):
        runner.run_registered_cab(tmp_path, "CAB-EXP-M1-B0", "test")


def test_registered_cab_test_accepts_matching_review_receipt(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    validation = manifest_dir / "data_validation_manifest.json"
    validation.write_text(json.dumps({"structural_checks": "passed"}))
    config = json.loads(runner.CAB_CONFIG_PATH.read_text())
    bbq_config = json.loads((runner.ROOT / "configs/bbq_build.json").read_text())
    membership_config = json.loads(
        (runner.ROOT / "configs/membership_data.json").read_text()
    )
    input_paths = [
        config["output_path"],
        bbq_config["items_path"],
        bbq_config["permutations_path"],
        f"{membership_config['output_directory']}/training_presentations.jsonl",
    ]
    for relative in input_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    path = tmp_path / config["output_path"]
    path.write_text(
        json.dumps({**prompt(0), "split": "test", "explicit_implicit": "explicit"})
        + "\n"
    )
    input_hashes = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in input_paths
    }
    token_lengths = manifest_dir / "token_length_manifest.json"
    token_lengths.write_text(
        json.dumps(
            {
                "status": "passed",
                "validation_manifest_sha256": hashlib.sha256(
                    validation.read_bytes()
                ).hexdigest(),
                "inputs_sha256": input_hashes,
            }
        )
    )
    receipt = {
        "schema_version": 1,
        "validation_manifest_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "token_length_manifest_sha256": hashlib.sha256(
            token_lengths.read_bytes()
        ).hexdigest(),
        "governance_lock_sha256": hashlib.sha256(
            runner.GOVERNANCE_LOCK_PATH.read_bytes()
        ).hexdigest(),
        "token_length_status": "passed",
        "human_review_status": "approved",
        "approved_by": ["Vineet Malewar"],
    }
    (manifest_dir / "data_gate_approval.json").write_text(json.dumps(receipt))
    captured = {}

    def fake_run(selected, adapter, spec, root):
        captured.update(selected=selected, spec=spec)
        return tmp_path / "confirmed.json"

    monkeypatch.setattr(runner, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner, "load_model", lambda _key: FakeAdapter())
    monkeypatch.setattr(runner, "run_inference", fake_run)
    assert (
        runner.run_registered_cab(tmp_path, "CAB-EXP-M1-B0", "test")
        == tmp_path / "confirmed.json"
    )
    assert captured["spec"].confirmatory is True
    assert [row["prompt_id"] for row in captured["selected"]] == ["P-0"]
    valid_lengths = token_lengths.read_bytes()
    token_lengths.write_text(
        json.dumps(
            {
                "status": "failed",
                "validation_manifest_sha256": hashlib.sha256(
                    validation.read_bytes()
                ).hexdigest(),
            }
        )
    )
    with pytest.raises(RunnerError, match="incomplete or stale"):
        runner.run_registered_cab(tmp_path, "CAB-EXP-M1-B0", "test")
    token_lengths.write_bytes(valid_lengths)
    path.write_text("tampered\n")
    with pytest.raises(RunnerError, match="incomplete or stale"):
        runner.run_registered_cab(tmp_path, "CAB-EXP-M1-B0", "test")


def test_partition_written_before_manifest_is_recovered(monkeypatch, tmp_path):
    rows = [prompt(index) for index in range(3)]
    original = runner._atomic_write
    failed = False

    def fail_manifest_once(path, content):
        nonlocal failed
        if path.name == "CAB-TEST-M1-B0.json" and not failed:
            failed = True
            raise OSError("simulated interrupted manifest write")
        original(path, content)

    monkeypatch.setattr(runner, "_atomic_write", fail_manifest_once)
    with pytest.raises(OSError, match="interrupted manifest"):
        run(rows, FakeAdapter(), tmp_path)
    monkeypatch.setattr(runner, "_atomic_write", original)
    resumed = FakeAdapter()
    manifest = json.loads(run(rows, resumed, tmp_path).read_text())
    assert manifest["completed_rows"] == 3
    assert resumed.calls["Synthetic prompt 0"] == 0
    assert resumed.calls["Synthetic prompt 1"] == 0
    assert resumed.calls["Synthetic prompt 2"] == 1
