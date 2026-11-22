import json
import sys
from types import SimpleNamespace

import pytest

from inference import adapter


class FakeVector:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        return FakeScalar(self.values[index])


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMatrix:
    def __init__(self, values):
        self.values = values
        self.ndim = 2
        self.shape = (1, len(values))

    def to(self, _device):
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            _row, column = index
            return FakeVector(self.values[column])
        return FakeVector(self.values)


class FakeTokenizer:
    chat_template = "official-template"
    eos_token_id = 10

    def __init__(self, input_length=3):
        self.input_length = input_length
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return {"input_ids": FakeMatrix(list(range(self.input_length)))}

    def decode(self, ids, **kwargs):
        assert ids.tolist() == [9, 10]
        assert kwargs["skip_special_tokens"] is True
        return " answer "


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return FakeMatrix(input_ids[0].tolist() + [9, 10])


def make_adapter(input_length=3):
    config = adapter.load_config()
    spec = adapter.resolve_model("M1")
    tokenizer = FakeTokenizer(input_length)
    model = FakeModel()
    return adapter.ModelAdapter(spec, tokenizer, model, config), tokenizer, model


def test_pinned_models_and_conditions():
    for key in ("M1", "M2"):
        spec = adapter.resolve_model(key)
        assert len(spec.revision) == 40
        assert spec.tokenizer_revision == spec.revision
    with pytest.raises(adapter.AdapterError, match="Unknown"):
        adapter.resolve_model("M3")
    config = adapter.load_config()
    assert adapter.decode_config(config=config).do_sample is False
    assert adapter.decode_config(sensitivity_seed=42, config=config).top_p == 0.9
    with pytest.raises(adapter.AdapterError, match="Unregistered"):
        adapter.decode_config(sensitivity_seed=43, config=config)


def test_generation_uses_official_template_and_returns_only_new_tokens(monkeypatch):
    monkeypatch.setattr(adapter, "_seed_rng", lambda _seed: None)
    loaded, tokenizer, model = make_adapter()
    result = loaded.generate("A test prompt", "baseline")
    assert result["raw_output"] == " answer "
    assert result["normalized_output"] == "answer"
    assert result["input_token_count"] == 3
    assert result["output_token_count"] == 2
    assert result["finish_reason"] == "eos"
    assert result["model_revision"] == loaded.spec.revision
    assert result["tokenizer_revision"] == loaded.spec.tokenizer_revision
    assert tokenizer.messages == [
        {
            "role": "system",
            "content": loaded.config["conditions"]["baseline"]["system_instruction"],
        },
        {"role": "user", "content": "A test prompt"},
    ]
    assert tokenizer.template_kwargs["truncation"] is False
    assert tokenizer.template_kwargs["add_generation_prompt"] is True
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs
    assert model.kwargs["max_new_tokens"] == 256


def test_overlength_or_changed_policy_fails_before_generation(monkeypatch):
    monkeypatch.setattr(adapter, "_seed_rng", lambda _seed: None)
    loaded, _, _ = make_adapter(4097)
    with pytest.raises(adapter.AdapterError, match="maximum"):
        loaded.generate("Too long", "baseline")
    loaded, _, _ = make_adapter()
    with pytest.raises(adapter.AdapterError, match="Unregistered"):
        loaded.generate(
            "Prompt",
            "baseline",
            decoding=adapter.DecodeConfig(999, True, 0.7, 0.9, 256),
        )


def test_changed_system_instruction_is_rejected(tmp_path):
    config = adapter.load_config()
    config["conditions"]["mitigation"]["system_instruction"] += " changed"
    path = tmp_path / "inference.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="Changed mitigation"):
        adapter.load_config(path)


def test_loader_uses_pinned_fp16_eager_and_rejects_cpu_offload(monkeypatch):
    calls = {}
    dtype = object()
    tokenizer = FakeTokenizer()

    class LoadedModel:
        device = "cuda:0"
        is_quantized = False
        hf_device_map = {"": "cuda:0"}

        def eval(self):
            return self

    LoadedModel.dtype = dtype

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls["tokenizer"] = (repo, kwargs)
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls["model"] = (repo, kwargs)
            return LoadedModel()

    fake_torch = SimpleNamespace(
        float16=dtype,
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM
        ),
    )
    monkeypatch.setattr(adapter, "configure_determinism", lambda _seed: None)
    loaded = adapter.load_model("M1")
    assert loaded.spec.key == "M1"
    assert calls["tokenizer"][1] == {
        "revision": loaded.spec.revision,
        "trust_remote_code": False,
    }
    assert calls["model"][1] == {
        "revision": loaded.spec.revision,
        "dtype": dtype,
        "device_map": "auto",
        "attn_implementation": "eager",
        "trust_remote_code": False,
    }
    LoadedModel.hf_device_map = {"": "cpu"}
    with pytest.raises(adapter.AdapterError, match="offloaded"):
        adapter.load_model("M1")
