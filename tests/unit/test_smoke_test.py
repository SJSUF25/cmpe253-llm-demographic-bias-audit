from dataclasses import dataclass

from inference.smoke_test import format_prompt


@dataclass
class Encoding:
    input_ids: object


class FakeTokenizer:
    def __init__(self, result: object) -> None:
        self.result = result

    def apply_chat_template(self, messages: object, **kwargs: object) -> object:
        assert messages
        assert kwargs == {"add_generation_prompt": True, "return_tensors": "pt"}
        return self.result


def test_format_prompt_extracts_input_ids_from_batch_encoding() -> None:
    expected = object()

    assert format_prompt(FakeTokenizer(Encoding(expected)), "test") is expected


def test_format_prompt_preserves_legacy_tensor_result() -> None:
    expected = object()

    assert format_prompt(FakeTokenizer(expected), "test") is expected
