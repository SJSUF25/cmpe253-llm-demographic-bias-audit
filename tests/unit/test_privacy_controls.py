from dataclasses import dataclass

import pytest

from privacy.controls import (
    PrivacyControlError,
    Span,
    filter_output,
    load_config,
    prepare_input,
    resolve_spans,
)
from privacy.synthetic_data import generate_records


@dataclass
class Detection:
    start: int
    end: int
    entity_type: str


class FakeAnalyzer:
    def __init__(self, results=()):
        self.results = results

    def analyze(self, **_kwargs):
        return self.results


@pytest.fixture
def record():
    return generate_records(
        {
            "schema_version": 1,
            "seed": 253,
            "tasks": [
                "summarization",
                "email_rewrite",
                "case_note",
                "action_items",
                "json_extraction",
            ],
            "records_per_task": 80,
            "development_per_task": 40,
        }
    )[0]


def test_p0_is_unchanged_and_p1_matches_reference(record):
    raw, raw_audit = prepare_input(record, "P0")
    assert raw == record["input_text"]
    assert raw_audit["replacement_counts"] == {}
    redacted, audit = prepare_input(record, "P1", analyzer=FakeAnalyzer())
    assert redacted == record["reference"]["redacted_input"]
    assert set(audit["replacement_counts"].values()) == {1}
    assert len(audit["replacement_counts"]) == 5


def test_known_field_beats_wrong_presidio_type_and_ticket_is_preserved(record):
    text = record["input_text"]
    email = record["pii"]["EMAIL_ADDRESS"]
    start = text.index(email)
    ticket = text.index("TKT-0000")
    analyzer = FakeAnalyzer(
        [
            Detection(start, start + len(email), "LOCATION"),
            Detection(ticket, ticket + 8, "PERSON"),
        ]
    )
    redacted, _ = prepare_input(record, "P1", analyzer=analyzer)
    assert redacted == record["reference"]["redacted_input"]
    assert "TKT-0000" in redacted


def test_p2_catches_normalized_known_values_and_presidio_new_pii(record):
    person = record["pii"]["PERSON"]
    obfuscated = person.upper().replace(" ", "---")
    output = f"{obfuscated}; also Jane Doe."
    start = output.index("Jane Doe")
    analyzer = FakeAnalyzer([Detection(start, start + 8, "PERSON")])
    preserved, _ = filter_output(output, record, "P1", analyzer=analyzer)
    assert preserved == output
    filtered, audit = filter_output(output, record, "P2", analyzer=analyzer)
    assert filtered == "[PERSON]; also [PERSON]."
    assert audit["replacement_counts"] == {"PERSON": 2}


def test_longest_span_then_known_source_then_entity_priority():
    priority = load_config()["entity_priority"]
    spans = [
        Span(0, 9, "EMAIL_ADDRESS", "exact"),
        Span(0, 9, "ADDRESS", "presidio"),
        Span(2, 7, "PERSON", "presidio"),
    ]
    assert resolve_spans(spans, 9, priority) == [spans[0]]
    assert (
        resolve_spans(
            [
                Span(0, 9, "EMAIL_ADDRESS", "presidio"),
                Span(0, 9, "ADDRESS", "presidio"),
            ],
            9,
            priority,
        )[0].entity
        == "ADDRESS"
    )


def test_invalid_condition_fails(record):
    with pytest.raises(PrivacyControlError, match="condition"):
        prepare_input(record, "P3")
