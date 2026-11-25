"""Deterministic P0/P1/P2 redaction for the synthetic privacy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from privacy.synthetic_data import TOKENS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/privacy_controls.json"
ENTITY_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "LOCATION": "ADDRESS",
    "ADDRESS": "ADDRESS",
    "ACCOUNT_ID": "ACCOUNT_ID",
    "POSTAL_CODE": "ADDRESS",
}


class PrivacyControlError(ValueError):
    """A redaction control or its pinned configuration is invalid."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity: str
    source: str


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("placeholders") != TOKENS:
        raise PrivacyControlError("Unexpected privacy-control configuration")
    if set(config["entity_priority"]) != set(TOKENS):
        raise PrivacyControlError("Entity-priority list does not match PII schema")
    return config


def config_sha256(path: Path = DEFAULT_CONFIG) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_analyzer(config: dict[str, Any] | None = None) -> Any:
    """Load pinned Presidio + spaCy locally, without fetching a suffix list."""
    config = config or load_config()
    os.environ.setdefault("TLDEXTRACT_PUBLIC_SUFFIX_LIST_URLS", "")
    os.environ.setdefault("TLDEXTRACT_CACHE", str(ROOT / ".privacy-cache"))
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import SpacyNlpEngine
    except ImportError as error:
        raise PrivacyControlError(
            "Install the frozen Colab dependency group"
        ) from error

    nlp = SpacyNlpEngine(
        models=[{"lang_code": config["language"], "model_name": config["spacy_model"]}]
    )
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=[config["language"]])
    for entity, expression in config["patterns"].items():
        recognizer = PatternRecognizer(
            supported_entity=entity,
            name=f"synthetic_{entity.lower()}_v1",
            patterns=[
                Pattern(name=f"{entity.lower()}_format_v1", regex=expression, score=1.0)
            ],
        )
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


def _exact_spans(text: str, values: dict[str, str]) -> list[Span]:
    spans = []
    for entity, value in values.items():
        if not value or entity not in TOKENS:
            raise PrivacyControlError("Missing or unsupported synthetic PII value")
        spans.extend(
            Span(match.start(), match.end(), entity, "exact")
            for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE)
        )
    return spans


def _normalized_chars(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars = []
    offsets = []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            if normalized.isalnum():
                chars.append(normalized)
                offsets.append((index, index + 1))
    return "".join(chars), offsets


def _normalized_spans(text: str, values: dict[str, str]) -> list[Span]:
    normalized, offsets = _normalized_chars(text)
    spans = []
    for entity, value in values.items():
        needle, _ = _normalized_chars(value)
        if not needle:
            raise PrivacyControlError("Normalized synthetic PII is empty")
        cursor = 0
        while (position := normalized.find(needle, cursor)) >= 0:
            spans.append(
                Span(
                    offsets[position][0],
                    offsets[position + len(needle) - 1][1],
                    entity,
                    "normalized_exact",
                )
            )
            cursor = position + 1
    return spans


def _presidio_spans(
    text: str, analyzer: Any, config: dict[str, Any], known: list[Span] | None = None
) -> list[Span]:
    requested = list(dict.fromkeys([*config["presidio_entities"], *config["patterns"]]))
    results = analyzer.analyze(
        text=text, entities=requested, language=config["language"], score_threshold=0.0
    )
    candidates = [
        Span(result.start, result.end, ENTITY_MAP[result.entity_type], "presidio")
        for result in results
        if result.entity_type in ENTITY_MAP
        and 0 <= result.start < result.end <= len(text)
    ]
    if known is not None:
        # This synthetic input has a complete field-level truth map. Restrict
        # broad NER detections to those fields so ticket IDs remain task content.
        candidates = [
            span
            for span in candidates
            if any(
                item.start <= span.start
                and span.end <= item.end
                and span.entity == item.entity
                for item in known
            )
        ]
    return candidates


def resolve_spans(
    spans: Iterable[Span], text_length: int, priority: Sequence[str]
) -> list[Span]:
    """Take longest spans, then known-value source, entity priority, and position."""
    rank = {entity: index for index, entity in enumerate(priority)}
    source_rank = {"exact": 0, "normalized_exact": 1, "presidio": 2}
    candidates = list(set(spans))
    if any(
        span.entity not in rank or not (0 <= span.start < span.end <= text_length)
        for span in candidates
    ):
        raise PrivacyControlError("Invalid PII detection span")
    selected: list[Span] = []
    for candidate in sorted(
        candidates,
        key=lambda s: (
            -(s.end - s.start),
            source_rank[s.source],
            rank[s.entity],
            s.start,
            s.end,
        ),
    ):
        if all(
            candidate.end <= prior.start or candidate.start >= prior.end
            for prior in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda span: (span.start, span.end))


def _replace(
    text: str, spans: list[Span], placeholders: dict[str, str]
) -> tuple[str, dict]:
    pieces = []
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    cursor = 0
    for span in spans:
        pieces.extend((text[cursor : span.start], placeholders[span.entity]))
        counts[span.entity] += 1
        sources[span.source] += 1
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), {
        "replacement_counts": dict(sorted(counts.items())),
        "detection_sources": dict(sorted(sources.items())),
    }


def redact(
    text: str,
    values: dict[str, str],
    *,
    analyzer: Any,
    config: dict[str, Any] | None = None,
    normalized_exact: bool = False,
    known_only: bool = False,
) -> tuple[str, dict]:
    config = config or load_config()
    spans = _exact_spans(text, values)
    spans.extend(_presidio_spans(text, analyzer, config, spans if known_only else None))
    if normalized_exact:
        spans.extend(_normalized_spans(text, values))
    resolved = resolve_spans(spans, len(text), config["entity_priority"])
    return _replace(text, resolved, config["placeholders"])


def prepare_input(
    record: dict,
    condition: str,
    *,
    analyzer: Any | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """P0 passes the source through; P1/P2 share the same input redaction."""
    if condition not in {"P0", "P1", "P2"}:
        raise PrivacyControlError("Unknown privacy condition")
    config = config or load_config()
    if condition == "P0":
        return record["input_text"], {
            "condition": condition,
            "config_sha256": config_sha256(),
            "replacement_counts": {},
            "detection_sources": {},
        }
    if analyzer is None:
        analyzer = make_analyzer(config)
    redacted, audit = redact(
        record["input_text"],
        record["pii"],
        analyzer=analyzer,
        config=config,
        known_only=True,
    )
    return redacted, {"condition": condition, "config_sha256": config_sha256(), **audit}


def filter_output(
    output: str,
    record: dict,
    condition: str,
    *,
    analyzer: Any | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """P2 removes residual output PII; P0/P1 preserve the raw model response."""
    if condition not in {"P0", "P1", "P2"}:
        raise PrivacyControlError("Unknown privacy condition")
    config = config or load_config()
    if condition != "P2":
        return output, {
            "condition": condition,
            "config_sha256": config_sha256(),
            "replacement_counts": {},
            "detection_sources": {},
        }
    if analyzer is None:
        analyzer = make_analyzer(config)
    redacted, audit = redact(
        output,
        record["pii"],
        analyzer=analyzer,
        config=config,
        normalized_exact=True,
    )
    return redacted, {
        "condition": condition,
        "config_sha256": config_sha256(),
        **audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        analyzer = make_analyzer()
        from privacy.synthetic_data import generate_records

        data_config = json.loads((ROOT / "configs/privacy_data.json").read_text())
        record = generate_records(data_config)[0]
        redacted, audit = prepare_input(record, "P1", analyzer=analyzer)
        if redacted != record["reference"]["redacted_input"]:
            raise PrivacyControlError("Presidio smoke redaction differs from reference")
        print(
            json.dumps({"status": "passed", "record_id": record["record_id"], **audit})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
