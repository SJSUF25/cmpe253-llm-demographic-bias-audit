"""Select one audited CAB development pair for an optional private live demo."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class LiveDemoError(ValueError):
    """The frozen development example or cached fallback is unavailable."""


# This is the first gender family in the frozen 85-item AI review packet. Its
# explicit pair was read during that review; model outputs still need inspection.
SCENARIO_ID = "f36d76e4a97528901aa1"


def reviewed_development_pair(root: Path) -> tuple[dict, dict]:
    """Return only the female/male explicit pair from the documented QA packet."""
    root = root.expanduser().resolve()
    packet_path = root / "private_outputs/data_qa/review_packet.json"
    validation_path = root / "manifests/data_validation_manifest.json"
    packet_bytes = packet_path.read_bytes()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    packet_sha = hashlib.sha256(packet_bytes).hexdigest()
    if validation.get("review_packet_sha256") != packet_sha:
        raise LiveDemoError("QA packet differs from structural validation")
    packet = json.loads(packet_bytes)
    matches = [
        item
        for item in packet["cab_development_families"]
        if item["scenario_id"] == SCENARIO_ID and item["attribute"] == "gender"
    ]
    if len(matches) != 1:
        raise LiveDemoError("Frozen CAB development scenario is missing")
    rows = [
        row for row in matches[0]["prompts"] if row["explicit_implicit"] == "explicit"
    ]
    by_group = {row["group"]: row for row in rows}
    if len(rows) != 2 or set(by_group) != {"female", "male"}:
        raise LiveDemoError("Expected female/male CAB pair is incomplete")
    if any(
        row["split"] != "development" or row["scenario_id"] != SCENARIO_ID
        for row in rows
    ):
        raise LiveDemoError("Live example is not from the frozen development family")
    normalized = []
    for group in ("female", "male"):
        row = by_group[group]
        text = row["prompt_text"]
        markers = row["identity_values"]
        if len(markers) != 1 or markers[0].lower() != group:
            raise LiveDemoError("Unexpected identity marker in live pair")
        replaced, count = re.subn(
            rf"\b{re.escape(markers[0])}\b",
            "[IDENTITY]",
            text,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise LiveDemoError("Identity marker does not occur exactly once")
        normalized.append(replaced)
    if normalized[0] != normalized[1]:
        raise LiveDemoError("CAB live pair changes more than identity")
    return by_group["female"], by_group["male"]


def cached_fallback(root: Path) -> dict:
    """Verify the safe numeric bundle and return it for instant fallback display."""
    root = root.expanduser().resolve()
    bundle_path = root / "demo_assets/bundle.json"
    manifest_path = root / "demo_assets/manifest.json"
    raw = bundle_path.read_bytes()
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = json.loads(raw)
    if (
        receipt.get("bundle_sha256") != hashlib.sha256(raw).hexdigest()
        or bundle.get("mode") != "cached_offline"
        or "fairness" not in bundle.get("sections", {})
    ):
        raise LiveDemoError("Cached demo bundle is missing or changed")
    return bundle
