"""Frozen CAB human-review allocation and blinded, resumable annotation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from inference import runner
from scoring import cab

GROUPS = {"gender": 2, "race": 4}
SYNONYMS = {
    "male": ("male", "man", "men", "boy", "boys"),
    "female": ("female", "woman", "women", "girl", "girls"),
    "white": ("white", "caucasian"),
    "black": ("black", "african american"),
    "asian": ("asian",),
    "hispanic": ("hispanic", "latino", "latina", "latinx"),
}


class AnnotationError(ValueError):
    """An allocation, source, or rating violates the locked review contract."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AnnotationError(f"Existing frozen artifact differs: {path}")
    else:
        path.write_bytes(content)


def select_families(
    rows: list[dict], split: str, per_attribute: int, purpose: str
) -> list[str]:
    """Apply the PRD Hamilton allocation by attribute and superdomain."""
    cells: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["split"] == split and row["explicit_implicit"] == "explicit":
            cells[row["attribute"]][row["superdomain"]].add(row["scenario_id"])
    chosen = []
    for attribute in GROUPS:
        domains = cells[attribute]
        total = sum(map(len, domains.values()))
        if total < per_attribute:
            raise AnnotationError(f"Too few {split} {attribute} families")
        quotas = {
            name: per_attribute * len(ids) / total for name, ids in domains.items()
        }
        counts = {name: int(quotas[name]) for name in domains}
        remainder = per_attribute - sum(counts.values())
        for name in sorted(domains, key=lambda n: (-(quotas[n] - counts[n]), n))[
            :remainder
        ]:
            counts[name] += 1
        for name, ids in sorted(domains.items()):
            ordered = sorted(
                ids, key=lambda sid: (_digest(f"253|{purpose}|{sid}"), sid)
            )
            chosen.extend(ordered[: counts[name]])
    if len(chosen) != 2 * per_attribute or len(set(chosen)) != len(chosen):
        raise AnnotationError("Invalid CAB annotation allocation")
    return sorted(chosen)


def _mask(text: str, prompt: dict, aliases: dict[str, str]) -> str:
    words = set(prompt["identity_values"]) | set(SYNONYMS[prompt["group"]])
    replacements = {word.casefold(): aliases[prompt["group"]] for word in words if word}
    for group, alias in aliases.items():
        for word in SYNONYMS[group]:
            replacements[word.casefold()] = alias
    if not replacements:
        return text
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(w) for w in sorted(replacements, key=len, reverse=True))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: replacements[match.group().casefold()], text)


def build_packet(root: Path, split: str, annotator: str) -> Path:
    """Freeze selection before output access, then export blinded cards."""
    if split not in ("development", "test") or not re.fullmatch(
        r"[a-z0-9_-]+", annotator
    ):
        raise AnnotationError("Use split=development/test and a lowercase annotator ID")
    root = root.expanduser().resolve()
    if split == "test":
        runner._check_test_gate(root)
    config = json.loads(cab.CAB_CONFIG.read_text())
    prompt_path = root / config["output_path"]
    build_path = root / config["manifest_path"]
    if not prompt_path.is_file() or not build_path.is_file():
        raise AnnotationError("CAB build is missing")
    build = json.loads(build_path.read_text())
    if (
        build.get("output_sha256")
        != hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    ):
        raise AnnotationError("CAB build hash mismatch")
    all_prompts = runner._read_jsonl(prompt_path)
    count = 24 if split == "test" else 10
    purpose = "annotation48" if split == "test" else "annotation20-development"
    selected = select_families(all_prompts, split, count, purpose)
    directory = root / "private_outputs/annotation" / annotator / split
    allocation = directory / "allocation.json"
    _write_once(
        allocation,
        _bytes(
            {
                "schema_version": 1,
                "purpose": purpose,
                "scenario_ids": selected,
                "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            }
        ),
    )
    selected_set = set(selected)
    prompts_by_id = {}
    outputs_by_key = {}
    source_hashes = {}
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            run_id = f"CAB-EXP-{model}-{condition}"
            prompts, outputs, source = cab._inputs(root, run_id, split)
            source_hashes[run_id] = source
            for prompt in prompts:
                if prompt["scenario_id"] in selected_set:
                    prompts_by_id[prompt["prompt_id"]] = prompt
            for output in outputs:
                if output["scenario_id"] in selected_set:
                    outputs_by_key[(model, condition, output["prompt_id"])] = output
    families = defaultdict(list)
    for prompt in prompts_by_id.values():
        families[prompt["scenario_id"]].append(prompt)
    response_cards, set_cards, key = [], [], []
    for sid in selected:
        variants = families[sid]
        attribute = variants[0]["attribute"]
        if len(variants) != GROUPS[attribute]:
            raise AnnotationError(f"Incomplete family: {sid}")
        groups = sorted(
            (p["group"] for p in variants),
            key=lambda g: _digest(f"253|alias|{annotator}|{sid}|{g}"),
        )
        aliases = {
            group: f"[GROUP_{chr(65 + index)}]" for index, group in enumerate(groups)
        }
        for model in ("M1", "M2"):
            for condition in ("B0", "F1"):
                set_id = _digest(f"253|set|{sid}|{model}|{condition}")[:24]
                card_ids = []
                for prompt in variants:
                    output = outputs_by_key[(model, condition, prompt["prompt_id"])]
                    record_id = _digest(f"253|response|{set_id}|{prompt['prompt_id']}")[
                        :24
                    ]
                    card_ids.append(record_id)
                    response_cards.append(
                        {
                            "record_id": record_id,
                            "set_id": set_id,
                            "prompt": _mask(prompt["prompt_text"], prompt, aliases),
                            "response": _mask(
                                output["raw_output"] or "", prompt, aliases
                            ),
                            "output_status": output["status"],
                        }
                    )
                    key.append(
                        {
                            "record_id": record_id,
                            "set_id": set_id,
                            "scenario_id": sid,
                            "attribute": attribute,
                            "prompt_id": prompt["prompt_id"],
                            "group": prompt["group"],
                            "model": model,
                            "condition": condition,
                            "alias": aliases[prompt["group"]],
                        }
                    )
                set_cards.append(
                    {
                        "set_id": set_id,
                        "record_ids": sorted(
                            card_ids,
                            key=lambda rid: _digest(f"253|set-order|{annotator}|{rid}"),
                        ),
                    }
                )
    response_cards.sort(
        key=lambda card: _digest(f"253|response-order|{annotator}|{card['record_id']}")
    )
    set_cards.sort(
        key=lambda card: _digest(f"253|comparison-order|{annotator}|{card['set_id']}")
    )
    expected = 576 if split == "test" else 240
    if len(response_cards) != expected or len(set_cards) != 4 * len(selected):
        raise AnnotationError("Annotation packet count mismatch")
    packet = {
        "schema_version": 1,
        "split": split,
        "annotator": annotator,
        "allocation_sha256": hashlib.sha256(allocation.read_bytes()).hexdigest(),
        "response_cards": response_cards,
        "set_cards": set_cards,
    }
    packet_path = directory / "packet.json"
    _write_once(packet_path, _bytes(packet))
    _write_once(
        directory / "sealed_key.json",
        _bytes({"mappings": key, "source_hashes": source_hashes}),
    )
    return packet_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args(argv)
    print(build_packet(args.root, args.split, args.annotator))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
