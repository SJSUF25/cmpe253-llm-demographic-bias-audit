"""Verify frozen data artifacts and prepare a private manual-QA packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from data.bbq_sampler import build_bbq
from data.cab_builder import build_cab
from data.download import DEFAULT_CONFIG as RAW_CONFIG
from data.download import download_datasets
from privacy.membership_data import write_corpus as write_membership_corpus
from privacy.synthetic_data import write_corpus as write_pii_corpus

ROOT = Path(__file__).resolve().parents[2]
CAB_CONFIG = ROOT / "configs/cab_build.json"
BBQ_CONFIG = ROOT / "configs/bbq_build.json"
PII_CONFIG = ROOT / "configs/privacy_data.json"
MEMBERSHIP_CONFIG = ROOT / "configs/membership_data.json"
QA_PATH = Path("private_outputs/data_qa/review_packet.json")
MANIFEST_PATH = Path("manifests/data_validation_manifest.json")


class DataGateError(ValueError):
    """The data gate cannot be validated without complete, matching artifacts."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _required_paths(root: Path) -> list[Path]:
    raw = json.loads(RAW_CONFIG.read_text(encoding="utf-8"))
    cab = json.loads(CAB_CONFIG.read_text(encoding="utf-8"))
    bbq = json.loads(BBQ_CONFIG.read_text(encoding="utf-8"))
    pii = json.loads(PII_CONFIG.read_text(encoding="utf-8"))
    membership = json.loads(MEMBERSHIP_CONFIG.read_text(encoding="utf-8"))
    paths = [root / "manifests" / raw["manifest_name"]]
    for dataset in raw["datasets"]:
        for source in [
            *dataset["files"],
            *([dataset["supplemental"]] if dataset.get("supplemental") else []),
        ]:
            paths.append(
                root
                / "raw_immutable"
                / dataset["key"]
                / dataset["revision"]
                / source["path"]
            )
    paths.extend(root / cab[key] for key in ("output_path", "manifest_path"))
    paths.extend(
        root / bbq[key] for key in ("items_path", "permutations_path", "manifest_path")
    )
    paths.extend(
        root / pii["output_directory"] / f"{split}.jsonl"
        for split in ("development", "test")
    )
    paths.append(root / pii["manifest_path"])
    paths.extend(
        root / membership["output_directory"] / f"{name}.jsonl"
        for name in (
            "ordinary_train",
            "ordinary_validation",
            "member_canaries",
            "nonmember_calibration",
            "nonmember_evaluation",
            "training_presentations",
        )
    )
    paths.append(root / membership["manifest_path"])
    return paths


def _pick(rows: list[dict], count: int, namespace: str) -> list[dict]:
    if len(rows) < count:
        raise DataGateError(f"Only {len(rows)} eligible {namespace} rows; need {count}")

    def sort_key(row: dict) -> tuple[str, str]:
        identifier = row.get("record_id", row.get("item_id"))
        return _sha(f"253|qa|{namespace}|{identifier}".encode()), identifier

    return sorted(
        rows,
        key=sort_key,
    )[:count]


def make_review_packet(root: Path) -> dict:
    """Choose fixed development-only QA examples without opening sealed test text."""
    cab_config = json.loads(CAB_CONFIG.read_text(encoding="utf-8"))
    bbq_config = json.loads(BBQ_CONFIG.read_text(encoding="utf-8"))
    pii_config = json.loads(PII_CONFIG.read_text(encoding="utf-8"))
    membership_config = json.loads(MEMBERSHIP_CONFIG.read_text(encoding="utf-8"))
    cab = _jsonl(root / cab_config["output_path"])
    families: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cab:
        if row["split"] == "development":
            families[(row["attribute"], row["scenario_id"])].append(row)
    selected_families = []
    for attribute in ("gender", "race"):
        eligible = [
            (scenario, rows)
            for (group, scenario), rows in families.items()
            if group == attribute
        ]
        if len(eligible) < 25:
            raise DataGateError(
                f"Only {len(eligible)} development {attribute} families"
            )
        chosen = sorted(
            eligible, key=lambda item: (_sha(f"253|qa|cab|{item[0]}".encode()), item[0])
        )[:25]
        selected_families.extend(
            {"attribute": attribute, "scenario_id": scenario, "prompts": rows}
            for scenario, rows in chosen
        )
    pii = _jsonl(root / pii_config["output_directory"] / "development.jsonl")
    pii_sample = [
        row
        for task in pii_config["tasks"]
        for row in _pick([r for r in pii if r["task"] == task], 2, f"pii|{task}")
    ]
    membership_dir = root / membership_config["output_directory"]
    ordinary = _jsonl(membership_dir / "ordinary_train.jsonl")
    ordinary_sample = [
        row
        for task in membership_config["ordinary_tasks"]
        for row in _pick(
            [r for r in ordinary if r["task"] == task], 2, f"ordinary|{task}"
        )
    ]
    members = _jsonl(membership_dir / "member_canaries.jsonl")
    member_sample = [
        row
        for repetition in (1, 5, 10)
        for row in _pick(
            [r for r in members if r["repetition"] == repetition],
            1,
            f"member|{repetition}",
        )
    ]
    calibration_sample = _pick(
        _jsonl(membership_dir / "nonmember_calibration.jsonl"), 2, "calibration"
    )
    bbq = _jsonl(root / bbq_config["items_path"])
    bbq_sample = []
    for category in bbq_config["categories"]:
        for context in ("ambig", "disambig"):
            for polarity in ("neg", "nonneg"):
                eligible = [
                    row
                    for row in bbq
                    if row["category"] == category
                    and row["context_condition"] == context
                    and row["question_polarity"] == polarity
                ]
                bbq_sample.extend(
                    _pick(eligible, 1, f"bbq|{category}|{context}|{polarity}")
                )
    return {
        "schema_version": 1,
        "review_status": "pending_human_review",
        "selection_rule": "SHA256(253|qa|namespace|record_id_or_item_id)",
        "cab_development_families": selected_families,
        "pii_development": pii_sample,
        "ordinary_train": ordinary_sample,
        "member_canaries": member_sample,
        "nonmember_calibration": calibration_sample,
        "bbq": bbq_sample,
        "sealed_test_contents_included": False,
    }


def validate_data(root: Path) -> Path:
    """Require and reverify every artifact; write no raw or derived source data."""
    root = root.expanduser().resolve()
    missing = [path for path in _required_paths(root) if not path.is_file()]
    if missing:
        first = missing[0].relative_to(root)
        raise DataGateError(f"Missing {len(missing)} required artifact(s): {first}")

    def no_network(_url: str) -> bytes:
        raise DataGateError("Validation attempted a network download")

    source_manifest = download_datasets(root, fetcher=no_network)
    cab_manifest = build_cab(root)
    bbq_manifest = build_bbq(root)
    pii_manifest = write_pii_corpus(root)
    membership_manifest = write_membership_corpus(root)
    packet = make_review_packet(root)
    if len(packet["cab_development_families"]) != 50:
        raise DataGateError("Manual QA requires exactly 50 CAB families")
    counts = {
        key: len(value) for key, value in packet.items() if isinstance(value, list)
    }
    packet_bytes = (
        json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    qa_path = root / QA_PATH
    manifest = {
        "schema_version": 1,
        "structural_checks": "passed",
        "overall_status": "pending_token_length_and_human_review",
        "human_review_status": "pending",
        "model_token_length_status": "pending_pinned_tokenizer_check",
        "source_manifests": {
            str(path.relative_to(root)): _sha(path.read_bytes())
            for path in (
                source_manifest,
                cab_manifest,
                bbq_manifest,
                pii_manifest,
                membership_manifest,
            )
        },
        "review_packet_path": str(QA_PATH),
        "review_packet_sha256": _sha(packet_bytes),
        "review_counts": counts,
    }
    manifest_path = root / MANIFEST_PATH
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    for path, content in ((qa_path, packet_bytes), (manifest_path, manifest_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise DataGateError(f"Existing QA artifact differs: {path}")
    for path, content in ((qa_path, packet_bytes), (manifest_path, manifest_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(validate_data(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
