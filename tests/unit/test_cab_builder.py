import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.cab_builder import CabBuildError, build_cab, build_prompt_rows

REVISION = "a" * 40
EXPECTED_IDENTIFIERS = [
    ("00682d98f474502accdc", "b787751ed7f6d60d8db6", "test"),
    ("0476f8c9d75a19a8df89", "08b57b12a18c52408041", "test"),
    ("0919912659cf443af90e", "20d824396c00a77934d9", "development"),
    ("16552eda0cc7754f3667", "b491eadfd0c0524116ad", "test"),
    ("1766fee6551840ad8e7d", "be4ff152a6b30d7fb6a4", "test"),
    ("9a48f6c9fc65bfbb2dcf", "3bab01ec20d132f5381d", "test"),
    ("9c81ea61011c64933f78", "2b41ea6b5e634711f9db", "test"),
    ("a12522816ac15b1bf636", "eef57e9f9241e30dff48", "test"),
    ("ab9493e5bf33722e94d1", "474aefc61192707954e4", "development"),
    ("ca2f4261c4bff791ff79", "b437f43c19a12153e1fd", "test"),
]


def _config() -> dict:
    return {
        "dataset": "CAB",
        "source_revision": REVISION,
        "license": "MIT",
        "attributes": {
            "gender": ["male", "female"],
            "race": ["white", "black", "asian", "hispanic"],
        },
        "expected": {
            "explicit_families": 10,
            "implicit_families": 10,
            "explicit_prompts": 30,
            "implicit_prompts": 30,
            "total_prompts": 60,
        },
        "split": {
            "namespace": "cmpe253:",
            "modulus": 10,
            "development_buckets": [0, 1],
        },
    }


def _rows() -> list[dict]:
    rows = []
    for index in range(10):
        attribute = "gender" if index < 5 else "race"
        explicit = (
            "{{male/female}}"
            if attribute == "gender"
            else "{{white/black/asian/hispanic}}"
        )
        implicit = (
            "{{Alex/Alice}}"
            if attribute == "gender"
            else "{{W Name/B Name/A Name/H Name}}"
        )
        common = {
            "attribute": attribute,
            "superdomain_explicit": f"Super {index}",
            "domain_explicit": f"Domain {index}",
            "topic": f"Topic {index}",
        }
        rows.extend(
            [
                {
                    **common,
                    "expl_impl": "explicit",
                    "example": f"Family {index}: I am {explicit}.",
                },
                {
                    **common,
                    "expl_impl": "implicit",
                    "example": f"Family {index}: My name is {implicit}.",
                },
            ]
        )
    return rows


def test_ten_family_identifier_fixture_is_stable() -> None:
    records = build_prompt_rows(_rows(), _config())
    actual = []
    for scenario_id in sorted({row["scenario_id"] for row in records}):
        first = next(
            row
            for row in records
            if row["scenario_id"] == scenario_id
            and row["explicit_implicit"] == "explicit"
            and row["group"] in {"male", "white"}
        )
        actual.append((scenario_id, first["prompt_id"], first["split"]))

    assert actual == EXPECTED_IDENTIFIERS
    assert len(records) == 60
    assert all("{{" not in row["prompt_text"] for row in records)


def test_variants_share_family_split_and_pair_only_within_signal_type() -> None:
    records = build_prompt_rows(_rows(), _config())
    by_scenario = {}
    for row in records:
        by_scenario.setdefault(row["scenario_id"], []).append(row)

    for family in by_scenario.values():
        assert len({row["split"] for row in family}) == 1
        for row in family:
            expected_pair_count = 1 if row["attribute"] == "gender" else 3
            assert len(row["paired_variant_ids"]) == expected_pair_count
            allowed = {
                candidate["prompt_id"]
                for candidate in family
                if candidate["explicit_implicit"] == row["explicit_implicit"]
                and candidate["prompt_id"] != row["prompt_id"]
            }
            assert set(row["paired_variant_ids"]) == allowed


def test_linkage_and_marker_failures_are_hard_errors() -> None:
    missing_implicit = _rows()[:-1]
    with pytest.raises(CabBuildError, match="without one explicit and implicit"):
        build_prompt_rows(missing_implicit, _config())

    malformed = copy.deepcopy(_rows())
    malformed[0]["example"] = "Wrong {{male/female/unknown}} marker"
    with pytest.raises(CabBuildError, match="arity 2"):
        build_prompt_rows(malformed, _config())


def test_build_is_idempotent_and_rejects_changed_input(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    raw_path = shared / "raw_immutable/D1" / REVISION / "cab.json"
    raw_path.parent.mkdir(parents=True)
    raw_bytes = json.dumps(_rows()).encode()
    raw_path.write_bytes(raw_bytes)
    config = {
        **_config(),
        "schema_version": 1,
        "config_id": "cab-test-v1",
        "input_path": f"raw_immutable/D1/{REVISION}/cab.json",
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "output_path": "private_outputs/processed_data/cab/prompts.jsonl",
        "manifest_path": "manifests/cab_build.json",
    }
    config_path = tmp_path / "cab_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manifest = build_cab(shared, config_path)
    first_manifest = manifest.read_bytes()
    first_output = (shared / config["output_path"]).read_bytes()
    build_cab(shared, config_path)
    assert manifest.read_bytes() == first_manifest
    assert (shared / config["output_path"]).read_bytes() == first_output

    raw_path.write_bytes(raw_bytes + b" ")
    with pytest.raises(CabBuildError, match="input SHA-256"):
        build_cab(shared, config_path)
