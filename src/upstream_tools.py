"""Resolve pinned upstream metadata and archive small governance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "upstream_registry.json"
DEFAULT_OUTPUT = ROOT / "docs" / "upstream"
MODEL_MANIFEST = ROOT / "configs" / "model_manifest.yaml"
DATA_MANIFEST = ROOT / "configs" / "data_manifest.yaml"


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for content."""
    return hashlib.sha256(content).hexdigest()


def safe_slug(value: str) -> str:
    """Convert a repository ID into a stable local directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", value)


def urls_for(resource: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Return immutable metadata and artifact URLs for one resource."""
    repo_id = resource["repo_id"]
    revision = resource["revision"]
    quoted_repo = "/".join(urllib.parse.quote(part) for part in repo_id.split("/"))
    if resource["provider"] == "huggingface_model":
        metadata_url = (
            f"https://huggingface.co/api/models/{quoted_repo}/revision/{revision}"
        )
        base = f"https://huggingface.co/{quoted_repo}/resolve/{revision}"
    elif resource["provider"] == "huggingface_dataset":
        metadata_url = (
            f"https://huggingface.co/api/datasets/{quoted_repo}/revision/{revision}"
        )
        base = f"https://huggingface.co/datasets/{quoted_repo}/resolve/{revision}"
    elif resource["provider"] == "github":
        metadata_url = f"https://api.github.com/repos/{quoted_repo}/commits/{revision}"
        base = f"https://raw.githubusercontent.com/{quoted_repo}/{revision}"
    else:
        raise ValueError(f"Unsupported provider: {resource['provider']}")
    return metadata_url, {name: f"{base}/{name}" for name in resource["artifacts"]}


def fetch(url: str) -> bytes:
    """Fetch one public upstream object with a descriptive user agent."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "cmpe253-llm-bias-audit/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def metadata_revision(provider: str, metadata: dict[str, Any]) -> str:
    """Extract the resolved revision from provider metadata."""
    return metadata["sha"] if provider != "github" else metadata["sha"]


def normalized_metadata(provider: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove volatile popularity and usage fields from archived API metadata."""
    if provider == "github":
        commit = metadata["commit"]
        return {
            "sha": metadata["sha"],
            "html_url": metadata["html_url"],
            "author": commit["author"],
            "committer": commit["committer"],
            "message": commit["message"],
            "tree_sha": commit["tree"]["sha"],
        }
    return {
        "id": metadata["id"],
        "sha": metadata["sha"],
        "lastModified": metadata.get("lastModified"),
        "private": metadata.get("private"),
        "gated": metadata.get("gated"),
        "disabled": metadata.get("disabled"),
        "license": metadata.get("cardData", {}).get("license"),
        "siblings": [
            {
                key: sibling[key]
                for key in ("rfilename", "size", "blobId", "lfs")
                if key in sibling
            }
            for sibling in metadata.get("siblings", [])
        ],
    }


def yaml_scalar(value: object) -> str:
    """Serialize a simple scalar in a YAML-compatible JSON representation."""
    return json.dumps(value, ensure_ascii=False)


def render_manifest(
    resources: list[dict[str, Any]], generated_at: str, title: str
) -> str:
    """Render a deterministic, human-readable manifest."""
    lines = ["schema_version: 1", f"generated_at_utc: {yaml_scalar(generated_at)}"]
    lines.append(f"manifest_type: {yaml_scalar(title)}")
    lines.append("resources:")
    for resource in resources:
        lines.extend(
            [
                f"  - key: {yaml_scalar(resource['key'])}",
                f"    kind: {yaml_scalar(resource['kind'])}",
                f"    provider: {yaml_scalar(resource['provider'])}",
                f"    repo_id: {yaml_scalar(resource['repo_id'])}",
                f"    revision: {yaml_scalar(resource['revision'])}",
                f"    tokenizer_revision: {yaml_scalar(resource['revision'])}"
                if resource["kind"] in {"model", "scorer"}
                else "    tokenizer_revision: null",
                f"    license: {yaml_scalar(resource['license'])}",
                f"    source_url: {yaml_scalar(resource['source_url'])}",
                f"    metadata_url: {yaml_scalar(resource['metadata_url'])}",
                f"    metadata_sha256: {yaml_scalar(resource['metadata_sha256'])}",
                "    artifacts:",
            ]
        )
        for artifact in resource["resolved_artifacts"]:
            lines.extend(
                [
                    f"      - path: {yaml_scalar(artifact['path'])}",
                    f"        url: {yaml_scalar(artifact['url'])}",
                    f"        size_bytes: {artifact['size_bytes']}",
                    f"        sha256: {yaml_scalar(artifact['sha256'])}",
                ]
            )
    return "\n".join(lines) + "\n"


def synchronize(registry_path: Path, output_dir: Path) -> None:
    """Verify revisions, archive small artifacts, and write model/data manifests."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    generated_at = datetime.now(UTC).isoformat()
    resolved: list[dict[str, Any]] = []
    for source in registry["resources"]:
        resource = dict(source)
        metadata_url, artifact_urls = urls_for(resource)
        metadata_content = fetch(metadata_url)
        metadata = json.loads(metadata_content)
        actual_revision = metadata_revision(resource["provider"], metadata)
        if actual_revision != resource["revision"]:
            raise ValueError(
                f"Revision mismatch for {resource['repo_id']}: "
                f"expected {resource['revision']}, received {actual_revision}"
            )
        stable_metadata = normalized_metadata(resource["provider"], metadata)
        if resource["provider"].startswith("huggingface"):
            declared_license = stable_metadata["license"]
            if not declared_license:
                raise ValueError(f"No declared license for {resource['repo_id']}")
            if declared_license.casefold() != resource["license"].casefold():
                raise ValueError(
                    f"License mismatch for {resource['repo_id']}: expected "
                    f"{resource['license']}, received {declared_license}"
                )
        stable_metadata_content = (
            json.dumps(stable_metadata, indent=2, sort_keys=True) + "\n"
        ).encode()

        resource_dir = output_dir / safe_slug(resource["repo_id"])
        resource_dir.mkdir(parents=True, exist_ok=True)
        (resource_dir / "metadata.json").write_bytes(stable_metadata_content)
        artifacts = []
        for name, url in artifact_urls.items():
            content = fetch(url)
            path = resource_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "url": url,
                    "size_bytes": len(content),
                    "sha256": sha256_bytes(content),
                }
            )
        resource.update(
            {
                "source_url": (
                    f"https://huggingface.co/{resource['repo_id']}"
                    if resource["kind"] != "dataset"
                    and resource["provider"].startswith("huggingface")
                    else f"https://huggingface.co/datasets/{resource['repo_id']}"
                    if resource["provider"] == "huggingface_dataset"
                    else f"https://github.com/{resource['repo_id']}"
                ),
                "metadata_url": metadata_url,
                "metadata_sha256": sha256_bytes(stable_metadata_content),
                "resolved_artifacts": artifacts,
            }
        )
        resolved.append(resource)

    models = [entry for entry in resolved if entry["kind"] in {"model", "scorer"}]
    datasets = [entry for entry in resolved if entry["kind"] == "dataset"]
    MODEL_MANIFEST.write_text(
        render_manifest(models, generated_at, "models_and_scorers"), encoding="utf-8"
    )
    DATA_MANIFEST.write_text(
        render_manifest(datasets, generated_at, "datasets"), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the upstream synchronization parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sync_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run upstream synchronization."""
    args = build_parser().parse_args(argv)
    synchronize(args.registry, args.output)
    print(MODEL_MANIFEST)
    print(DATA_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
