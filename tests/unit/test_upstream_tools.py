"""Unit tests for immutable upstream URL and manifest construction."""

from upstream_tools import render_manifest, safe_slug, urls_for


def test_hugging_face_urls_are_revision_pinned() -> None:
    resource = {
        "provider": "huggingface_model",
        "repo_id": "org/model",
        "revision": "a" * 40,
        "artifacts": ["README.md"],
    }

    metadata_url, artifacts = urls_for(resource)
    assert resource["revision"] in metadata_url
    assert resource["revision"] in artifacts["README.md"]
    assert "/main/" not in artifacts["README.md"]


def test_manifest_contains_hashes_and_no_floating_main() -> None:
    resource = {
        "key": "M1",
        "kind": "model",
        "provider": "huggingface_model",
        "repo_id": "org/model",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "source_url": "https://example.test/org/model",
        "metadata_url": f"https://example.test/revision/{'a' * 40}",
        "metadata_sha256": "b" * 64,
        "resolved_artifacts": [
            {
                "path": "docs/upstream/org--model/README.md",
                "url": f"https://example.test/resolve/{'a' * 40}/README.md",
                "size_bytes": 10,
                "sha256": "c" * 64,
            }
        ],
    }

    manifest = render_manifest([resource], "2026-09-30T00:00:00Z", "models")
    assert "revision:" in manifest
    assert "sha256:" in manifest
    assert "/main/" not in manifest


def test_repository_ids_have_safe_local_slugs() -> None:
    assert safe_slug("Qwen/Qwen3-4B") == "Qwen--Qwen3-4B"
