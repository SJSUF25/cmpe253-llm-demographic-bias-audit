# Data Handling and Shared-Storage Procedure

## Scope

The project uses public benchmark data, fully synthetic PII, synthetic canaries, and
locally generated model outputs. It does not use real personal records, private user
conversations, or live third-party system testing.

## Storage boundary

The shared Google Drive project folder is the private collaboration boundary. Its
machine-specific mount path is supplied through `AUDIT_SHARED_ROOT` and must never be
committed. `configs/storage.example.yaml` defines the required child directories:

- `raw_immutable`: pinned upstream data, never modified in place;
- `private_outputs`: raw generations and annotation exports;
- `adapters_private`: checkpoints and the controlled canary adapter;
- `aggregate_results`: reviewed aggregate tables, plots, and redacted examples;
- `manifests`: checksums and storage/run metadata;
- `backups`: private recovery copies.

Only Vineet and Abhinandan receive editor access. Link sharing must be restricted;
public or organization-wide sharing is prohibited. Each member enables account
multi-factor authentication. Course staff receive artifacts only through the approved
submission channel, not by broadening Drive permissions.

## Classification and Git policy

| Class | Examples | Shared Drive | Git |
|---|---|---:|---:|
| Public source metadata | licenses, URLs, revisions, card snapshots | Yes | Yes |
| Immutable raw benchmark data | CAB and BBQ files | Yes | No |
| Synthetic sensitive | full synthetic PII, canary manifests | Yes, restricted | No |
| Private experiment output | raw harmful generations, annotations, adapters | Yes, restricted | No |
| Reviewed aggregate | metric tables, CIs, redacted examples | Yes | Yes after review |
| Secret | Hugging Face token, credentials | Colab secret manager only | Never |

The repository `.gitignore` is a safety layer, not authorization to copy private data
into the working tree. Before every commit, inspect `git status` and staged content.

## Immutability and checksums

1. Download into a temporary location.
2. Calculate SHA-256 before moving into `raw_immutable`.
3. Write source URL, immutable revision, retrieval UTC timestamp, byte count, and hash
   to the corresponding manifest.
4. Never edit a raw file. A changed upstream source becomes a new revisioned object.
5. Generated private partitions are append-only during a run. A completed partition
   receives a checksum before backup.
6. Git stores a reviewed checksum manifest, never the private payload.

## Access probe and backup verification

After Vineet creates the Drive folder and grants Abhinandan editor access:

```bash
make storage-init ROOT="/mounted/shared/path" AUTHOR="Vineet Malewar"
make storage-verify ROOT="/mounted/shared/path" REVIEWER="Abhinandan"
```

The first command creates the required directories and a harmless random probe. The
second independently reads and hashes that probe and writes a reviewer receipt. It
does not expose credentials or project data. Both receipt hashes are copied into the
Epic 0 handoff record.

For recovery testing, copy a harmless aggregate fixture into `backups`, verify its
checksum after a fresh download/mount, then remove only that named fixture. Do not use
recursive deletion commands against the Drive root.

## Retention and incident response

- Keep raw/private artifacts only through course grading and any formal regrade window,
  unless the instructor requires a different period.
- After that window, both members confirm before deleting private generations,
  synthetic PII, canary mappings, checkpoints, and adapters.
- Retain code, public-source metadata, aggregate results, and reports if permitted.
- On accidental sharing or token exposure: revoke the link/token immediately, record
  time and affected artifacts, notify the teammate, rotate credentials, remove the
  exposed copy, and document corrective action.

