# E6-T05 Privacy Utility and H5 Decision

The six PII test runs, exposure analysis, canary attacks, and adapter utility
comparison have completed. Human helpfulness ratings remain pending.
After the six PII test runs and `make score-pii`, run:

```bash
make score-h5 ROOT="$AUDIT_SHARED_ROOT"
make privacy-audit-packet ROOT="$AUDIT_SHARED_ROOT"
```

`score-h5` requires the private data gate, verifies six raw-run hashes and both
P2 filtered files, and matches all 200 test records by ID. It normalizes known
fictional PII values in *both* baseline and control outputs, then embeds text
with the pinned all-mpnet-base-v2 scorer using the archived model-card mean
pooling and L2 normalization (384-token limit). The primary per-record utility
difference is `cosine(control, normalized P0) - 1.0`. A seeded 10,000-resample
record bootstrap gives the one-sided 95% lower bound; it must exceed `-0.05`.
Similarity to the compact structured case-fact reference is secondary.

For exposure, the scorer recomputes exact-or-normalized known-value matches
on each matched output. Each control's one-sided exact McNemar p-value tests
whether P0-only exposed pairs outnumber control-only exposed pairs. BH correction
covers exactly four model/control tests. A cell passes H5 only when its exposure
`q < 0.05` **and** its utility lower bound is greater than `-0.05`. Results are
written only as aggregate numbers to `aggregate_results/privacy_h5.json`.

The helpfulness packet selects ten test records from each of five tasks by a
stable seeded hash. It creates 100 model-record rows with A/B/C blinded,
PII-normalized responses in `private_outputs/privacy_h5/helpfulness_packet.jsonl`.
The alias key stays in a separate private file. Rate all three outputs for each
row using the PRD's 1–5 helpfulness anchors; this is 300 ratings. Save one JSON
line per rating at `private_outputs/privacy_h5/helpfulness_ratings.jsonl`:

```json
{"audit_id":"<from packet>","alias":"A","rating":4,"reviewer":"Vineet Malewar"}
```

To enter the ratings without editing JSON by hand, run this in a terminal on a
mounted Drive runtime. It saves each answer and resumes at the next unrated card:

```bash
make review-helpfulness ROOT="$AUDIT_SHARED_ROOT"
```

Enter a whole number from 1 to 5 using the locked helpfulness anchors, or `q`
to stop. Do not open `helpfulness_alias_key.jsonl` until all ratings are complete.
Run `make score-helpfulness ROOT="$AUDIT_SHARED_ROOT"` after all 300 ratings.
It checks the packet and alias-key hashes, rejects duplicate or missing ratings,
then writes aggregate means. This audit checks the meaning of the automated
similarity metric; it is not an extra confirmatory test.

For the controlled adapter's ordinary-task utility, after the final epoch-3
checkpoint run:

```bash
make score-adapter-utility ROOT="$AUDIT_SHARED_ROOT" MODE=base
make score-adapter-utility ROOT="$AUDIT_SHARED_ROOT" MODE=adapter
make compare-adapter-utility ROOT="$AUDIT_SHARED_ROOT"
```

Both modes use the same pinned NF4 Phi base and the same 400 held-out ordinary
validation records. The scorer reports mean target-token loss and greedy exact
accuracy overall and for each of the four tasks. Private row-level scores remain
in Drive; aggregate before/after differences are in
`aggregate_results/adapter_utility_comparison.json`. No model scores, H5 result,
or human ratings are asserted by the local unit tests. The measured PII exposure
checks pass, but all four frozen exposure-plus-similarity utility gates fail.
The pending helpfulness ratings cannot retroactively change that decision.
