# First L4 pilot review (2026-11-25)

The first E2-T04 pilot remains diagnostic. Its private report is
`manifests/pilot/pilot_report.json` and its 20-response packet is
`private_outputs/pilot/pilot_output_review.jsonl` under the shared Drive root.
The report records eight complete cells, zero execution errors, passing model
preflights, and one GPU class per model. It explicitly leaves the gate decision
at `pending_operator_output_review`.

| Model | CAB baseline length finishes | CAB mitigation length finishes | BBQ accuracy baseline → mitigation | Projected full CAB + BBQ GPU time |
| --- | ---: | ---: | ---: | ---: |
| M1 | 128/150 | 130/150 | 79% → 76% | 12.0 h |
| M2 | 148/150 | 150/150 | 82% → 75% | 10.1 h |

The 100-item BBQ sample is diagnostic; its accuracy differences cannot establish
the confirmatory fairness or utility result. The GPU time excludes other required
evaluations and Colab setup, downloads, and interruptions.

All 20 sampled CAB responses have `finish_reason=length`. Several M2 samples
contain invented follow-up user messages after an otherwise plausible answer.
The pinned Phi-4 mini generation configuration lists token `200020` (`<|end|>`)
and token `199999` (`<|endoftext|>`) as stopping tokens. The old adapter
overrode that list with only `tokenizer.eos_token_id`, which is `199999`.
Consequently it could continue past the chat-turn ending token. The adapter fix
uses the model's generation stop-token list and classifies either token as EOS.

The pilot gate must remain pending. Before a full run, use development-only
diagnostics to establish a response-length policy that reduces truncation while
preserving task utility, freeze the amended protocol and governance lock, renew
the data-gate receipt, and run a new versioned pilot without overwriting these
first-pilot artifacts. Review its sampled outputs and measured compute before
making the gate decision. The first-pilot outputs must not enter confirmatory
statistics.

After the adapter fix and diagnostic script are committed and pushed, run the
following in a new cell of the existing L4 Colab notebook. If the runtime was
restarted, rerun only the Drive mount and repository setup cells first. These
commands write separate development-only diagnostic artifacts; they do not rerun
or modify the first pilot.

```bash
%%bash
set -euo pipefail
cd /content/cmpe253-llm-demographic-bias-audit
git pull --ff-only
ROOT=/content/drive/MyDrive/CMPE253-Project
uv run --frozen --group colab python -m inference.length_diagnostic --root "$ROOT" --model M1
uv run --frozen --group colab python -m inference.length_diagnostic --root "$ROOT" --model M2
```

Compare the two `length_finishes` counts with the first-pilot counts above.
Review all 40 private diagnostic outputs for completeness, identity-based
assumptions, and invented dialogue before choosing a locked revised protocol.
