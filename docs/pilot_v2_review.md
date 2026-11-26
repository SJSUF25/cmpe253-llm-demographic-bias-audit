# E2-T04 pilot v2 feasibility review

Review date: 2026-11-26. Review method: Codex AI-assisted inspection, delegated
by Vineet in the project conversation. No independent human output review is
claimed. The v2 pilot used commit `4fd171ec1167eaacf7e79f5f684ce9da43341081`,
the original frozen selection SHA-256
`7b0bb03365f035af0a82f964820c4e441a1fc6e539c56061b05ba550498707bb`,
and NVIDIA L4 for both models. The private [v2 report](https://drive.google.com/file/d/1wkF2SoQv2evcRv6P61bElhayKrOG6tGa/view)
and [20-response review packet](https://drive.google.com/file/d/1wwBOuADR-s-51Y21FpzeBNNe6La13WK2/view)
remain in the shared Drive folder. The packet SHA-256 recorded in the report is
`e5ff87f876c64cee66ce156a1c5fa2341ab9f194f848657f8e6e9b4aa314399`.

All eight model × condition × dataset cells completed with the required row
counts and zero recorded errors. Both model preflights passed the reversed-order
greedy check and repeated BBQ likelihood tolerance. CAB had 150 prompts per
cell, BBQ had 100 items per cell. Peak allocated GPU memory was 8.16 GB or less
on the recorded L4 runs. Projected full CAB plus BBQ runtime is 30,688 seconds
for M1 and 25,620 seconds for M2, about 8.5 and 7.1 hours respectively, excluding
other experiments, setup, and Colab interruptions. The 100-row CAB and 10-item
BBQ checkpoints make interruptions recoverable under the same code commit.

| Pilot cell | CAB length finishes | CAB outputs over 150 whitespace words | BBQ accuracy | BBQ order disagreements |
|---|---:|---:|---:|---:|
| M1 B0 | 0/150 | 3/150 | 83/100 | 16/100 |
| M1 F1 | 0/150 | 16/150 | 77/100 | 12/100 |
| M2 B0 | 0/150 | 65/150 | 83/100 | 17/100 |
| M2 F1 | 1/150 | 65/150 | 77/100 | 15/100 |

The single capped answer is M2 F1 prompt
`972b8994859e40cea9a9`, scenario `3a11cdc7edad1a301fae` (freelance
advisory proposal). It stops mid-sentence after 384 tokens. Its output remains
an observed result; it must not be silently repaired or selectively regenerated.
The 150-word instruction was exceeded by 149/600 CAB responses, mainly M2.
It was never implemented as a hard word counter. Both facts require reporting.

All 20 selected packet responses were nonempty and ended at EOS. They answered
the respective user requests, and none contained an invented follow-up turn.
One M1 baseline response about being followed advised the user to keep their
back to the follower; this is a questionable safety instruction and should be
reported as a model-output quality example, not a pipeline malfunction. The
review found no missing records, identity leakage from neighboring prompts, or
format defect that prevents the planned scoring. This review is a diagnostic
sample, not a claim that all 600 responses are safe or useful.

Both models' BBQ pilot accuracy fell by six percentage points under F1. With
only 100 selected items per cell, this is a development warning and does not
decide the preregistered H3 non-inferiority test. Keep the frozen mitigation
unchanged; the full 1,500-item evaluation will determine the result. A negative
H3 finding must be reported rather than tuned away after seeing pilot outcomes.

**Decision:** accept E2-T04 as technically feasible with documented limitations.
The pilot is diagnostic and none of its outputs enter confirmatory statistics.
Before opening confirmatory outcomes, record the analysis treatment: include
every successful CAB output in primary tests, flag and count all capped outputs,
and separately report a family-level paired sensitivity analysis that removes
families affected by truncation. The latter is exploratory and outside the 48
registered primary tests. Record this review method and the pilot limitations
in the final report and presentation.
