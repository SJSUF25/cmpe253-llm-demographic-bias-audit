# System Boundary and Permitted Claims

Status: frozen candidate v1.1; final operator review requires passing E0-T04 Colab
smoke manifests persisted in Drive and a current governance lock.

## Evaluated system

The project evaluates single-turn, English-language behavior of two pinned,
open-weight conversational models: M1 `Qwen/Qwen3-4B-Instruct-2507` and M2
`microsoft/Phi-4-mini-instruct`. Models run locally in Colab Pro, one at a time, at
FP16 with their official pinned chat templates. CAB is the primary realistic
counterfactual benchmark. BBQ is a secondary structured and intersectional benchmark.
The fixed fairness instruction is the only primary fairness mitigation.

The pipeline also includes synthetic PII controls P0/P1/P2, a separately quantized
Phi adapter trained on constructed canaries, explanation ablations, perturbation and
stochastic sensitivity checks, and a cached-results demonstration. Raw generations,
adapters and sealed privacy labels stay in shared Drive under the documented link-access
exception; credentials are never stored there.

## Authorized inputs and execution

- Only pinned CAB and BBQ public benchmark data, deterministic synthetic records, and
  team-authored safe perturbations may be used.
- No real PII, resumes, user conversations, clinical data, or scraped personal data
  may enter the pipeline.
- No prompt may be sent to a live commercial or third-party inference service.
- Hugging Face access is used only to download the pinned public artifacts. Model
  inference remains inside the controlled Colab runtime.
- Adversarial probes are offline, synthetic, non-operational, and limited to testing
  whether identity is improperly treated as evidence.

## Permitted claims

Results may describe measured counterfactual demographic differences, refusal gaps,
BBQ scores, mitigation effects, synthetic-PII exposure, controlled canary leakage,
explanation behavior, and robustness for the exact pinned models, prompts, samples,
hardware-comparison rules, and analysis plan. Confidence intervals and prespecified
tests quantify uncertainty. CAB-to-BBQ comparisons are external-validity evidence;
their effect sizes are never pooled or directly subtracted.

## Claims that are prohibited

The project does not establish that either model is generally fair, safe, private,
unbiased, production-ready, or representative of all people and languages. It does
not infer membership in unknown pretraining data, prove differential privacy, audit a
live deployed service, or generalize beyond the tested male/female and U.S.-centric
race markers. White as the statistical reference group is a preregistered modeling
choice, not a normative default. The work does not cover all genders, races, ages,
religions, disabilities, nationalities, or sexual orientations.

## Stop conditions

A run stops rather than silently adapting when a pinned artifact cannot be resolved,
input exceeds 4,096 tokens, validation fails, GPU class changes within a model's
comparison, deterministic checks fail, real or unexpected PII is found, or a required
manifest/hash is absent. Model substitution, primary-run quantization, and post hoc
changes to hypotheses, samples, outcomes, or thresholds are forbidden.

## Ethical release and retention

Public artifacts contain aggregate tables, reviewed redacted examples, code, and
non-sensitive manifests only. Potentially harmful raw generations and membership
labels are not published. Private outputs and adapters are retained only through
course grading and verification, then deleted by Vineet after recording a deletion
receipt; aggregate evidence and hash-only manifests may be retained. Any exception
requires a written change record before the action.

## Authority and precedence

The course requirements, PRD v2.5, this boundary, the machine-readable experiment
registry, and pinned manifests jointly define the study. On conflict, Vineet records
a resolution before continuing. Results may
not be inspected to decide a resolution that changes a confirmatory analysis.
