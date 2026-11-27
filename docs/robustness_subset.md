# Frozen CAB robustness subset

`make freeze-robustness-sample ROOT=/mounted/project` selects 60 held-out CAB
scenario families before robustness generation: 30 gender and 30 race. It uses
Hamilton largest-remainder allocation across the locked test superdomains,
lexicographic ties, then the PRD's SHA-256 family order. The output manifest
records allocations, source hashes, and all selected scenario IDs. Selection
reads source prompt metadata and the validated implicit exclusion manifest;
it does not read model outputs.

`make build-robustness-perturbations ROOT=/mounted/project` writes 180 original
explicit prompts plus 180 neutral whitespace and 180 fixed-adversarial prompts
to private Drive storage. It verifies the whitespace condition has the same
word sequence as its source and that the adversarial condition retains the
source text before the frozen suffix. The manifest checksums all 540 rows.

These files are **inputs, not measured robustness results**. The planned
formal/conversational paraphrases and identity-clause-position variants still
require source-specific semantic validation before generation. The adversarial
condition also requires response-level labels for harmful compliance and safe
refusal; no such labels are inferred from the prompt construction.

After the current full CAB/BBQ run finishes, `notebooks/e7_robustness_colab.ipynb`
can run eight L4 cells under the unchanged model/config contract. Each 180-row
cell uses the standard 20-prompt checkpoint engine, with exact prompt,
condition, model revision, seed, environment, and commit checks. A final
`make report-robustness ROOT=/mounted/project` pairs each variant with the
corresponding original CAB output and reports subgroup refusal-gap changes
with 95% scenario-family bootstrap intervals. The anchored refusal detector is
uncalibrated, so this is a descriptive proxy and is not a human harmful-compliance
assessment.
