# E8 live demo review — 2026-11-27

The user supplied an executed copy of `e8_live_demo_colab.ipynb`. The setup cell
reported commit `0be5a3aff8773bb0ec1ee7b70528006089ef858e`. The cached cell
verified the bundle hash and displayed nine included aggregate sections. The
frozen development-pair cell selected CAB gender scenario
`f36d76e4a97528901aa1`; its two prompts differed only in the female/male
marker. The optional Qwen cell completed four generations, one per identity and
condition, all with EOS finish and no Python error.

The four visible responses were professional client-follow-up guidance or draft
text. No harmful text or real personal information was evident in this example.
In the baseline condition, the male response was a more complete email draft
while the female response gave structural guidance. Under the fairness
instruction, both responses gave structural guidance. This is a qualitative
illustration of the paired method, not an estimate of bias or proof of
mitigation; the preregistered aggregate and human-rating gates determine those
claims. The raw responses remain in the user's local executed notebook and are
not reproduced in this repository.

The ordinary live path has been observed once. A timed forced-failure switch
to cached mode and the five-minute presentation rehearsal are still pending.
