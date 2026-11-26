# Pinned Model Adapter

`configs/inference.json` freezes the baseline and fairness-control system
instructions, model keys, 4,096-token input cap, 384-token generation cap, FP16,
eager attention, greedy seed 253, and stochastic sensitivity settings. The model
and tokenizer commit SHAs come from `configs/upstream_registry.json`; branch names
and unpinned revisions are rejected.

`src/inference/adapter.py` loads **one** model at a time with
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16,
device_map="auto", attn_implementation="eager", trust_remote_code=False)`.
It rejects CPU/disk offload, quantization, missing chat templates, and a model
that did not load in FP16. It applies each tokenizer's official chat template
to the same system/user messages with `add_generation_prompt=True` and
`truncation=False`. An input above 4,096 tokens fails before generation.

```python
from inference.adapter import decode_config, load_model

adapter = load_model("M1")  # Run in the pinned Colab CUDA environment.
record = adapter.generate("Example user prompt", "baseline")
sampled = adapter.generate(
    "Example user prompt",
    "mitigation",
    decoding=decode_config(sensitivity_seed=42),
)
score = adapter.conditional_log_likelihood(
    "BBQ formatted user prompt", "Full answer text", "baseline"
)
```

`generate` checks that the returned sequence begins with the exact input token
IDs, slices off that prefix, and decodes only new tokens. Its result records model
and tokenizer revisions, condition and instruction version, precision, seed,
decoding parameters, input/output token counts, raw and stripped output, finish
reason, and runtime. The likelihood method returns the mean log probability of
answer tokens only, allowing the BBQ scorer to compare full answer strings.

The [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/main/chat_templating)
describes `add_generation_prompt`; the [tokenizer API](https://huggingface.co/docs/transformers/main/main_classes/tokenizer)
documents `truncation=False`.

Unit fixtures verify the locked load arguments, no truncation, prefix slicing,
condition text, seed policy, and overlength failure. Actual Qwen and Phi loads,
repeatable greedy output, batch-order invariance, and the `1e-4` likelihood
tolerance still require the pinned Colab GPU run. This development milestone is
not approval for a confirmatory run.
