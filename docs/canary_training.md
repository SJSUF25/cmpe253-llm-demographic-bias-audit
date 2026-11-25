# E6-T03 Controlled Canary Adapter Training

Run `make train-canary ROOT="$AUDIT_SHARED_ROOT"` in a clean pinned Colab GPU
checkout after building and validating the private membership corpus. The training
script checks the corpus manifest and regenerates its expected 3,200 presentations
before loading Phi. It does not read the sealed non-member attack partitions.

The frozen configuration is `configs/canary_training.json`: Phi-4-mini-instruct M2,
NF4 with double quantization and FP16 compute, rank-16/alpha-32/dropout-0.05 LoRA on
all linear layers except the output head, three epochs, 2 examples per microbatch,
8-step gradient accumulation, paged AdamW 8-bit, 2e-4 learning rate, 0.01 weight
decay, cosine schedule, 3% warmup, and gradient clipping at 1.0. The official chat
template masks the entire user prompt and refuses sequences over 512 tokens. No
example packing or truncation occurs.

Each epoch has a deterministic shuffled **presentation index** order. This matters
because repeated canaries share record IDs. The full index order, adapter,
optimizer/scheduler state, file hashes, training loss, resolved LoRA module names,
model revision, data hash, code commit, and package-lock hash are saved privately:

- `adapters_private/canary_m2_qlora/epoch_1` through `epoch_3`;
- `manifests/canary_training/epoch_1.json` through `epoch_3.json`.

The resolved module names and exact input/configuration hashes are recorded in
`manifests/canary_training/plan.json` before the first optimizer step.

An interrupted epoch is restarted from the preceding verified epoch checkpoint.
Training cannot resume after a code/configuration/data change or from a changed
checkpoint. The epoch-3 adapter is selected without reference to attack results.
Keep the adapter and optimizer state in Drive; they are not Git artifacts.

Local unit tests check source hashes, exclusion by the generator contract,
deterministic shuffle order, prompt masking, length rejection, and checkpoint
tampering. They do not verify GPU memory, bitsandbytes/PEFT integration, or the
actual three-epoch result. Those require the Colab run and its receipts.
