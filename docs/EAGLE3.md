# EAGLE3 in Composable

EAGLE3 uses `context.type=eagle3` and `backbone.type=eagle3`, with one dense
GQA layer and no `proposal` field or module. `block_size=8` means seven TTT
predictions plus the known anchor. The selected context layers are independent
of draft depth; the Qwen3-4B configuration uses `[1,9,17,25,33]`.

The target features are concatenated and linearly projected without a context
norm. Each decoder step separately normalizes the hidden and token embedding,
concatenates them before Q/K/V projection, and applies attention, residuals and
MLP. The **raw** output recurs; a trainable output RMSNorm precedes the shared,
frozen target LM head. The shared embedding and head remain outside draft
checkpoint weights and are provided through `SharedOps` at inference.

## Training

Use the existing launcher with either example YAML after setting the target,
hidden-cache and output paths:

```sh
python scripts/config/launch_composable.py --config scripts/config/composable_eagle3_sampled.yaml --dry-run
python scripts/config/launch_composable.py --config scripts/config/composable_eagle3_all.yaml --dry-run
```

Remove `--dry-run` to launch a new run. The examples use the current six-epoch
Muon/AdamW runtime settings; optimizer and scheduler flags remain configurable.
Online target feature acquisition through the existing HF/SGLang backend and
offline hidden caches use the same TTT objective. Offline training does not run
a target transformer per batch.

`--supervision-mode sampled` is the default. It uses the current anchor sampler
and `--num-anchors` (default 512 with `sample_epoch`). `sample_epoch` keys anchors
by sample identity, epoch and seed, preserving them across padding and batch
partition changes. `legacy` retains the shared RNG sampling strategy.
`--supervision-mode all` enumerates all eligible complete blocks and ignores
the anchor-count limit. Both exclude anchor zero (there is no preceding target
hidden) and require every token of the anchor-plus-predictions block to be
supervised and nonpadding. An all-padding or too-short sample contributes zero
terms; the runner rejects optimizer windows with no supervision.

For anchor `a`, TTT depth `r=0..K-1` consumes `x[a+r]` and predicts `x[a+r+1]`.
The initial feature is target hidden `a-1`; teacher logits come from the target's
normalized final hidden at `a+r`. The target feature cache contract is unchanged.

**Both modes run K full-sequence forwards.** The training cache has K physical
chunks of length L. A query sees the causal first-chunk prefix and its own
diagonal in subsequent chunks. RoPE uses `i+r`, not physical slot `r*L+i`.
Only selected outputs are gathered and scored in objective chunks, avoiding
full-sequence multi-depth vocabulary tensors. Hidden and K/V gradients remain
connected across depths; checkpoint recomputation does not mutate KV history.

The default objective is `soft_ce`, with
`loss_decay_gamma=4.481420117724551`, giving depth weights `0.8**r`. It uses the
framework's weighted-token denominator, **not** DeepSpec's local `B*L`
normalization. There is no second TTT decay. Existing hard CE, soft CE, forward
KL, TV, L1, NLA, weighted token sums, BV, D-PACE, TV-Survival and offline
OPD-Survival retain their native reductions. Block objectives keep their own
weighting semantics; setting gamma does not override objectives that explicitly
ignore positional decay. BV uses the autoregressive policy without the parallel
score floor and still requires full-vocabulary T1 provenance. Selector objectives
and online `opd_data_mode=sampled_anchor` are unsupported for this fixed-data TTT.

Training emits the existing token accuracy, loss, learning-rate,
`train/walk_accepted_length` and `train/expected_accept_len` keys. The two lengths
include the known anchor and remain fixed-data training diagnostics. Metrics use
the selected supervision starts; `supervision/anchors` records their additive
count. Checkpoint metadata records supervision mode and TTT length. Switching
supervision mode requires a new training run or `init_from`, not a resume.

## Inference and acceptance evaluation

`draft.spec_generate(target, input_ids, max_new_tokens, stop_token_ids,
temperature, top_p)` returns prompt plus generated IDs; `get_last_decode_stats()`
exposes cycle diagnostics. The inference helper `generate_eagle3` returns the
benchmark result dictionary directly. Inference currently accepts one unpadded
Qwen3 prompt per request and supports greedy or temperature/top-p sampling.

Prompt prefill pairs target feature `i` with token `i+1`, including the first
target-selected token. Subsequent drafts pair the previous draft hidden with
the sampled token embedding. Target and draft KV are separate. Verification
keeps the target's accepted prefix, discards all temporary draft KV, then
rebuilds draft entries using true target features and the accepted successor
tokens plus correction/bonus. The new correction does not yet have target KV.
Stop tokens and budget-truncated cycles follow the existing benchmark accounting.

The new harness is `tests/acceptance/qwen3/run.py`. It accepts the
existing `--target --draft --manifest --output --source --temperature` arguments
and shard options, uses the frozen panel without regenerating datasets, and
keeps the component-acceptance-v1 verifier, per-request RNG streams, T0/T1
top-p settings and summary definitions. EAGLE3 does not require a mask token.
The historical experiment harness and queues are not edited. This harness
measures acceptance, not latency or inference speedup.

## Validation and attribution

Tests live in `tests/unit/test_composable_eagle3.py`; upstream reference files
and their SHA-256 manifest live in `tests/eagle3_integration/reference/`.
The reference revision is DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2`.
The decoder equations, TTT mask and cache rebuild are based on DeepSpec and its
SpecForge adaptations. Upstream MIT license and NOTICE, including SpecForge's
Apache-2.0 attribution, are retained beside the reference files.
