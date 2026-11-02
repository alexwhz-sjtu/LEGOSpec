# Sampled-anchor online OPD

`--opd-data-mode sampled_anchor` samples current Parallel/RNN proposals at
random anchors in the existing training corpus. It does not generate complete
answers, corrections or bonus tokens. The old `teacher_forced` mode remains the
default, including its offline empirical acceptance rule.

## Inputs

Both existing data inputs work:

- `--train-data-path`: compute selected context hidden and target KV in one
  prefix forward per document.
- `--train-hidden-states-path`: use saved context hidden and final hidden from
  the existing cache. Load a complete frozen target as well, build its prefix
  KV once and score the actual draft branches. Cached hidden does not replace
  per-layer target KV. Original continuation hidden is never used to supervise
  draft-induced continuation states.

The same cache manifest/model/layer checks used by offline training apply.
Cached features must belong to the exact frozen target checkpoint. The raw
selected-layer features go through the trainable draft context projection.

Selected decoder outputs, including the last block, are captured before final
normalization. Teacher anchor and branch features remain post-final-norm for
the frozen LM head. Updated HF hidden caches retain these two streams
separately; legacy caches keep their recorded feature convention.

## Full-batch execution

All anchors are processed together. Both `objective_chunk_blocks` and
`opd_target_chunk_anchors` must be **0**. Positive values fail validation; an
OOM is reported without reducing anchors or enabling chunking automatically.

1. Sample and sort complete-block anchors using the existing eligibility rule.
2. Obtain context hidden, from the target or the cache.
3. Run the trainable context fusion/backbone once. Sample from detached features.
4. Score every sampled branch using shared prefix KV. Each branch sees source
   positions through its own anchor and its own causal token history. Prefix KV
   is never replicated per anchor. Only prediction hidden is returned, then KV
   is released.
5. Replay the proposal with `[anchor, sampled_tokens[:-1]]` and compute the
   full-vocabulary loss. CUDA fuses KL/overlap statistics to reduce temporary
   vocabulary tensors; this does not split the anchor batch or truncate vocab.
6. Backpropagate and update using the existing runner/FSDP normalization.

Sampling is full-vocabulary **temperature 1**, with no top-k/top-p truncation.
The verifier draws independent uniforms and accepts a sampled token with
`min(1, p(token)/q_sample(token))`. Acceptance stops at the first rejection.
There is no argmax-match verifier in this mode. RNN state resets for every
block; sampling and replay use the same predecessor chain and zero dropout.
EOS does not truncate the fixed-horizon training block.

Accepted positions use forward KL, all remaining positions use reverse KL with
absolute-position decay, and accepted positions plus the first rejection use
Survival. Existing defaults remain `beta_rej=1`, `rejection_decay=0.8`,
`survival_weight=1`. There is no confidence objective.

## Accumulation and resume

Accumulation one performs no metadata backbone prepass. For larger windows,
collect sampled blocks/teacher hidden/acceptance boundaries first, reduce the
three component denominators over the complete window and DP ranks, and
replay those fixed records with gradients. Target scoring is not repeated.
`window_buffer_mb` bounds CPU record storage; excess records use the existing
trusted temporary spill mechanism and are deleted after consumption.

The composable CLI defaults to `--anchor-sampling sample_epoch`. Both online
OPD and teacher-forced objectives receive the same precomputed layout. A
private CPU generator keyed by `(version, seed, epoch, sample_index)` samples
complete supervised blocks; padding, rank, batch splitting and unrelated RNG
consumption do not change a sample's anchors. If a token-only input does not
supply `sample_index`, its unpadded tokens and masks provide a content identity.
Comparisons must use the same identity convention (the same cache is simplest).
The shared default is 512 anchors; an explicit `--num-anchors` overrides it.

Epochs are attached when the input iterator yields a batch, including when an
accumulation window crosses an epoch. Prepared layouts survive spill/replay.
Fresh OPD proposal and verifier draws still use the checkpointed model RNG;
replay consumes no new sampling randomness. Checkpoints record the anchor
version, seed and count and reject resume with a different anchor contract.
`--anchor-sampling legacy` preserves historical anchor randomness. Changing
sampling/data mode or context source requires a new run via `init_from`.

Matched 01-weight/cache comparison templates are
`scripts/config/anchor_comparison_opd_rnn.yaml` and
`scripts/config/anchor_comparison_sft_rnn.yaml`. Both share the
Muon/AdamW settings, seed, sample order, complete blocks, batch size and update
budget. SFT uses the original `0.1 CE + 0.9 L1` objective; OPD uses sampled
OPD-Survival. Current formal settings are four epochs, Muon LR 0.01 and AdamW
LR 3e-4. The server controller in `tests/anchor_opd_comparison_20261030/` owns
short validation, both formal fits and the matched T0/T1 evaluations.

## Launch

Use `scripts/config/anchor_opd_online_hidden.yaml` or
`scripts/config/anchor_opd_cached_hidden.yaml` with the existing composable
launcher. They are ten-update templates using the trained Parallel checkpoint;
choose a fresh output directory. For RNN, replace both `draft_config` and
`init_from` with the matching RNN recipe/checkpoint. No formal job is launched
by adding these files.

Requirements: pinned project environment, dense full-attention Qwen3 with
standard RoPE, HF target, collocated GPUs, TP=1, Parallel/RNN and OPD-Survival.
Embedding/LM head reuse the resident frozen target. CUDA scoring uses compiled
FlexAttention; dense CPU SDPA is the numerical reference, not a GPU fallback.

## Verification

Numerical tests and isolated GPU/FSDP smoke tools are in
`tests/anchor_opd_20261029/`. The GPU smoke runs two updates and exports a JSON
receipt; `--full` uses the actual Qwen3-4B target, trained drafter, batch 2,
sequence length 4096 and 512 anchors. Compile time is included in smoke timing.
Small-model tests do not establish full-size memory feasibility.

Shared-anchor checks are in `tests/anchor_alignment_20261030/`. CPU model and
adapter tests cover both Parallel/RNN and CE/L1/weighted-SFT objectives. The
real-cache audit reads only tokens/masks and compares exact anchor layout
fingerprints across objectives, epochs, padding and batch sizes.
