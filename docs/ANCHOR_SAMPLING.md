# Matching OPD and SFT anchors

The composable CLI defaults to `--anchor-sampling sample_epoch`. Both sampled
OPD and ordinary teacher-forced objectives use the same complete-block sampler
and default to 512 anchors. Explicit `--num-anchors` values still take priority.

For hidden-cache inputs, an anchor layout is a pure function of the sampler
version, `seed`, `epoch`, `sample_index`, tokens and supervision mask. Per-sample
CPU generators isolate this choice from model initialization, draft token
sampling, verifier uniforms and CUDA RNG. Padding, sample order, DP rank and
microbatch partitioning do not change a sample's selected positions. Epochs are
stamped by the input iterator before accumulation prefetch. Replay and spill
retain the collected layout. Different epochs use different anchor draws.

Token-only inputs without `sample_index` use a hash of unpadded tokens and masks
as their identity. Both sides of a comparison must use the same identity
convention; the paired configurations use the same hidden cache.

`--anchor-sampling legacy` selects historical randomness. Resume checks reject
changes to the anchor contract, seed or anchor count; use `init_from` for a new
experiment. Historical checkpoints can still supply initialization weights.

The paired templates are:

- `scripts/config/anchor_comparison_sft_rnn.yaml`: original `0.1 CE + 0.9 L1`.
- `scripts/config/anchor_comparison_opd_rnn.yaml`: sampled OPD-Survival.

Both initialize from the 01 final RNN checkpoint and use the regenerated 80k
cache, seed 42, 512 anchors, block size 8, max length 4096, batch size 2 per GPU
and the same Muon/AdamW settings. The current formal configuration uses four
epochs, `muon_lr=0.01` and `muon_adamw_lr=3e-4`, without a step cap. A separate
server controller runs short validation jobs before the formal pair and then
evaluates both checkpoints. Matched update counts do not imply matched
wall time: OPD additionally runs the frozen target on sampled branches.

Tests and real-cache audit receipts are in `tests/anchor_alignment_20261030/`.
The cache audit reads token/mask fields only, without loading hidden tensors.
