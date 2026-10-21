# Component Acceptance Benchmark v1

This protocol evaluates **the final conditional proposal**, including RNN or
Selector corrections. Training accuracy, teacher-forced overlap and acceptance
proxies are not benchmark results. Latency and speedup are outside this protocol.

## Frozen inputs

* Sampling seed: **42**. Regular datasets: randomly sample **min(128, N)**
  without replacement. LongBench v2: independently sample **min(50, N)** in each
  of its six `domain` categories, after determining eligibility for the 32k
  track. Sampling uses a SHA-256 rank of seed/dataset/category/source ID, rather
  than depending on a library's shuffle implementation. Never resample per model.
* GSM8K main/test; Alpaca/train; MBPP full/test; MATH-500/test;
  LiveCodeBench official `version_tag=v6` temporal increment (`test6.jsonl`,
  175 problems; explicitly not the cumulative `release_v6` union); MT-Bench 80 questions;
  LongBench v2/data.json. Exact immutable revisions are in `sources.json`.
  These are acceptance workloads, not official task-score leaderboard runs.
* Qwen3-4B tokenizer and chat template, `enable_thinking=False`, no additional
  system prompt. Save exact input token IDs, prompts, original IDs, source
  revisions and hashes in a single manifest. No answer/reference solution is
  appended to a user prompt. MBPP includes its public test examples.
* MT-Bench retains both turns. Turn 2 uses a **frozen target-only greedy answer**
  to turn 1, generated once with the same target, max 512 new tokens. This
  canonical history is shared across all draft models and both sampling modes;
  it is not each draft model's own earlier answer. Record its hash separately.
* Regular prompts: at most 8,192 input tokens, preserving the tail if necessary;
  explicitly record truncation. LongBench v2 is a derived **32k input track**:
  preserve the question and options, head/tail truncate only the context until
  the complete chat input is 32,768 tokens (tolerance 64 tokens). Use only
  records whose untruncated chat input reaches this length; never pad short
  documents. Report eligible and selected counts per domain. Record exact
  original token counts when fully tokenized; for very large sources record
  original character counts and a conservative token lower bound, not a
  fabricated exact token count. It is not an
  official LongBench v2 quality score. Selection is category-balanced, not
  length-distribution representative.

## Shared generation and verification

* Batch size 1; BF16 target/draft weights; shared FP32 probability processing;
  512 maximum new tokens; natural tokenizer/model EOS stopping; no minimum
  output length, repetition penalty or forced continuation.
* Two settings: **T=0 (greedy)** and **T=1, top-p=0.95, top-k=0**. The latter
  applies identical temperature/top-p processing separately to the final
  conditional proposal q and target distribution p, including sparse Selector
  support. Proposal probabilities must match the actual sampled candidates.
* Block size 8 means **7 proposed tokens + 1 guaranteed token per cycle**.
  Freeze target model files and model config/weights hashes. Inference backends
  may differ for compatibility, but record them; no parameter changes.
* Greedy: longest matching prefix against target argmax, then target replacement
  or bonus. Random: accept token x with min(1,p(x)/q(x)); on first rejection
  draw from normalize(max(p-q,0)); if all 7 accepted, draw target bonus.
  Crop both target and draft state to the confirmed prefix. All models use
  this one verifier, not their separately optimized native sampling loops.
* Each request uses a seed derived from protocol seed, sample ID and sampling
  setting. Use separate proposal and verifier RNG streams; execution order,
  GPU sharding and model enumeration must not change a request's RNG seed.

## Acceptance results and selection

Save per-request JSONL including every cycle, accepted draft tokens k, logical
cycle length tau=1+k, actual committed tokens, EOS/budget censoring, generated
IDs, and failures. Save model/manifest/protocol hashes and environment metadata.

The primary statistic is **mean tau over complete, uncensored cycles**. Also
report mean accepted draft tokens, cycle-weighted and request-macro tau, the
committed-token/cycle ratio including terminal cycles, censoring rate, number of
requests with zero complete cycles, and position-wise acceptance. Do not hide
short-answer or failed requests; zero complete cycles is null, not zero or one.
Report regular tasks individually and their unweighted macro average. Report
LongBench by domain and the unweighted domain macro separately, never let its
larger sample count dominate the regular-task average. Report T=0 and T=1
separately. Paired request bootstrap: seed42, 2,000 replicates.

For the staged architecture search, use a **separate fixed selection panel**
(seed43, disjoint source IDs, max32 per regular task / max10 per long domain),
not the reported seed42 test panel. Select loss by the unweighted mean of the
five regular-task request-macro tau (GSM8K/Alpaca/MBPP/LiveCodeBench/MATH500)
across both temperatures and both RNN/parallel proposals. MT-Bench's entire
80-question pool is used for the test panel, so it is excluded from loss
selection; small LongBench domains need not have leftover selection samples.
Missing runs or undefined task means block automatic selection.
Tie within 0.01 tokens: hard CE, then NLA, then CE+TV, then D-PACE (predeclared
simple-rule order). Freeze winner before context/conv training. The test panel
is for the final report and is never consulted by the automated selector.

## Storage and integrity

`manifest.jsonl` and `selection_manifest.jsonl` are immutable after creation.
`manifest_metadata.json` records sources, counts, eligibility, tokenization,
canonical histories, software and all SHA-256 hashes. Per-model/setting/shard
outputs resume by exact request ID under an exclusive lock. A changed model or
manifest must create a new result directory; never merge incompatible files.
Result summaries require all scheduled request IDs or explicitly list failures
and mark the panel incomplete. Smoke tests are labeled and never fill the paper
result table. Runtime `.cache`, pycache and compiled kernels are excluded from
source freeze manifests.
