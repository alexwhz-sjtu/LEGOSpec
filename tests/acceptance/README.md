# 可复用 acceptance/tau 测评

保留 `qwen3/` 与 `qwen35/` 的独立命令行 harness、统一最终 proposal verifier、分片续测、完整覆盖汇总和概率回归测试。没有带入一次性队列、原机器地址、正式面板、canonical histories 或历史结果。

- `qwen3/` 使用训练环境的 Transformers4.57.1。
- `qwen35/` 使用独立环境的 Torch2.9.1 + Transformers5.2.0，依赖见该目录 `requirements.txt`；将仓库加入 `PYTHONPATH`，不要把主项目的固定4.57依赖重新装进此环境。
- Qwen3.5 adapter 在 prompt prefill 后按 token 前向，并保存/回退 conv、recurrent 与 KV 状态，避免把 GDN 当作普通 KV cache 裁剪。当前固定 context 层 `[3,11,19,23,27]`。

示例（在仓库根目录）：

```bash
python tests/acceptance/qwen3/run.py \
  --target ./models/Qwen3-4B --draft ./runs/qwen3_4b/41/checkpoints/final \
  --source . --manifest /path/to/panel/manifest.jsonl \
  --output ./results/4b41/T0 --temperature 0 --shard 0 --shards 8
python tests/acceptance/qwen3/summarize.py \
  --results ./results/4b41/T0 --manifest /path/to/panel/manifest.jsonl
```

先按 `PROTOCOL.md` 准备自己的固定面板和 `manifest_metadata.json`（包括目标权重、tokenizer 和 manifest SHA）；非 smoke 模式会严格检查。所有分片完成后才汇总。T1 另用 `--temperature 1` 与独立目录；协议为 top_p0.95、max_new512、seed42、block8（7候选+1保证token），分别报告常规任务和长文领域 macro tau，不报告延迟加速。

跨 target 不共用错误的 token IDs：Qwen3.5 必须用自己的 tokenizer/template 和固定的原生 target-only MT 历史，记录真实原生长度。长文 native32768-cap 中不足32k的输入不能写成严格32k。

CPU verifier 检查：`python tests/acceptance/qwen3/test_verifier.py`。迁移 GPU 环境后应先做独立 target greedy 对齐、回退状态逐元素检查、重复请求隔离，再跑正式测评。
