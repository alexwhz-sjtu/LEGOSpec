# LEFOSpec handon

2026-11-03 整理的 LEGOSpec 开发交接副本。Python 包名仍为 `litespec`，便于保持模型注册和 checkpoint 导出的接口一致。

## 内容

- `litespec/`：DLite、DSpark、DFlash/DFlash2 和 composable_v3 训练代码；EAGLE3、DFlare、Anchor OPD；Muon/FSDP；独立模型导出。
- `configs/`：组件配方，包括 dense/SWA512/GDN、Conv k2/g16、Parallel/Markov/RNN、ResidualSelector 与 TopKSelector。
- `configs/qwen35_9b/`、`examples/qwen35_9b/`：当前 11 组 9B 配方，顺序 **35、36、31、32、01、05、28、41、23、10、34**。已取消的 37、04、08、19、24 不进入该清单。
- `scripts/data/`：数据 inspect/adapter/convert、多轮回答生成、旧版及异步 packed hidden 抓取。
- `scripts/runtime/`：隔离的 SGLang 0.5.9 抓取兼容补丁、CPU W&B 定时上传工具。
- `sglang_legospec/`：最新本地修改的 SGLang 推理实现，包括原生 LEGOSpec worker、SWA/ring cache、GDN 状态池、拒绝采样、Selector 支持与相关内核；保留 Python/Rust 构建源码。
- `tests/unit/`、`tests/acceptance/`：可复用的回归和 acceptance/tau 测评代码。

没有复制数据集、模型权重、hidden 缓存、实验结果、SSH 控制器、训练/测评队列、PID/状态文件或凭据；也不会控制原机器上的任务。

## 环境要分开

训练/抓取环境使用 **Linux、Python 3.11、PyTorch 2.9.1、Transformers 4.57.1、SGLang 0.5.9、FLA 0.5.2**。CUDA 版本按机器选择 `cu128` 或 `cu130`。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# CUDA13 示例；CUDA12.8 应替换 index-url。
python -m pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[gdn,dev]'
```

Hopper 上 FLA0.5.2 对部分 Triton 版本有反向正确性保护。`gdn` extra 提供 TileLang0.1.15；GDN 训练使用 `FLA_TILELANG=1`，不要删除上游保护。

`sglang_legospec/python/pyproject.toml` 所属推理 fork 使用另一套依赖（其当前声明为 Torch2.14.1/Transformers5.19.0），**不可覆盖安装到上述训练环境**。使用单独虚拟环境，并按该目录的依赖与实际 CUDA 环境构建。该 fork 的 LEGOSpec 原生入口接受 DFlash context 和 RNN/Selector 配方；不能把所有训练配方都视为已支持的原生服务模型。

Qwen3.5 的 HF acceptance adapter 使用独立的 Transformers5.2.0 环境，见 [测评说明](tests/acceptance/README.md)。

## 开始训练

先把目标模型和自己的缓存放在示例 YAML 指定位置，或直接修改路径。训练命令不会下载训练数据。

```bash
# 先检查最终参数，不启动 GPU。
python -m scripts.config.launch_composable --config examples/qwen35_9b/35.yaml --dry-run
# 确认数据、模型和空闲 GPU 后再执行。
FLA_TILELANG=1 python -m scripts.config.launch_composable --config examples/qwen35_9b/35.yaml
```

4B41 的例子是 `examples/qwen3_4b/41.yaml`。9B 示例保留 global batch16、512 anchors、block8、max_length4096、seed42、6epoch、Muon 和各自损失；只替换运行路径/日志名称。示例采用 `final_only` 与 `save_training_state: false`，输出最终权重及 `training_metadata.json`，不保存 optimizer/RNG，因此该模式的落盘权重不能提供完整训练状态续训。需要中途恢复时，显式选择保存训练状态的策略。

多节点入口为 `scripts/composable/run_training_multinode.sh`，每节点运行一次：

```bash
MASTER_ADDR=NODE0_INTERNAL_ADDRESS NODE_RANK=0 NNODES=2 NPROC_PER_NODE=8 \
  bash scripts/composable/run_training_multinode.sh examples/multinode.yaml --dry-run
```

另一节点设置 `NODE_RANK=1`。各节点使用相同 YAML/模型/数据和可互通的地址；示例两节点的 global batch 会随 world size 变化。没有附带原机器 keeper 或自动抢占逻辑。旧算法的 `scripts/config/*.yaml` 仅作参数模板，`configs/custom_draft.json` 是需要自行提供的兼容模型配置占位符；完整可直接改路径使用的 composable 示例在 `examples/`。

## 数据和 hidden

新数据先运行 `scripts/data/convert_dataset.py inspect` 检查五条记录，再选择 prompt 字段及 `multi`/`first`，创建 adapter 并转换。现有 adapter 在 `scripts/data/adapters/`。回答生成保留非空校验和多轮新生成历史，入口为 `scripts/data/regenerate_train_data.py`。

```bash
python -m torch.distributed.run --nproc-per-node=8 \
  -m scripts.data.capture_hidden_states \
  --target-model-path ./models/Qwen3.5-9B --target-model-backend sglang \
  --capture-profile qwen35_9b --target-layer-ids 3,11,19,23,27 \
  --chat-template qwen35 --data-path ./data/answers.jsonl \
  --output-path ./data/qwen35_9b/hidden/cache \
  --cache-dir ./cache/qwen35_9b/capture --max-length 4096 \
  --batch-size 2 --tp-size 1 --sglang-attention-backend triton --fail-on-error
```

层号均为 **零起 decoder 输出层**。9B 抓 `[3,11,19,23,27]`，另存第31层最终归一化 teacher hidden；4B profile 对应 `[1,9,17,25,33]`，teacher35。两种 raw-tar 布局分别严格检查，不能混用；loader 为兼容训练添加的 teacher-layer alias 不等于额外抓了一条 context 流。

优化入口保留 packed requests、异步 D2H、内存有界缓冲、流式 SHA、异步分片提交以及同身份断点续抓；Triton/Inductor 编译缓存按 rank 隔离。它只接受上述两个经过适配的 profile；任意其他模型/层集合使用旧入口 `scripts/data/prepare_hidden_states.py`。9B 抓取在训练环境走原生 SGLang，不能使用 HF4.57 伪装加载。

SGLang0.5.9 的可选 vision import 和 L2norm 动态长度 JIT 补丁见 [runtime 说明](scripts/runtime/README.md)。补丁只写入显式指定的新 vendor 目录并检查已知源文件哈希，不修改全局安装。

## 配方与回归

41：4×GQA/SWA512 + 1×GQA/dense、全层 Conv k2/g16、ResidualSelector r256/top16，Base0.1CE + Final(0.1CE+0.9L1)，γ4。

34：5×GQA/dense + Conv k2/g16、TopKSelector r256/top16、`candidate_only`，Base CE + Final候选 CE，γ4。ResidualSelector 的词表外尾部分布与 TopKSelector 的候选支持语义不同，不能互换。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/unit -q
```

CPU 测试覆盖目标 context/teacher 分离、两种缓存布局、模型组件、梯度、cold export、EAGLE3、Anchor OPD、Selector 与配置。GPU 测试按可用硬件显式运行；新交接集成不能用历史实验的 GPU 收据冒充重新 GPU 验证。

详细接口继续见 [组件字段手册](docs/COMPOSABLE_CONFIG.md)、[EAGLE3](docs/EAGLE3.md)、[Anchor OPD](docs/ANCHOR_OPD.md)。文件清单/来源和本次验证记录位于 `tests/handoff/`。

## 本次整理验证

198个主项目 Python 文件语法检查、12份训练示例 dry-run、4B/9B profile 合约与隔离补丁 smoke 已通过。完整模型 CPU 回归先遇到 yunchang 导入时初始化 CUDA 的环境限制；已将该导入延迟到实际分布式初始化，但远端 SSH 随后超时，尚未复跑。未重新进行 GPU 回归。具体状态见 `tests/handoff/validation.json`。
