# 运行辅助工具

## SGLang0.5.9 抓取兼容

在训练环境中将已安装的 SGLang0.5.9 复制到新目录再补丁：

```bash
python scripts/runtime/patch_sglang059.py --output ./runtime_vendor
export PYTHONPATH="$PWD/runtime_vendor:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

补丁对 L2norm 原始 SHA 有精确门禁，仅减少变长首调用的 JIT specialization，不修改 BF16 运算；另对未使用 vision FA3 的导入采用按使用报错，文本9B通过 MM SDPA。不能用于 `sglang_legospec/` 的另一版本。

原9B运行环境另使用 CuDNN9.16.0.29。需要该版本时独立安装并在启动前配置库路径，例如：

```bash
python -m pip install --target ./runtime_vendor/cudnn nvidia-cudnn-cu13==9.16.0.29
export LD_LIBRARY_PATH="$PWD/runtime_vendor/cudnn/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

首次在新 GPU/依赖上采用补丁前，需做原/新内核 BF16逐元素一致性与 native teacher/logits/packed 验证。包中保留原/新内核源码，历史微基准不代表整体吞吐。

## W&B 定时上传

```bash
# 使用本机已有登录或环境中的凭据；代码不会保存凭据。
python scripts/runtime/sync_wandb.py --root ./wandb \
  --entity YOUR_ENTITY --project YOUR_PROJECT --interval 1800
```

工具只运行 CPU 上传，使用离线 run 原路径、项目 override、`beta sync --no-skip-synced`，不会恢复或启动训练。`--dry-run` 预览，`--once` 单轮；并发由文件锁保护。CLI 成功仅记为同步成功，不能当作完整训练、最终权重验收或云端 finished 的证明；这些验收应按具体项目执行。
