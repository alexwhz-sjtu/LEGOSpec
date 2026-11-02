# 配方

根目录保留可复用的4B/组件模型 JSON；`qwen35_9b/` 是当前11组9B配置，按 `recipes.json` 顺序执行。

模型 JSON 仅定义模型/组件，运行参数在 `examples/` 的 YAML。9B 使用 hidden4096、vocab248320、32层 teacher、GQA16/4头、head_dim256；4B 使用对应原生尺寸。context 层、SWA窗口、Conv、proposal、loss及γ均保留其配方定义。

`examples/qwen35_9b/NN.yaml` 对应 `configs/qwen35_9b/NN.json`。`examples/qwen3_4b/41.yaml` 给出新增4B41。`custom_draft.json` 仅作为旧算法 YAML 中的占位文件名，不是本包提供的配方。
