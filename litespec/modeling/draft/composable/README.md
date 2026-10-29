# Composable draft v3

模型执行 Context → LayerStack → Proposal；训练目标单独由 Objective 负责。
每层明确选择 mha/gqa/mla/gdn 和 dense/sliding_window/dsa/linear，参数在配置末尾集中。
共享配置不共享权重。统一 backbone 支持任意层数、任意 MHA/GQA/MLA/GDN 排列。
GDN 仅 linear；DSA 整塔检索一次共享，仅 MHA/GQA，不支持 Selector。

- [字段手册](../../../../docs/COMPOSABLE_CONFIG.md)
- [HTML 阅读版](../../../../docs/COMPOSABLE_CONFIG.html)
- [所有配方](../../../../configs/README.md)

手写JSON仅包含模型组件；计算backend、数据、batch、优化器和调度放运行YAML/CLI。
配置只支持composable_v3，不提供历史checkpoint兼容；已有权重和冻结实验不作改写。
模型输出、loss公式、Selector并行转移表、GPU事件布局、hidden捕获保存优化保持现有实现。
只有已确认context提交到缓存，各层成功后统一发布新状态；GDN已确认历史回退需状态快照或重算。

独立导出通过 export_composable 保留推理依赖，冷加载无需训练包或本仓库导入。
验证记录保存在 tests/attention_v3_20261025；本轮不启动训练或正式测评。
