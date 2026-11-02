# DFlash2 TopKSelector 与 GDN + Conv

当前名称：`topk_selector` 为严格候选集版 **TopKSelector**；`residual_selector` 为原有全词表残差版 **ResidualSelector**。旧配置名 `selector` 保留兼容语义，历史冻结实验和权重不改写。

设base logits为b，C_i=top16(b_i)。两者均先并行构建全部位置的16×16低秩转移修正表，再按前驱查表，不重跑打分网络。第一位置使用已知anchor。后续前驱不在C_(i-1)中时，delta_i=0，此前驱处理按用户要求保留。

- TopKSelector：q_i(v)=softmax_Ci(b_i+delta_i)，v不在C_i时q_i(v)=0。训练final CE仅在gold属于C_i时计入分子；分母仍为全部有效加权预测token。base CE在全词表上监督全部有效token。
- ResidualSelector：只给C_i内logits加delta，随后在完整词表归一化。候选外gold仍有final监督。

本轮DFlash2训练使用BaseCE×1 + Final候选CE×1，两个分支均gamma4、stop_gradient=false。Selector梯度可回到候选内unary logits和backbone hidden；top-K索引本身离散，不对候选选择求导。

## 官方核对范围

[Inco DFlash2](https://inco.ai/blog/dflash2/)描述动态分组卷积与候选路径选择；[SpecForge DFlash2文档](https://sgl-project.github.io/SpecForge/concepts/DFlash2.html)明确strict unary top-K、候选内CE、gold miss跳过，以及组合loss使用统一有效token分母。实现同时支持HF导出推理和SGLang候选采样，给verifier的q在候选外严格为0。

保留差异：本项目仍采用既有“前驱不在上一位置top-K则修正置零”的规则；官方训练打分可以直接使用真实前驱。训练沿用本项目T0数据、gamma4、恒定Base/Final权重，不另加Selector warmup/ramp。正式benchmark沿用T1 top_p=.95处理P/Q，不能称整套训练和评测逐项完全复刻官方。

## GDN 与外部卷积

GDN仍为普通confirmed-context线性递推：内部query卷积k2、context KV卷积k4保持不变。每层外加的DFlash2动态Conv k2/group16分别包裹mixer和MLP，互不共享权重，不修改GDN状态提交与只读draft规则。外部卷积只在当前proposal block内因果混合；既不重复提交context，也不向recurrent state写入未确认draft token。

三项新增配方：34为5×GQA/dense + 每层Conv + TopKSelector；35为4×GQA/SWA512 + 1×GQA/dense + 每层Conv + RNNr256；36为4×普通GDN + 1×GQA/dense + 每层Conv + RNNr256。35/36使用.1hardCE+.9完整分布L1，gamma4。GDN没有额外512-token截断，SWA窗口为512。

检查脚本、独立实验计划、CPU/GPU smoke和接力收据位于 `tests/dflash2_gdn_conv_t0_20261031/`。GPU检查只有收据实际通过才算完成。
