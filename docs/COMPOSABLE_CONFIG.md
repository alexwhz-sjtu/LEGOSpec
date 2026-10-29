# Composable v3 配置字段手册

本手册对应当前源码的 composable_v3。所有模型配方使用标准 JSON；每层选择 attention/access，数值参数在文件末尾共享。共享的是超参数，各层权重、槽位 embedding 和状态独立。本轮不提供 v1/v2 checkpoint 加载兼容，也不启动训练。

## 1. 读取方法与配置边界

以下字段路径均位于 composable_config 内，除非注明“顶层”。未知字段、未使用的共享配置、逐层数值覆盖和非法组合均报 ValueError；错误会指出字段或冲突。字符串区分大小写；数字不能用字符串或布尔值代替；所有浮点数必须有限，JSON 不使用 NaN/Infinity。null 仅在明确允许的位置有效；空对象只可用于使用默认值的共享配置，不能作为 layer。

顶层手写 JSON 仅允许 architectures、schema_version、block_size、composable_config。target 路径、backend、数据、优化器和 batch 放运行 YAML/CLI。保存后的 config.json 还含自动解析的 HF/执行元数据，不应作为手写配方中的覆盖入口。

attention_configs 与 access_configs 放在 composable_config 末尾；每种类型仅一份。先声明层，再填写共享配置。层数由 layers 长度派生；运行参数 num_draft_layers 不覆盖显式列表。

## 2. 顶层与结构字段

## 顶层手写字段

### `architectures`

可选字符串数组。唯一合法值为 ["ComposableDraftModel"]；省略时自动设置，不能借此选择 attention。null/空数组不合法。

### `schema_version`

必填字符串，唯一选项 composable_v3。v1/v2 明确拒绝；不自动迁移旧 checkpoint。

### `block_size`

可选整数，>=2。包含一个 anchor 和 block_size−1 个预测槽；默认运行参数 block_size（默认8），直接构造 Config 时默认8。null/0/false不合法。所有层、proposal、objective 共用。

### `composable_config`

必填对象，包含本手册所列模型组件；不能以省略包装的旧格式替代。

## composable_config 的直接字段

### `context`

必填对象，选择 target hidden 注入。

### `backbone`

必填对象，声明完整层序列。

### `proposal`

必填对象，选择输出与采样补偿 head。

### `objective`

必填对象，选择训练目标。

### `mask_token_id`

可选整数，[0,vocab_size)。省略由运行系统解析并保存：显式CLI > 配方值 > tokenizer mask token > 未使用的模型词表行。0可以是有效ID；null/false非法，不扩词表。

### `attention_configs`

必填对象，键只能为 mha/gqa/mla/gdn，必须与 layers 实际使用类型完全一致。每个值为对应参数对象，{}使用默认值；不允许未使用的类型或null。

### `access_configs`

可选对象，键为 dense/sliding_window/dsa/linear，只能配置已使用方式。省略或缺少已使用键时填入其默认值；dense与linear没有参数，值为{}。

## context.*

### `type`

必填字符串：dflash=拼接所选 target hidden→Linear→RMSNorm，所有层共享所得 context；dflare=各层独立学习静态 target 层融合权重。均不复用 target attention KV。

### `target_layer_ids`

可选整数数组或null，非空、升序、无重复，范围[0,num_target_layers)。指 transformer 层输出，HF hidden_states 取 layer_id+1。省略/null使用既有 DFlash 深度选层；CLI显式 target_layer_ids 优先。数量不必等于 draft 层数。

## backbone.*

### `type`

必填字符串，唯一选项 layer_stack。旧 transformer/gdn_context_memory/gdn_mha_hybrid 不接受。

### `layers`

必填非空数组，逐项对应一层，允许MHA/GQA/MLA/GDN任意次序和层数；空layer不合法。缓存、target context映射和gradient checkpointing均按实际层索引。

## backbone.layers[i].*

### `attention`

必填字符串：mha=等Q/K/V头数softmax；gqa=分组KV softmax；mla=共享低秩KV latent后展开softmax；gdn=Gated DeltaNet递推混合器。只填类型，不能写内嵌参数对象。

### `access`

必填字符串：dense=全部合法历史＋本anchor完整block；sliding_window=历史[a−W,a)＋完整block；dsa=整塔共享top-k检索历史＋完整block；linear=GDN递推历史读取。具体搭配见限制章节。

### `local_operator`

可选对象，省略等同{"type":"none"}。只支持非GDN层的dflash2_conv，不改变attention访问mask。

## backbone.layers[i].local_operator.*

### `type`

必填字符串none或dflash2_conv。none无其他参数；dflash2_conv为attention和MLP分别添加动态分组卷积，二者不共享权重。GDN只允许none。

### `kernel_size`

dflash2_conv时可选正整数，默认2，必须<=block_size；null/0不合法。

### `group_size`

dflash2_conv时可选正整数，默认16，必须整除hidden_size；none时禁止填写。

## attention_configs.mha.*

### `num_attention_heads`

可选正整数，默认target Q heads；必须与target Q heads一致，不能以此改变Q头数。

### `num_key_value_heads`

MHA默认等于Q，且必须与Q相等；GQA默认target KV heads，可显式修改，必须整除Q。不是KV latent rank。

### `head_dim`

可选正偶数，默认target显式head_dim；target缺失时使用其配置类计算结果，不用hidden_size/Q覆盖显式值。可显式修改，Q/K/V均使用该维度。

## attention_configs.gqa.*

### `num_attention_heads`

可选正整数，默认target Q heads；必须与target Q heads一致，不能以此改变Q头数。

### `num_key_value_heads`

MHA默认等于Q，且必须与Q相等；GQA默认target KV heads，可显式修改，必须整除Q。不是KV latent rank。

### `head_dim`

可选正偶数，默认target显式head_dim；target缺失时使用其配置类计算结果，不用hidden_size/Q覆盖显式值。可显式修改，Q/K/V均使用该维度。

## attention_configs.mla.*

### `num_attention_heads`

可选正整数，默认16，独立于target Q heads；展开Q/K/V头数由此确定。

### `num_key_value_heads`

可选正整数，默认16，当前展开式MLA必须等于num_attention_heads。不等直接报错，不支持另一套GQA展开方式。修改头数时应同时修改两者。

### `q_lora_rank`

可选null或正整数，默认null=hidden直接投影到所有Q heads；正整数=共享Q低秩down→RMSNorm→up。不是每头rank，也不是额外LoRA微调模块。0/false非法。

### `kv_lora_rank`

可选正整数，默认512；所有heads共享的KV压缩latent宽度，不乘head数。

### `qk_nope_head_dim`

可选正整数，默认128；每个Q/K head不做RoPE的部分。

### `qk_rope_head_dim`

可选正偶数，默认64；每个Q/K head做RoPE的部分。QK总维度=NoPE+RoPE，默认192。

### `v_head_dim`

可选正整数，默认128；每个展开V head宽度，输出投影为heads×v_head_dim→hidden_size。

### `rope_interleave`

可选布尔值，默认true=交错旋转；false=NeoX分半旋转。null/0不合法。

## attention_configs.gdn.*

### `linear_num_key_heads`

可选正整数，默认16，同时决定Q/K heads。Q/K投影宽度=heads×linear_key_head_dim。

### `linear_num_value_heads`

可选正整数，默认32，必须为linear_num_key_heads整数倍。Q/K在递推读取时按组扩展至V heads，层参数不共享。

### `linear_key_head_dim`

可选正整数，默认128；Q/K每头维度。

### `linear_value_head_dim`

可选正整数，默认128；V每头维度。V投影宽度默认4096，输出回hidden_size。

### `query_conv_kernel_size`

可选正整数，默认2；Q因果深度卷积核宽。block模式每个anchor独立，前缀补零；confirmed_history模式必须为4。

### `context_conv_kernel_size`

可选正整数，默认4；沿已确认context的K/V因果深度卷积核宽，持久缓存保存kernel−1个历史投影。kernel=1合法且无历史卷积尾部。

### `query_conv_mode`

可选字符串block或confirmed_history，默认block。confirmed_history的首槽卷积输入为[q_ctx(p−3),q_ctx(p−2),q_ctx(p−1),q_slot0]，后续槽逐步替换历史；q_ctx由同层input_norm/Wq生成，不使用target自己的attention Q。短历史同文档补零，anchor分支不串接。

## access_configs.dense

dense的值只能是{}，没有数值字段。访问合法历史和本anchor完整block；不能访问未来context或其他anchor。

## access_configs.linear

linear的值只能是{}，仅GDN使用。已确认context写入递推/卷积状态，draft槽读取状态，不写历史。linear不是额外可切换的GDN算法；卷积模式由GDN参数选择。

## access_configs.sliding_window.*

### `window_size`

可选正整数，默认512。以绝对位置计历史窗口W，不包含本block；同一block所有槽使用同一历史范围[a−W,a)。持久缓存每样本最多保留W个已确认KV。0/null非法。

## access_configs.dsa.*

### `num_query_heads`

可选正整数，默认4；indexer Q heads，与主attention heads独立，indexer K为共享单头。

### `head_dim`

可选正整数，默认128；indexer每头维度，独立于主attention head_dim。

### `rope_head_dim`

可选非负偶数，默认64，必须<=indexer head_dim；0表示indexer不做RoPE。

### `index_topk`

可选正整数，默认1024；先合法历史mask再检索。短历史全选，空历史只访问当前block。

### `anchor_chunk_size`

可选正整数，默认16；indexer/稀疏执行的anchor分块数，不改变模型top-k或loss的objective_chunk_blocks。

### `rope_theta`

可选有限正数；省略继承target rope_theta，控制indexer旋转频率。null/0不合法。

### `stage`

可选字符串warmup或sparse，默认sparse。warmup访问完整历史并只训练indexer KL；sparse使用top-k并联合主目标。切换阶段和优化器由训练流程负责，配置不会自动启动任务。

### `execution`

可选字符串gather或reference，默认gather。gather用普通gather＋SDPA；reference用完整KV与稀疏mask核对。两者均为SDPA，不是FlexAttention/eager分支。

### `enabled`

可选布尔值，默认true；false仅用于推理/数值reference，将访问退回dense，不产生indexer辅助项，新训练入口拒绝false。

### `loss_weight`

可选有限非负数，默认1.0；sparse阶段indexer KL系数，主目标与KL分别按各自有效计数跨rank/累积窗口归一化。0关闭联合KL贡献；warmup仍优化KL，不用该联合权重。

## Proposal选项

proposal.type选项为parallel、markov、rnn、sequential_rnn、selector。parallel仅用base logits；markov是前驱token条件的加性修正；rnn额外使用递推hidden并加性修正；sequential_rnn使用顺序替换head。RNN状态每个block重置。Selector先并行计算所有槽位top-K×top-K转移表，再按路径查表/归一化/采样；路径有依赖，但不按已采样前驱重跑打分网络。后续前驱不在上一槽base top-K时修正为0，final=base；当前真标签不在当前top-K仍正常用完整词表监督。

## proposal（parallel）.*

### `type`

必填字符串，当前对象选parallel；含义见上节。

## proposal（markov）.*

### `type`

必填字符串，当前对象选markov；含义见上节。

### `rank`

可选正整数，默认256；对应head的低秩/递推状态宽度。parallel禁止该字段，其他proposal之间不共享状态。

## proposal（rnn）.*

### `type`

必填字符串，当前对象选rnn；含义见上节。

### `rank`

可选正整数，默认256；对应head的低秩/递推状态宽度。parallel禁止该字段，其他proposal之间不共享状态。

## proposal（sequential_rnn）.*

### `type`

必填字符串，当前对象选sequential_rnn；含义见上节。

### `rank`

可选正整数，默认256；对应head的低秩/递推状态宽度。parallel禁止该字段，其他proposal之间不共享状态。

## proposal（selector）.*

### `type`

必填字符串，当前对象选selector；含义见上节。

### `rank`

可选正整数，默认256；对应head的低秩/递推状态宽度。parallel禁止该字段，其他proposal之间不共享状态。

### `top_k`

可选正整数，默认16，<=vocab_size；每槽从base logits选K个候选以预计算转移，不将其他词概率归零。

### `selector_mode`

可选字符串，v3唯一合法值full_vocab_residual：只对当前top-K加入转移delta，但在完整词表softmax。candidate_only不支持v3/新训练。

### `stop_gradient`

可选布尔值，默认false；true仅detach修正分支输入hidden，不冻结Selector，不切断base logits或backbone主损失梯度。

## Objective语义与组合

objective.type的逐位置选项：hard_ce=真实标签CE；soft_ce=target全词表软标签CE；forward_kl=KL(target||draft)；tv=0.5×sum|p−q|；l1=sum|p−q|；nla=−log(sum min(p,q))，overlap下限1e−5；lk_alpha为nla别名，不另提供alpha参数。除hard_ce外这些逐位置项均需冻结teacher logits。它们以有效token位置权重归一化。

block选项：dpace使用真实标签概率的detached前缀/后缀credit，需完整监督block但无需teacher；bv使用target/draft完整词表比率与前缀接受积分，需teacher和完整block，并要求运行参数bv_sampling_provenance=target_full_vocab_t1；tv_survival=标准TV＋survival_weight×(1−overlap前缀累积乘积)，需teacher和完整block。这三项保留自身block公式，不叠加固定位置gamma。tv_survival按完整block数×预测槽数归一化，dpace/bv按完整block数归一化。

weighted_sum仅组合逐位置目标，组件先使用相同有效位置权重，再加权；不允许嵌套weighted_sum或block目标。selector_losses只用于selector，分别计算base/final任意现有objective，按各自原生分母归一化后乘branch weight相加；有效分支的teacher、完整block需求取并集。任一有效BV分支仍要求对应轨迹来源。零权重分支不计算、不引入teacher/完整block要求。

## objective 通用字段

### `type`

必填字符串：hard_ce、soft_ce、forward_kl、tv、l1、nla、lk_alpha、dpace、bv、tv_survival、opd_survival、weighted_sum、selector_losses。含义和归一化见上节。

### `loss_decay_gamma`

可选null或有限正数，默认4.0。预测位置j从0开始，w_j=exp(−j/gamma)，分子、分母均乘有效权重；null关闭衰减。0/false非法。唯一设置位置为最外层objective，不能写proposal、Selector分支或weighted_sum组件。block目标不使用固定衰减。

## objective（weighted_sum）字段

### `type`

必须为weighted_sum。

### `loss_decay_gamma`

沿用最外层通用gamma规则；作为Selector分支时禁止自己设置。

### `components`

必填非空数组，元素只含type和weight；type限hard_ce/soft_ce/forward_kl/tv/l1/nla/lk_alpha；weight必填有限非负数，允许0，至少一项>0。例如0.1CE＋0.9历史TV(L1)写hard_ce/0.1＋l1/0.9，不能用tv冒充L1。

## objective（dpace）字段

### `type`

必须为dpace。

### `loss_decay_gamma`

通用字段允许保存，但block公式不使用固定decay；混合Selector另一逐位置分支仍使用父gamma。

### `alpha`

可选有限数[0,1]，默认0.5，smooth=(1−alpha)×draft_label_probability+alpha。0/1合法，null/false非法。

### `reduction`

可选字符串，唯一选项complete_block_mean，省略填入此值；旧batch/sequence reduction拒绝。

## objective（tv_survival）字段

### `type`

必须为tv_survival。

### `loss_decay_gamma`

block公式不使用固定decay，不能改变其survival前缀形式。

### `survival_weight`

可选有限非负数，默认1.0，0退化为对完整block的标准TV项；不等同完整词表L1。

## objective（opd_survival）字段

离线 acceptance-aware KL＋可达前缀 survival，需冻结teacher完整词表logits及完整监督block。预测位置从1开始，K不含anchor。当前forward的draft argmax与固定target标签的连续匹配长度为r，H=min(r+1,K)，包含第一个失败位置。r/H及接受、拒绝、可达标记停止梯度；target冻结，draft的KL和overlap保留梯度。这是固定target路径上的离线目标，不声称复现draft偏离后的在线target验证状态。

L=L_acc+beta_rej×L_rej+survival_weight×L_surv。接受项按k≤r的位置等权平均KL(p||q)；拒绝项按k>r的位置加权平均KL(q||p)，u_k=rejection_decay^(k−1)，使用block绝对位置，分母为拒绝位置权重之和；survival项为1−sum_m sum_(k≤H_m) prod_(j≤k)(1−0.5×sum|p_j−q_j|)/(M×K)。M为完整监督block数，H不足K时分母仍为M×K。三项直接相加，不额外除2。

三项分别按整个优化窗口、所有draft DP ranks的计数归一化；不能先对各chunk或各rank取均值再平均。空接受或拒绝集合对应项为0，无完整监督block的全局窗口报错。Selector各支路独立计算r/H，不复用另一支路的分组。weighted_sum不接受此block目标。

### `type`

必须为opd_survival。

### `loss_decay_gamma`

block目标不叠加固定位置gamma；本轮配置显式写null。

### `beta_rej`

可选有限非负数，默认1.0；控制拒绝KL项权重，接受项系数固定为1。

### `rejection_decay`

可选有限数，0<rejection_decay≤1，默认0.8；仅控制拒绝项的绝对位置权重，不从首次拒绝位置重新计数。

### `survival_weight`

可选有限非负数，默认1.0；0关闭survival项，仍保留接受/拒绝KL。

## objective（selector_losses）字段

### `type`

必须为selector_losses，proposal必须为selector；selector新训练也必须使用该wrapper。

### `loss_decay_gamma`

父层统一gamma，默认4，null关闭。base/final中的逐位置项继承；block项使用原block公式。

### `base`

可选objective对象，默认hard_ce、weight=0.1。

### `final`

可选objective对象，默认hard_ce、weight=1.0。当前top-K残差加到base完整词表logits后，对final计算该目标；前驱未命中时仍计算final=base的loss，修正网络在该位置无梯度。

Selector的base/final均可额外设置weight（有限非负数，默认base0.1/final1.0）；至少一个分支权重>0。分支可选除selector_losses以外的全部objective；weighted_sum分支内组件仍各自需要weight。不得在分支/组件设置loss_decay_gamma。

## objective.components[i] 或 objective.base/final.components[i]

### `type`

必填字符串，选hard_ce/soft_ce/forward_kl/tv/l1/nla/lk_alpha，禁止block目标、嵌套组合和Selector wrapper。

### `weight`

必填有限非负数，0合法，至少一个组件>0；不自动将组件系数归一为和1。

## objective.base.weight / objective.final.weight

### `weight`

可选有限非负数，base默认0.1、final默认1.0；0不计算该分支，至少一个分支>0；各分支先按原生分母归一化再乘权重。

## 运行字段与checkpoint execution_config

### `attention_backend`

运行YAML arguments.attention_backend / CLI --attention-backend，选eager/sdpa/flex_attention；composable默认sdpa。MHA/GQA/MLA的dense/SWA支持这些分支；DSA只支持sdpa。不属于手写模型JSON参数。

### `flex_attention_backend`

YAML对应字段/CLI --flex-attention-backend，选TRITON/FLASH，默认TRITON。仅attention_backend=flex_attention生效；要求dropout=0、QK/V每头维度>=16；FLASH需PyTorch>=2.11及支持的设备，不能静默回退。当前CUDA验证环境为PyTorch2.9.1，已验证TRITON，未验证FLASH。

### `gdn_backend`

新增YAML arguments.gdn_backend / CLI --gdn-backend，选auto/fla/reference，默认auto。auto在CUDA上使用FLA、CPU上reference；fla要求CUDA及已安装FLA，错误不降级到CPU；reference为小规模FP32扫描，验证用。

## 自动保存的target_attention参考元数据

### `num_attention_heads`

自动捕获target Q heads；供MHA/GQA默认值和Q头数一致性检查，不是MLA/GDN头数。

### `num_key_value_heads`

自动捕获target KV heads；供GQA默认值，不会把真MHA变为GQA。

### `head_dim`

自动捕获target显式head_dim。Qwen3-4B为128，不是2560/32得到的80。

## 自动字段、搭配限制和失败行为

保存后的HF config还含model_type=litespec_composable、architectures、num_hidden_layers（layers长度）、num_target_layers（target深度）、hidden_size、intermediate_size、hidden_act、vocab_size、rms_norm_eps、attention_bias、attention_dropout、initializer_range、rope_theta、rope_scaling/rope_parameters、max_position_embeddings、BOS/EOS/PAD、dtype、tie_word_embeddings、use_cache和Transformers标准元数据。它们从target text config继承；手写JSON不重复这些HF字段，也不作为覆盖入口。attention实际投影由共享配置决定，最终输出回hidden_size。

支持矩阵：MHA/GQA支持dense、sliding_window，或整塔dsa；MLA支持dense、sliding_window；GDN仅linear。其余搭配配置阶段拒绝。层内当前block按原DFlash规则可见，不访问别的anchor；target未来context与其他文档被隔离。DSA要求所有层MHA/GQA且全部access=dsa，在最开始从首层anchor的pre-RoPE Q检索一次，全层/全槽复用；不支持部分DSA、MLA/GDN混入、Selector或非SDPA backend。

每个attention类型在同一文件只能有一份参数，同一种access也只有一份配置；需要不同GDN卷积或不同滑窗W时用另一模型配方。layers内填写num_attention_heads、rank、window_size等均报未知字段；在共享attention中填backend也报错。profiles必须对应实际使用类型，不接受拼错的键或未使用条目。

持久缓存只提交已确认context；拒绝的draft不能写入。各种状态按真实层索引独立管理，所有层成功后才提交。文档切换清空对应样本历史。GDN递推状态不支持直接裁剪已确认历史；需要回退时恢复完整状态快照或重算前缀。SWA使用绝对位置且物理保留至多W个KV。GDN训练保留已验证GPU事件布局和chunk kernel；仅允许内核所需的小型边界元信息，常规索引不回传CPU。

proposal.loss_decay_gamma、旧Selector CE系数/warmup/ramp参数、candidate_only、空GDN layers以及旧backbone类型均不接受。checkpoint续训还须满足当前loss contract/version和架构形状；本轮不提供旧权重转换或兼容加载。以上规则只决定解析与执行，不会启动训练、测评或恢复队列。

## 完整示例与非法示例

以下文件均是可加载的完整JSON：

### composable_mha

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "mha",
          "access": "dense"
        },
        {
          "attention": "mha",
          "access": "dense"
        },
        {
          "attention": "mha",
          "access": "dense"
        },
        {
          "attention": "mha",
          "access": "dense"
        },
        {
          "attention": "mha",
          "access": "dense"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {
      "dense": {}
    },
    "attention_configs": {
      "mha": {}
    }
  }
}
```

### composable_gqa

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {
      "dense": {}
    },
    "attention_configs": {
      "gqa": {}
    }
  }
}
```

### composable_mla

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "mla",
          "access": "dense"
        },
        {
          "attention": "mla",
          "access": "dense"
        },
        {
          "attention": "mla",
          "access": "dense"
        },
        {
          "attention": "mla",
          "access": "dense"
        },
        {
          "attention": "mla",
          "access": "dense"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {},
    "attention_configs": {
      "mla": {
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "q_lora_rank": null,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      }
    }
  }
}
```

### composable_gdn

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "gdn",
          "access": "linear"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {
      "linear": {}
    },
    "attention_configs": {
      "gdn": {}
    }
  }
}
```

### composable_mixed

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "mla",
          "access": "sliding_window"
        },
        {
          "attention": "gdn",
          "access": "linear"
        },
        {
          "attention": "mha",
          "access": "dense"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {
      "linear": {},
      "dense": {},
      "sliding_window": {}
    },
    "attention_configs": {
      "gdn": {},
      "gqa": {},
      "mla": {},
      "mha": {}
    }
  }
}
```

### composable_dsa

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash"
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "gqa",
          "access": "dsa"
        },
        {
          "attention": "gqa",
          "access": "dsa"
        },
        {
          "attention": "gqa",
          "access": "dsa"
        },
        {
          "attention": "gqa",
          "access": "dsa"
        },
        {
          "attention": "gqa",
          "access": "dsa"
        }
      ]
    },
    "proposal": {
      "type": "parallel"
    },
    "objective": {
      "type": "hard_ce",
      "loss_decay_gamma": 4.0
    },
    "access_configs": {
      "dsa": {}
    },
    "attention_configs": {
      "gqa": {}
    }
  }
}
```

### composable_selector_final_ce_tv

```json
{
  "architectures": [
    "ComposableDraftModel"
  ],
  "block_size": 8,
  "schema_version": "composable_v3",
  "composable_config": {
    "context": {
      "type": "dflash",
      "target_layer_ids": [
        1,
        9,
        17,
        25,
        33
      ]
    },
    "backbone": {
      "type": "layer_stack",
      "layers": [
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        },
        {
          "attention": "gqa",
          "access": "dense"
        }
      ]
    },
    "proposal": {
      "type": "selector",
      "selector_mode": "full_vocab_residual",
      "rank": 256,
      "top_k": 16,
      "stop_gradient": false
    },
    "objective": {
      "type": "selector_losses",
      "loss_decay_gamma": 4.0,
      "base": {
        "type": "hard_ce",
        "weight": 0.1
      },
      "final": {
        "type": "weighted_sum",
        "components": [
          {
            "type": "hard_ce",
            "weight": 0.1
          },
          {
            "type": "l1",
            "weight": 0.9
          }
        ],
        "weight": 1.0
      }
    },
    "access_configs": {},
    "attention_configs": {
      "gqa": {}
    }
  }
}
```

### 典型错误

- {"attention":"gdn","access":"dense"}：GDN只允许linear。
- {"attention":"gqa","access":"dense","num_key_value_heads":8}：逐层参数非法，应写attention_configs.gqa。
- MHA Q=32/KV=8：使用gqa；mha要求等头数。
- MLA Q=16/KV=8：当前展开式MLA不支持不等头数。
- GDN Q/K=16、V=24：V必须是Q/K整数倍。
- query_conv_mode=confirmed_history、query_conv_kernel_size=2：历史接续模式要求conv4。
- 一个dense层与一个dsa层：DSA必须整塔启用。
- DSA＋flex_attention/eager/selector：当前不支持，明确报错。
- parallel下填rank/top_k/stop_gradient：字段不生效且v3禁止，不再静默忽略。
- 在Selector final或weighted_sum组件里填gamma：只在最外层objective配置。
- 字段名拼错、null代替必填对象、字符串数字或boolean冒充整数：类型/字段校验失败。

## 新checkpoint冷加载

新v3模型保存共享配置以及独立execution_config；from_pretrained时可显式传execution_config覆盖执行方式，例如CPU核对使用{"attention_backend":"sdpa","gdn_backend":"reference"}。此覆盖不改变模型投影参数；DSA仍限制SDPA。旧v1/v2不作加载兼容。

## 运行示例

仅展示参数形式，不自动执行：python -m scripts.composable.train --draft-config configs/composable_mixed.json --attention-backend sdpa --gdn-backend fla …。数据、batch、optimizer、保存和分布式参数仍按训练文档配置。DSA必须使用sdpa。训练当前保持停止。
