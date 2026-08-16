# Stage2 `anchor_view` 方案文档

## 1. 背景与目标

当前 `stage1 / baseline` 已经具备单视角独立生成能力。`stage2` 的目的不是追求“所有视角整体外观都一致”，而是引入一个训练时可见、质量更可靠的固定监督视角 `anchor_view`，帮助模型在生成另一个透视视角时，尽可能保持与 `anchor_view` 重叠区域内的静态场景身份一致。

更准确地说，任务目标是：

- 仍然以车辆为中心生成目标透视视角；
- 允许透视变化、可见面变化、遮挡变化；
- 但在与 `anchor_view` 重叠的区域内，原来的 building 仍应是同一个 building；
- 避免生成结果在重叠区域把静态结构“漂移”成完全无关的内容。

因此，`stage2` 的核心不是“全图一致性”，而是：

- `anchor_view` 条件下的目标视角生成；
- overlap-aware 的静态结构身份保持；
- 在 baseline 独立生成能力之上做参考视角纠偏。

## 2. 两类 `anchor` 的严格区分

这里必须区分两类完全不同的概念，避免后续设计和实现混淆。

### 2.1 `hybrid` 模式中的 anchor

这是 `models/stage2/pose_aware_anchor_query.py` 中的内部 latent query / global anchor：

- 作用：聚合全局上下文；
- 本质：模型内部的 learnable memory / query；
- 输入来源：当前样本的 direct memory 和 pose；
- 不对应真实监督视角；
- 不应与 `stage2` 的 `anchor_view` 混为一谈。

建议术语：

- `global anchors`
- `anchor queries`
- `scene anchors`

### 2.2 `stage2` 中的 `anchor_view`

这是训练时固定五视角中的一个真实监督视角：

- 作用：作为 reference view；
- 本质：高可信度、真实可见的条件输入；
- 作用对象：帮助生成另一个 target view；
- 强调的是 reference-conditioned generation，而不是内部 query 机制。

建议术语统一为：

- `anchor_view`
- `reference_view`

## 3. 任务定义

`stage2` 的任务形式应当定义为：

- 输入：`target view` 的常规条件 + `anchor_view` + `relative pose(anchor->target)`；
- 输出：`target view`；
- 约束重点：`anchor_view` 与 `target view` 的重叠区域；
- 监督重点：目标视角生成正确，同时重叠区域静态结构不漂移。

这意味着它是一个**非对称、有向**任务：

- `anchor_view` 是参考条件；
- `target view` 是主要生成对象；
- 不是双向对等一致性训练。

## 4. 非目标

以下内容不应被视为 `stage2` 的主目标：

- 让两个视角整张图逐像素相同；
- 让所有区域都保持一致；
- 强迫 target view 拟合 anchor_view 看不到的新区域；
- 把遮挡变化和透视变化视为错误；
- 将其当作“全局多视角一致性”的对称问题来做。

## 5. 现状诊断

当前仓库中已有一套 `anchor_view` 训练雏形，但存在以下问题。

### 5.1 设计目标是合理的

现有思路抓住了一个核心事实：

- 训练时固定监督视角是真实的、稳定的；
- 比 zero-shot 生成出的其它视角更可靠；
- 因而非常适合作为 target 视角生成时的 reference signal。

这个出发点是成立的，应当保留。

### 5.2 当前实现过重

目前的实现将多件事情同时压给模型学习：

- baseline 自身 CE；
- `anchor_view` 条件下的 target CE；
- BEV 投影一致性；
- 特征级一致性；
- warp 后图像条件的解释与利用。

这些目标同时优化会显著增加训练难度，并放大几何误差、遮挡误差与颜色误差。

### 5.3 当前 conditioner 表征偏弱且偏噪声

当前 `AnchorViewConditioner` 主要基于：

- warp 后的 `anchor_view` 图像；
- `anchor->target` 相对位姿。

这意味着模型需要从一个已经带有透视变化、遮挡、插值误差、无效区域的 warped RGB 条件中，再恢复可用于 target token 生成的稳定结构信息，训练难度很高。

### 5.4 约束重心不够聚焦于 overlap region

你真正关心的是：

- 重叠区域中的静态结构身份不漂移。

但当前一致性损失里，`BEV projection RGB loss` 更偏向全局现象一致，不能精准表达“同一栋楼还是这栋楼”的需求。它适合作为弱正则，不适合作为主驱动。

## 6. 设计原则

新的 `stage2` 方案建议遵循以下原则。

### 6.1 baseline first

- baseline 先学会单视角独立生成；
- `stage2` 不替代 baseline，而是在其基础上做参考视角纠偏。

### 6.2 target-centered

- 所有主损失围绕 `target view`；
- `anchor_view` 主要提供条件，不是被对等优化的对象。

### 6.3 overlap-aware

- 只在 `anchor_view` 和 `target view` 的有效重叠区域施加一致性；
- mask 外区域不强拉。

### 6.4 identity over appearance

- 优先保持静态结构身份与几何语义；
- 不追求颜色/像素级严格对齐。

### 6.5 incremental training

- 先训最稳的条件生成能力；
- 再逐步加入更难的一致性项；
- 避免一次性打开全部 loss。

## 7. 推荐的总体方案

## 7.1 总体形式

建议把 `stage2` 定义为：

- `reference-view-conditioned target generation`

即：

1. baseline 保持现有 target 视角生成流程；
2. 额外输入一个真实的 `anchor_view`；
3. 使用 `anchor->target` 相对位姿对其进行几何调制；
4. 将其作为 target memory 的辅助条件；
5. 主目标仍是 target 视角正确生成；
6. 辅助目标是重叠区域的结构身份一致。

### 7.2 视角关系

固定五视角仍采用当前有向链式关系：

```python
ANCHOR_VIEW_MAP = {
    "front": None,
    "left_to_front_30": "front",
    "right_to_front_30": "front",
    "left_axis": "left_to_front_30",
    "right_axis": "right_to_front_30",
}
```

对应关系：

- `front -> left_to_front_30`
- `front -> right_to_front_30`
- `left_to_front_30 -> left_axis`
- `right_to_front_30 -> right_axis`

这些 pair 应被看作：

- `anchor_view -> target_view`

而不是无向 pair。

## 8. 模型接入建议

## 8.1 不建议的接法

不建议继续把 `stage2` 理解成“再加一条重的并列分支”，例如：

- 整体再跑一遍完整生成；
- 把参考视角和目标视角放在对等位置；
- 同时施加强 BEV / RGB / feature 多重一致性。

这会让优化目标过多，训练不稳定。

## 8.2 推荐的接法

推荐把 `anchor_view` 看作 target 的辅助条件，并以残差/门控形式接入 target memory。

抽象流程如下：

1. target 分支照常构建基础条件：
   - BEV sampled feature
   - coordinate feature
   - pose token / direct memory
2. 从 `anchor_view` 提取参考条件：
   - 几何 warp 后的可见提示
   - 相对位姿编码
   - 可选的结构特征编码
3. 通过一个轻量 adapter 得到 `anchor residual`
4. 使用相对位姿驱动的 gate 控制注入强度
5. 将 `anchor residual` 加到 target memory 上
6. 再送入原有的 `hybrid` 主干预测 target tokens

推荐形式：

```text
target_memory = target_base_memory
anchor_delta  = AnchorViewAdapter(anchor_view, rel_pose)
gate          = sigmoid(PoseMLP(rel_pose))
fused_memory  = target_memory + gate * anchor_delta
```

这里的关键思想是：

- target 自身生成能力是主干；
- `anchor_view` 只作为修正项；
- 避免让参考条件“盖过” baseline 主路径。

### 8.3 与 `hybrid` 中 global anchors 的关系

`stage2` 的 `anchor_view` 条件应作为 **target memory 的外部参考修正** 接入；
而 `hybrid` 中的 global anchors 继续负责模型内部全局上下文建模。

两者关系应当是：

- `anchor_view`：提供 reference signal；
- `global anchors`：提供 latent global reasoning。

二者互补，不冲突。

## 9. 条件表征建议

### 9.1 当前最容易不稳定的点

如果直接使用 warp 后 RGB 作为主要参考条件，问题是：

- 插值噪声大；
- 遮挡和不可见区域多；
- 强透视变化时信息不稳定；
- 模型要自己从图像细节中重新恢复结构身份。

### 9.2 更稳的条件形式

推荐优先级如下：

1. `relative pose(anchor->target)`
2. overlap / valid mask
3. 几何对齐后的结构特征
4. warp 后 RGB 仅作为辅助线索

因此建议 `AnchorViewConditioner / Adapter` 最终输出的不是“尽量复原 target 图像”，而是：

- 一个带 mask 的结构性 residual；
- 一个用于 memory 修正的参考特征；
- 而不是主导生成内容的第二主干。

## 10. 损失设计建议

### 10.1 主损失：target CE

最核心的损失应当是：

- 给定 `anchor_view + relative pose + target condition`，预测 `target_view` 的 token CE。

即：

- baseline CE 是基础；
- conditional target CE 是 `stage2` 的主驱动。

### 10.2 核心辅助损失：overlap-masked feature consistency

最值得保留的一致性项是特征级一致性，但必须限定在 overlap region 内。

目标是：

- 在 anchor 与 target 都能对齐到的有效区域内；
- 约束稳定静态结构的特征保持身份一致；
- 不要求全图一致。

实现要点：

- 使用已有 warp / coords / valid mask 构造 overlap mask；
- 仅对 overlap 区域取样本点；
- 在这些点上做 feature matching 或 contrastive consistency。

### 10.3 弱辅助损失：BEV / RGB 一致性正则

`BEV projection RGB loss` 可以保留，但建议：

- 只作为很弱的 regularizer；
- 不作为主损失；
- 在早期阶段甚至可完全关闭。

原因是它过于依赖平面假设和像素对齐，对动态物体与遮挡非常敏感。

## 11. 训练策略建议

推荐采用三阶段策略。

### 阶段 A：baseline 预训练

目标：

- 先把单视角独立生成训稳；
- 不引入 `anchor_view` 条件。

建议：

- 保持现有 baseline 训练；
- 以 target CE 为核心。

### 阶段 B：仅引入 `anchor_view` 条件生成

目标：

- 先验证 `anchor_view` 作为 reference 是否真的能降低 target 预测难度；
- 先不引入强一致性约束。

建议：

- 从 baseline checkpoint 恢复；
- 接入 `anchor_view` adapter；
- 主训 conditional target CE；
- 一开始可冻结部分 backbone，只训练 adapter / gate / 少量接入层。

### 阶段 C：逐步加入 overlap consistency

目标：

- 在生成质量不掉的前提下，提升重叠区域身份保持。

建议顺序：

1. 先加 overlap-masked feature consistency；
2. 再视情况加非常小权重的 BEV / RGB 正则。

## 12. 超参数建议

`stage2` 不建议沿用 baseline 的激进设定，应更偏微调范式。

建议范围：

- 学习率：`1e-5 ~ 2e-5`
- `anchor_view` 相关分支可适当更高 LR
- consistency 权重：从很小值启动

推荐启动策略：

- `target_conditional_ce`: 1.0
- `overlap_feature_consistency`: 0.02 ~ 0.05
- `bev/rgb regularizer`: 0 或极小值

训练原则：

- 先证明 `anchor_view` 条件本身有效；
- 再逐步增加一致性约束。

## 13. 推荐的实现重构方向

### 13.1 保留的部分

- 固定五视角数据组织；
- anchor-target 有向 pair 逻辑；
- baseline 主干；
- overlap / valid mask 的几何计算基础；
- feature-level consistency 的大方向。

### 13.2 需要弱化或延后使用的部分

- 强 BEV projection RGB loss；
- 过重的并列双分支训练；
- 过早同时打开全部损失项。

### 13.3 建议新增或重构的模块

- `AnchorViewAdapter`
  - 输入：`anchor_view`, `relative pose`, `valid mask`
  - 输出：target memory residual
- `PoseGate`
  - 输入：`relative pose`
  - 输出：residual 注入强度
- `OverlapMaskedConsistency`
  - 输入：anchor / target 对齐后的特征与 overlap mask
  - 输出：仅作用于重叠区域的一致性损失

## 14. 代码改造清单

建议按“先消歧义、再轻量接入、最后加一致性”的顺序推进，避免一开始同时改太多模块。

### 14.1 命名与入口清理

- 训练脚本统一使用：
  - `world3d/train/train_anchor_view_stage2.py`
- 训练类统一使用：
  - `ArAnchorViewStage2Trainer`
- 明确默认配置入口使用：
  - `configs/ar_anchor_view.yaml`
- 在文档和日志中统一使用：
  - `anchor_view`
  - `reference_view`
- 避免使用过于宽泛的 `anchor` 来指代 `anchor_view`。

### 14.2 训练器改造

目标文件：

- `world3d/train/train_anchor_view_stage2.py`

建议改动：

- 保留当前五视角分组与 pair 组织逻辑；
- 明确训练主任务是 `anchor_view -> target_view` 的有向条件生成；
- 将 loss 拆成更清晰的三部分：
  - baseline CE
  - target conditional CE
  - overlap-only consistency
- 在日志中单独输出：
  - `baseline_ce`
  - `target_cond_ce`
  - `overlap_consistency`
  - `total_loss`
- 保留回退路径：
  - `use_anchor_view_training=False` 时仍可退回 baseline 行为。

### 14.3 `predictor` 接入点改造

目标文件：

- `models/stage2/simplified_token_predictor.py`

建议改动：

- 在 `hybrid` 路径中增加显式的 `anchor_view` 条件接入点；
- 接入方式优先使用 residual / gate，而不是重 memory 并列拼接；
- 让 `anchor_view` 条件作用于 target memory，而不是替代主干；
- 保持与 `hybrid` 内部 `global anchors` 解耦。

建议新增输入语义：

- `anchor_view_memory`
- `anchor_view_mask`
- `anchor_view_pose_delta`

### 14.4 conditioner / adapter 改造

目标文件：

- `world3d/train/anchor_view_conditioning.py`

建议改动：

- 将当前 `AnchorViewConditioner` 逐步收敛为更明确的 `AnchorViewAdapter`；
- 输出目标从“生成一张 warp 后图像特征”调整为：
  - target memory residual
  - valid / overlap mask
  - 可选的 pose gate 特征
- 降低对 warp RGB 的依赖，保留其作为辅助线索而不是唯一主表征。

### 14.5 一致性损失改造

目标文件：

- `world3d/train/anchor_view_consistency_loss.py`

建议改动：

- 将一致性项从“组合式大损失”收敛成 overlap-aware 版本；
- 明确区分：
  - overlap-masked feature consistency
  - optional weak BEV/RGB regularization
- 默认先启用 feature consistency；
- 默认关闭或极弱化 `BEV projection RGB loss`。

### 14.6 配置层改造

目标文件：

- `configs/ar_anchor_view.yaml`
- `world3d/config.py`

建议改动：

- 增加更明确的 stage2 配置项，例如：
  - `anchor_view_mode: residual`
  - `anchor_view_use_overlap_consistency: true`
  - `anchor_view_use_bev_regularizer: false`
  - `anchor_view_gate: true`
- 将 stage2 学习率与 baseline 学习率解耦；
- 保证 batch 设定与五视角 grouped sampler 兼容。

### 14.7 推荐的最小落地顺序

第一批最小改动：

- 只做命名清理；
- 在 `hybrid` 中打通 `anchor_view` 条件接入口；
- 只训练 conditional target CE。

第二批改动：

- 加 overlap-masked feature consistency；
- 增加更细日志与可视化。

第三批改动：

- 再评估是否保留极弱的 BEV/RGB regularizer。

## 15. 验证指标建议

`stage2` 的验证不应只看总 loss，应重点看以下现象。

### 14.1 定性观察

- 重叠区域建筑是否保持同一 identity；
- target 视角是否仍有合理透视变化；
- 新暴露区域是否自然，不被强拉成 anchor 外观；
- 遮挡边界是否稳定。

### 14.2 定量观察

- target CE 是否下降；
- overlap 区域 feature consistency 是否改善；
- baseline 生成质量是否保持；
- 是否出现模式塌陷或过度复制 `anchor_view` 的现象。

## 16. 保持现状 vs 优化重构

### 保持现状

优点：

- 不需要额外开发；
- 可继续复用已有代码。

缺点：

- 训练目标过重；
- 信号不够聚焦在 overlap identity；
- 容易难训、收敛慢、效果不稳定；
- 不完全贴合当前澄清后的任务定义。

### 优化重构

优点：

- 更贴合真实目标；
- 更容易分阶段验证；
- 更能回答“同一 building 是否被保住”这个核心问题；
- 更适合作为 baseline 上的可控增强模块。

缺点：

- 需要补一轮设计和接线；
- 需要重新安排训练策略与损失权重。

### 建议结论

建议选择：

- **保留 `stage2` 目标**；
- **优化实现方案**；
- **不建议原样保持当前实现继续硬训**。

## 17. 推荐实施顺序

建议按以下顺序推进。

### 第一步

- 明确术语：`global anchors` 与 `anchor_view` 分离；
- 明确 `stage2` 是 target-centered 的 reference-conditioned generation。

### 第二步

- 在现有训练器中保留 pair 组织；
- 把 `anchor_view` 接成 target memory 的轻量 residual condition；
- 先只训 conditional target CE。

### 第三步

- 加入 overlap-masked feature consistency；
- 观察重叠区域 identity 是否明显更稳。

### 第四步

- 仅在前面两步有效后，再评估是否需要极弱的 BEV / RGB 正则。

## 18. 最终结论

这条 `stage2` 方向本身是值得做的，因为它抓住了一个很自然的训练优势：

- 使用训练时真实可见、可靠的固定视角 `anchor_view`，帮助 target 视角生成更稳定地保持静态场景身份。

但它应当被定义为：

- **以车辆为中心的目标视角生成**
- **参考视角条件输入**
- **仅对重叠区域做身份保持约束**

而不是：

- 全局对称多视角一致性问题。

因此建议：

- 保留目标；
- 重构实现；
- 先做轻量条件生成，再逐步加入 overlap-aware consistency。
