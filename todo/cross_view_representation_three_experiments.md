# 三个关键实验验证点：跨视角 BEV 表征论文证据链

## 总体目标

当前工作的核心不是证明"卫星图像提升 NVS"，而是验证：

> 在极端天顶视角（satellite）与街面视角（ground）的跨视角条件下，卫星观测是否能够恢复一个由地面观测定义的世界索引场景表征，并分析该表征中的可迁移信息边界。

核心假设：

\[ I\_{satellite}+I\_{sparse-ground} `\rightarrow `{=tex}`\hat{Z}`{=tex}
`\approx `{=tex}Z\^\*\_{ground} \]

其中：

-   (Z\^\*)：由 dense ground observation 定义的场景隐空间；
-   (`\hat{Z}`{=tex})：由卫星和稀疏地面观测恢复的表征。

需要验证的问题：

1.  恢复的信息是否确实包含垂直几何结构？
2.  卫星贡献是否来源于 elevation-induced viewpoint gap？
3.  表征层面的恢复为什么不能直接等价于渲染性能提升？

------------------------------------------------------------------------

# Experiment 1：Height / Geometry Probe

## 科学问题

卫星恢复的 latent 信息到底是什么？

如果核心假设成立：

> Satellite provides information that helps recover vertical scene
> structure encoded in the ground-defined latent.

那么：

\[ `\hat{Z}`{=tex}\_{sat} \]

应该包含比 XY prior 更强的高度相关信息。

------------------------------------------------------------------------

## 实验设计

冻结所有表征：

-   (Z\^\*)
-   (`\hat{Z}`{=tex}\_{sat})
-   (`\hat{Z}`{=tex}\_{XY})
-   (`\hat{Z}`{=tex}\_{gnd})

训练轻量 probe：

\[ f_h(Z(x,y)) `\rightarrow `{=tex}h(x,y) \]

预测：

-   LiDAR height
-   DEM elevation
-   relative height

------------------------------------------------------------------------

## 对比组

### Oracle latent

Dense ground:

\[ Z\^\* \]

衡量理想上限。

### Satellite recovered latent

\[ `\hat{Z}`{=tex}\_{sat} \]

验证卫星恢复能力。

### XY prior

\[ `\hat{Z}`{=tex}\_{XY} \]

排除单纯空间位置先验。

### Sparse ground only

\[ `\hat{Z}`{=tex}\_{gnd} \]

验证卫星额外贡献。

------------------------------------------------------------------------

## 关键指标

-   Height RMSE
-   MAE
-   Pearson correlation
-   Rank correlation

------------------------------------------------------------------------

## 成功标准

如果：

\[ `\hat{Z}`{=tex}*{sat} \> `\hat{Z}`{=tex}*{XY} \]

特别是在稀疏观测情况下：

说明：

> Satellite information is encoded into the latent as recoverable
> geometric structure.

------------------------------------------------------------------------

# Experiment 2：Vertical Complexity Stratification

## 科学问题

卫星优势是否真的来自 vertical ambiguity？

如果跨视角 gap
的主要来源是高度维度，那么卫星贡献应该随着场景垂直复杂度增加而增强。

------------------------------------------------------------------------

## 实验设计

利用 LiDAR / height map 将区域划分：

## Flat regions

例如：

-   道路
-   空地
-   停车区域

特点：

\[ `\sigma`{=tex}(h)`\approx0`{=tex} \]

理论预测：

\[ Gain\_{sat}`\approx0`{=tex} \]

------------------------------------------------------------------------

## Medium vertical regions

例如：

-   低层建筑
-   普通街区

------------------------------------------------------------------------

## High vertical complexity regions

例如：

-   多层建筑
-   建筑边界
-   高度变化明显区域

理论预测：

\[ Gain\_{sat}`\uparrow`{=tex} \]

------------------------------------------------------------------------

## 评价方式

定义：

\[ Gain = M(`\hat{Z}`{=tex}*{sat})-M(`\hat{Z}`{=tex}*{XY}) \]

分别计算：

-   latent similarity gain
-   height probe gain
-   rendering gain

------------------------------------------------------------------------

## 成功标准

如果：

\[ Gain\_{high-height} \> Gain\_{flat} \]

说明：

> Satellite contribution is associated with resolving elevation-related
> ambiguity rather than providing generic semantic cues.

------------------------------------------------------------------------

# Experiment 3：Decoder-only Adaptation

## 科学问题

为什么 latent 恢复提升没有直接转化为 rendering 提升？

当前观察：

\[ d(`\hat{Z}`{=tex}\_{sat},Z\^\*)`\downarrow`{=tex} \]

但是：

\[ D(`\hat{Z}`{=tex}\_{sat}) \]

未必提升。

需要判断：

-   latent 没有有效信息；
-   还是 frozen decoder 无法读取新的 latent 分布。

------------------------------------------------------------------------

## 实验设计

保持 Stage B encoder 固定：

\[ `\hat{Z}`{=tex}=F(I\_{sat},I\_{ground}) \]

只调整 decoder。

------------------------------------------------------------------------

## 四组实验

  Latent                     Decoder           目的
  -------------------------- ----------------- --------------------------
  (Z\^\*)                    Frozen decoder    Oracle upper bound
  (`\hat{Z}`{=tex}\_{sat})   Frozen decoder    当前结果
  (`\hat{Z}`{=tex}\_{sat})   Adapted decoder   验证 latent 信息是否存在
  (`\hat{Z}`{=tex}\_{XY})    Adapted decoder   排除 decoder 自身提升

------------------------------------------------------------------------

## 关键分析

定义 decoder realization gap：

\[ `\eta `{=tex}= `\frac{RenderingGain}{LatentGain}`{=tex} \]

比较：

\[ `\eta`{=tex}\_{sat} \]

和：

\[ `\eta`{=tex}\_{XY} \]

------------------------------------------------------------------------

## 成功标准

如果：

\[ AdaptedDecoder(`\hat{Z}`{=tex}*{sat}) \>
AdaptedDecoder(`\hat{Z}`{=tex}*{XY}) \]

而 frozen decoder 中优势消失：

说明：

> Cross-view information transfer happens at representation level before
> being realized by a downstream decoder.

------------------------------------------------------------------------

# 最终论文证据链

三个实验共同证明：

## 1. Recoverability

卫星可以恢复 ground-defined latent：

\[ I\_{sat} `\rightarrow`{=tex} `\hat{Z}`{=tex} `\approx `{=tex}Z\^\* \]

------------------------------------------------------------------------

## 2. Information boundary

恢复的信息主要对应：

\[ `\text{shared geometry}`{=tex} \]

尤其：

\[ `\text{vertical/elevation structure}`{=tex} \]

而非 view-dependent appearance。

------------------------------------------------------------------------

## 3. Representation-readout separation

latent 中存在的信息与 decoder 是否能够利用它是两个独立问题：

\[ Representation Recovery `\neq`{=tex} Rendering Realization \]

------------------------------------------------------------------------

# 推荐实验优先级

1.  Height / Geometry Probe（最高优先级）
2.  Vertical Complexity Stratification
3.  Decoder-only Adaptation

这三个实验完成后，论文定位可以从：

"satellite-ground fusion"

提升为：

"extreme cross-view scene representation recoverability".
