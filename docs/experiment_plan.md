# 实验方案 — Persistent Georeferenced World State（KITTI-360 v1）

> 当前唯一对照。骨干：冻结 DINOv2-ViT-B/14（卫星）、冻结 VGGT（车端）。
> 只训两个 write head（satellite/XY）、measurement encoder 与 updater。  
> 不覆盖 UAV、hole、删帧、RGB 主指标、VLA、从零训卫星 ViT。

---

## 1. 要证明什么

\[
Z_0 = W_s(I_{\mathrm{sat}}),\qquad
G_t = \mathrm{VGGT}_{\mathrm{frozen}}(C_t),\qquad
Z_t = U(Z_{t-1}, G_t, M_t)
\]

KITTI-360 没有 UAV，只能评 **车稍后会开到、现在还没开到** 的路段（future-route 上事后有静态 LiDAR 的格子）。不评轨迹外完整 3D。

四件事：

1. 卫星能否在到达前给出 ahead 布局；
2. 该布局是否优于 coordinate-only / 错位卫星；
3. ground chunk 到达后是否把精确几何写入 \(Z\)；
4. 后续更新是否不破坏已写入区域。

失败解读：1+2 不行 → 没有卫星增益；3 不行 → updater 没写入；4 不行 → 只是一次性 fusion，不必叫状态。

---

## 2. 训什么、冻什么

| 模块 | 角色 | 训/冻 |
|---|---|---|
| VGGT | 把 12 m chunk（剔除查询帧）变成 \(G_t, M_t\) 的 depth/conf | 冻 |
| DINOv2-ViT-B/14 | 从 south-up 卫星 BEV 抽 patch 特征（resize 到 224，ImageNet 归一化） | 冻 |
| `bilinear_splat` | 世界 XY → BEV 格子 | 不学 |
| Stage A readers（height / density / depth） | 定义 \(Z\) 可读几何 | 先训后冻；Stage B 中 depth **不反传** |
| `GroundMeasurementEncoder` | VGGT depth/conf + RGB + 标定位姿 → BEV 测量 \(G_t, M_t\) | **训**（单实例，两链共享同一 \(G_t\)；两链损失共同更新） |
| 卫星 write head | DINO 特征 (+ 固定 XY 编码) \(\to Z_0\) | **训**（小投影/卷积，不训 DINO） |
| XY write head | 零特征 + 同一固定 XY 编码走同一 write-head 结构 | **训**（与卫星 write head 容量对齐） |
| updater \(U\) | \(Z_{t-1},G_t,M_t\to Z_t\) | **训**（单实例，两链共享） |

训练形态为 **shared_assimilation**：同一 batch 内 \(Z_t^{sat}=U(Z_{t-1}^{sat},G_t)\) 与
\(Z_t^{xy}=U(Z_{t-1}^{xy},G_t)\) 共享完全相同的 \(G_t\)、\(M_t\)、updater 与 readers，
初始化是唯一被操纵变量（E2 的"同一 updater"由结构保证）。\(G_t\) 每 chunk 只计算一次。

VGGT 越强越好：它越强，越说明卫星的价值在「车还没到」，不在「少几帧」。  
DINO 同理：不从零训卫星 ViT。\(W_s\) 只学「这些 patch 特征怎么写进地理格子」。XY 对照关掉 DINO 特征、只保留同一 write head + 固定 XY 编码，容量对齐的是 write head 而不是整个 ViT。

---

## 3. 数据

| 用途 | 数据 |
|---|---|
| 静态世界真值 | `data_3d_semantics` 累计点云；去掉 car/truck/person/rider/bicycle；没点 = unknown |
| 当前包去掉前景 | `data_3d_bboxes` 从该帧 LiDAR / VGGT splat 剔除动态 |
| 卫星 | 512×512、north-up、0.196 m/px、车辆居中；resample 到 100 m south-up tile |
| 车端 | 12 m 弧长 chunk，固定多帧，只为 VGGT 有运动基线 |
| Split | 有语义 ID 的 train/val 划 scene，tile 地理不重叠。官方 test 无语义标签，不当静态主评测 |

停着的车语义仍是 car，从静态 \(Z\) 中整类丢掉。RGB 不当主监督。

几何约定：south-up、pixel-center、普通双线性均值 splat；`z_datum` = 该 scene 静态点 world-Z 或 LiDAR 光心中位数（全序列固定，禁止逐 chunk quantile）。

---

## 4. 样本单位

一个 **100 m × 100 m scene tile**（落入一张卫星资产）+ 沿轨迹有序 chunks \(C_1,\ldots,C_T\)（最多 8 包）。

第 \(t\) 步：

\[
M_{\mathrm{visited}}^t = \bigcup_{i=1}^{t} M_i,\qquad
M_{\mathrm{ahead}}^t = M_{\mathrm{future\text{-}route}} \setminus M_{\mathrm{visited}}^t
\]

\(M_{\mathrm{future\text{-}route}}\) = 整段轨迹最终扫到的**静态** LiDAR 格子。  
Off-route 只记格子数，不进主表。  
统计按 **scene** 配对，重叠 chunk 不当独立样本。

Writer / updater 的 forward **不得**看到累计真值、未来 chunk、future-route mask。

---

## 5. 四个实验

横轴：已走米数 / 已同化 chunk 数。有效格子 < 256 的 region 不报 headline。

### E1 提前布局（\(t=0\)，\(M_{\mathrm{ahead}}\)）

| 条件 | 输入 |
|---|---|
| satellite-only \(Z_0\) | 配准卫星 |
| matched XY | 同容量坐标先验 |
| random / cross-shift 卫星 | 内容错位 |

指标：height MAE、density。  
通过：aligned 显著优于 XY，且优于 random/shift。

### E2 地理内容，不是坐标模板

同一 updater：`sat-init + ground` vs `XY-init + ground`。

\[
G_{\mathrm{prior}}(t) = E(Z_t^{\mathrm{XY}}; M_{\mathrm{ahead}}^t) - E(Z_t^{\mathrm{sat}}; M_{\mathrm{ahead}}^t)
\]

通过：到达前 \(G_{\mathrm{prior}}>0\)，scene 级 95% paired bootstrap CI 不含 0。到达后差距缩小视为 ground 纠错，不视为卫星失败。

### E3 到达后写入（当前 \(M_t\)）

\[
G_{\mathrm{update}}(t) = E(Z_{t-1}; M_t) - E(Z_t; M_t)
\]

对照：sat+ground、ground-only、one-shot、累计静态云 upper bound。  
通过：median \(G_{\mathrm{update}}(t)>0\)；support 外 latent 变化 \(\le 10^{-7}\)；held-out depth 在该包 query 上也为正。

### E4 不遗忘

\[
F_{1\rightarrow t} = E(Z_t; M_1) - E(Z_1; M_1)
\]

通过：中位相对退化 \(\le 2\%\)，95 分位 \(\le 10\%\)。  
再比 one-shot（相同最终输入一次写完）：最终误差不能差过 2%；persistent 必须任意 \(t\) 可查、遗忘更低。否则不必叫状态。

---

## 6. 训练

**Stage A（一次，然后冻住）**  
用静态累计语义点云拟合 world encoder + height/density/depth readers。只为定义可读的 \(Z\)。

**Stage B（主训练）**  
每个 scene：冻 DINO 抽卫星特征，只训 write head 得到 \(Z_0\)；再按序 \(Z_t=U(Z_{t-1},G_t,M_t)\)。  
前缀 no-grad replay；反传最近 \(\le 4\) 步。  
\(G_t\) 来自冻结 VGGT→splat，对 updater detach。DINO 特征对 write head 也可 detach（骨干无梯度）。

Loss：

- 当前 \(M_t\)：冻结 height + density（局部写入）
- 历史 visited：retention
- \(Z_0\)：对静态累计云的 masked distill + height/density（初始化）
- 无 depth loss、无 RGB 主损失、不把 \(Z_t\) 回归成整张未来 \(Z_{\mathrm{world}}\)

分支：不再有独立训练的分支。训练只跑一次 **shared_assimilation**
（同 batch 双链：sat-init 与 XY-init 共享测量流/updater/readers，loss 为两链之和）；
`ground_only`（空初始化）与 `one_shot`（测量并集一次写入）只是评测期 `--control`
变体。Random/shift 同样只评不训。

---

## 7. 执行顺序

1. 接入 `data_3d_semantics` + bbox 动态过滤，重建 world targets。  
2. VGGT 按 chunk 出 \(G_t\)（冻），接到 updater。  
3. 单 scene overfit：E1–E4 方向对、support 外不变。  
4. Stop/go：E1 aligned>XY 且 >shift；median \(G_{\mathrm{update}}>0\)；\(F\) 过门控。失败则不扩规模。  
5. 通过后再 32 scene × 3 seed，scene 级 bootstrap。

---

## 8. 不做

- chunk 内删帧 / spatial hole  
- 轨迹外完整 3D、屋顶/背面无真值区域  
- 卫星恢复车、人、立面高频纹理  
- 训新的几何基础模型或视频 world model  
- RGB 当主几何证据  
- VLA / planning（最多作为后续下游，不进 v1）
