# 持久、地理对齐、可更新的三维世界状态

> 研究讨论稿：从一次性卫星—车端融合，转向异步空间证据的持续状态估计

## 0. 当前结论

原先的任务设定是：在同一个局部场景中，从一个高重叠的车端多帧 chunk 中删除若干帧，再检验卫星图能否恢复由此损失的表征能力。

这个设定可以实现，但不适合作为论文主问题。根本原因是：

- chunk 内相邻帧高度冗余；
- 少一帧不一定减少真实三维空间中的观测支持；
- 当前 `Ns=1` 已经包含一个完整的多相机 rig，而不是单幅 RGB；
- `Ns=2` 还可能保留很大的车辆运动基线；
- 因此“删帧”是人为施加的输入扰动，不一定形成卫星不可替代的信息缺口。

新的推荐主线是：

> **不同高度、不同时间到达的空间观测，能否被持续吸收到一个地理坐标固定、可更新、可查询的三维世界状态中？**

英文工作定义可写为：

> **Persistent Georeferenced World-State Assimilation from Asynchronous Overhead and Ground Observations**

卫星图不再负责“补回少掉的一帧”，而是在车辆到达之前初始化大范围、低频、静态的空间结构；车辆随后以连续 chunk 的形式逐段到达，对同一个世界状态进行局部确认、细化和纠错。

---

## 1. 为什么不再把 chunk 内删帧作为主任务

### 1.1 帧数不是空间证据量

减少输入帧数只是一个数据操作：

\[
\mathcal C_{\mathrm{dense}}
\rightarrow
\mathcal C_{\mathrm{sparse}}
\]

它是否真正减少了空间证据，取决于被删除帧是否带来了新的射线覆盖、视差、遮挡解除或地理区域。对于连续驾驶视频，相邻帧往往观察几乎相同的表面，因此：

\[
|\mathcal C_1| < |\mathcal C_2|
\not\Rightarrow
\Omega(\mathcal C_1) \subsetneq \Omega(\mathcal C_2)
\]

其中 \(\Omega(\mathcal C)\) 表示输入在三维世界中的有效观测支持。

如果多一帧和少一帧没有显著改变 \(\Omega\)，让卫星恢复两者之间的差异就会显得人为。

### 1.2 更强的几何 encoder 会削弱原动机

如果论文主张是“VGGT 在少帧时退化，因此需要卫星”，审稿人可以直接追问：

> 换成更强的单目深度、多视图 geometry foundation model 或更长的 chunk 后，卫星是否仍然必要？

这种动机绑定在特定 encoder 的能力缺陷上，不够稳定。真正无法被更强 encoder 消除的是：

- 车辆尚未到达的区域；
- 道路轨迹永远不经过的区域；
- 车端低视点长期受到遮挡的结构；
- 跨不同时间、不同设备获得的离线空间证据。

### 1.3 短期外推也不是合适的主任务

此前使用 Wan2.1 类视频生成模型的实验已经说明：短期外推主要由历史帧外观和自回归惯性决定，卫星增益较小；而自动驾驶又不一定需要非常长期的 RGB 视频外推。

因此不应把卫星的价值继续绑定在：

> “能否把接下来的几帧生成得更像。”

更合适的问题是：

> “在车辆尚未观察、已经观察和永远不会观察的空间中，系统当前知道什么，这些知识来自哪里，又如何随新观测到达而更新？”

---

## 2. Related work 给出的边界

截至 2026 年，卫星—车端融合已经覆盖了多个直接任务：

| 研究位置 | 代表工作 | 已经回答的问题 | 对本工作的约束 |
|---|---|---|---|
| Satellite-assisted SSC / occupancy | [SGFormer](https://arxiv.org/html/2503.16825v2)、[SA-Occ](https://arxiv.org/abs/2503.16399)、[GeoScene](https://arxiv.org/html/2608.03618v1) | 卫星能否改善车端遮挡和不可见区域的语义 occupancy | 不能只把输出换成另一个 occupancy/height head |
| Aerial-assisted HD mapping | [AerialFusionMapNet](https://arxiv.org/html/2606.24784v1)、[Cross-View Supervision](https://arxiv.org/html/2605.12218v2) | 如何融合俯视与车端 BEV feature，改善在线矢量地图 | 普通 BEV cross-attention、feature alignment 和错位鲁棒性不再足够新 |
| Geographic retrieval for AD | [Spatial Retrieval Augmented Autonomous Driving](https://arxiv.org/html/2512.06865v1) | 地理图像能否帮助检测、地图、occupancy、规划和视频世界模型 | “卫星是额外空间上下文”已经被系统验证 |
| Cross-view 3D reconstruction | [Cross-View Splatter](https://arxiv.org/html/2605.19656v1)、[Cross3R](https://arxiv.org/html/2605.07978v1) | 卫星、UAV、地面图像能否共同预测 3DGS、点云和相机位姿 | “多源图像进入一个统一 3D 输出”不能单独作为贡献 |
| Cross-view neural map | [SNAP](https://arxiv.org/abs/2306.05407) | 能否由地面与俯视图学习用于定位和语义理解的 2D neural map | “共享 neural map”这个名称本身也不足够 |
| Streaming geometry state | [CUT3R](https://arxiv.org/abs/2501.12387) | 地面图像流能否持续更新 persistent geometry state | 仅提出持久 state 也不新，必须体现异步、多高度、地理对齐和可查询性 |
| Geometry latent prediction | [VGGT-World](https://arxiv.org/html/2603.12655v1) | 能否在时间轴上预测冻结 VGGT 中间层的未来 geometry tokens | 本工作不应复制短时间 latent forecasting，而应转向空间状态的建立与更新 |

因此，真正相对空缺的位置不是上述任一单项，而是它们的交叉：

> **用离线 overhead prior 初始化一个世界坐标中的状态，再由连续到达的车端 geometry chunks 对这个状态进行增量更新，并用同一个冻结接口进行多种三维查询。**

这个结论需要谨慎表述为“相对较少被系统研究”，而不是声称从未有人做过 persistent map 或 aerial-ground fusion。

---

## 3. 核心动机

### 3.1 车端 VLA 的短期视觉上下文不是长期空间记忆

当前自动驾驶 VLA 或视频 world model 通常把若干连续帧组织成 chunk。这样可以提供局部运动和几何线索，但它仍然有三个限制：

1. 上下文窗口中的图像高度冗余；
2. 历史图像以时间顺序存储，不是以世界坐标组织；
3. 车辆尚未到达或不经过的区域不会出现在历史 chunk 中。

因此，更长的视频上下文并不等价于一个可在任意世界位置查询的空间记忆。

### 3.2 卫星、UAV 和车辆提供的是不同观测算子

三种数据都可能是 RGB 或彩色几何数据，关键区别不是传统意义上的“模态”，而是观测高度、覆盖范围和几何强度：

| 观测源 | 主要优势 | 主要缺失 |
|---|---|---|
| Satellite | 大范围连续覆盖、道路布局、建筑 footprint、开放区域 | 精确高度、立面、实时状态 |
| UAV | 中等范围、屋顶与高度、斜视三维结构 | 覆盖成本高，当前数据规模有限 |
| Vehicle | 局部精细几何、立面、当前状态 | 走廊式覆盖、低视点遮挡、轨迹外区域缺失 |

这些证据往往不是同步到达的：

- satellite 可能在车辆出发前已经存在；
- UAV 可能来自一次离线测绘；
- vehicle chunks 随车辆行驶在线到达；
- 不同来源还可能存在时间差、错位和内容冲突。

这使问题更接近“世界状态过滤与证据同化”，而不是一次性的多模态 feature fusion。

### 3.3 实际用途

一个持久的 georeferenced state 可以作为 VLA 或规划系统的 allocentric spatial memory：

- 在车辆到达前提供前方道路和静态布局；
- 在到达后由车端观测写入精细几何和当前状态；
- 支持跨 chunk 的长期记忆，而不需要保留全部原始视频；
- 为地图构建、占据查询、深度查询、表面查询和规划提供共同状态；
- 明确区分“直接观测”“外部先验”和“模型补全”。

---

## 4. 问题定义

### 4.1 世界状态

对每个固定地理 tile 定义世界状态：

\[
Z_t \in \mathbb R^{H\times W\times C}
\]

或更完整地定义为可连续查询的三维 field：

\[
Q(Z_t,\mathbf x,d)
\rightarrow y_d,
\qquad
\mathbf x=(x,y,z)
\]

其中 \(d\) 表示 query/readout 类型，例如：

- density / occupancy；
- surface 或 SDF；
- height；
- depth；
- static semantics；
- RGB 或低频 appearance。

### 4.2 Satellite initialization

在车辆观测到达前，由配准 satellite tile 初始化状态：

\[
Z_0=W_s(I_{\mathrm{sat}},P_{\mathrm{sat}})
\]

这里 \(Z_0\) 不应被解释为完整三维真值，而是一个带不确定性的静态空间先验。

### 4.3 Ground chunk update

车辆沿轨迹到达后，第 \(t\) 个固定长度 chunk 为：

\[
\mathcal C_t=\{I_{t,1},\ldots,I_{t,K}\}
\]

ground geometry encoder 生成当前测量：

\[
G_t=E_g(\mathcal C_t,K_t,T_t)
\]

状态更新器将当前测量写入已有状态：

\[
Z_t=U(Z_{t-1},G_t,M_t,C_t^{\mathrm{conf}})
\]

其中：

- \(M_t\)：当前 chunk 在世界坐标中的观测支持；
- \(C_t^{\mathrm{conf}}\)：measurement confidence；
- \(U\)：局部、证据感知的状态更新器。

### 4.4 状态不是一次性 fusion feature

普通融合通常是：

\[
\hat y_t=D(F(I_{\mathrm{sat}},\mathcal C_t))
\]

每次输入都重新计算一个针对特定任务的 feature，然后立即由 head 输出结果。

本工作要求：

\[
Z_{t-1}
\xrightarrow{\mathcal C_t}
Z_t
\xrightarrow{Q_d}
y_d
\]

即状态本身能够跨时间保留，并在没有重新输入全部历史观测时继续被查询和更新。

---

## 5. 世界状态必须满足的契约

仅仅使用 cross-attention 不能证明信息进入了共享世界状态。一个真正的 state 至少应满足以下性质。

### 5.1 Persistence

处理 \(\mathcal C_t\) 时不需要重新输入 \(\mathcal C_1,\ldots,\mathcal C_{t-1}\)。历史证据应已经被压缩到 \(Z_{t-1}\) 中。

### 5.2 Geographical canonicality

不同时间和不同来源的观测必须写入同一个固定世界坐标，而不是相对当前 target camera 的坐标。

这要求：

- 固定 XY tile origin；
- 固定 metric scale；
- 统一 vertical datum；
- query 和 writer 使用完全相同的坐标约定。

### 5.3 Local correction

当车端证据与 satellite prior 冲突时，更新应主要发生在新证据支持区域：

\[
\Delta Z_t(x,y)
\approx 0,
\qquad (x,y)\notin M_t
\]

在 \(M_t\) 内，可靠的 ground geometry 应能够细化或覆盖 satellite 推断。

### 5.4 Retention

更新当前位置时，不应破坏之前已经建立的远处状态。需要显式评估 catastrophic forgetting，而不是只评价最终一步。

### 5.5 Multi-query readability

同一个 \(Z_t\) 应支持多个 frozen readers，而不是每个任务重新训练一个 satellite fusion head。

更强的验证是：

- 用部分 readers 监督 writer；
- 保留至少一个 reader 不参与 writer 训练；
- 检验 held-out frozen reader 是否也从状态更新中受益。

例如，writer 使用 height+density loss 训练，但最终冻结的 target-depth reader 也得到改善。

### 5.6 Provenance and uncertainty

除了 latent content，状态最好保存显式证据字段：

\[
S_t(x,y)
=
\{z_t,\;c_t,\;p_t,\;\tau_t\}
\]

其中：

- \(z_t\)：latent content；
- \(c_t\)：confidence；
- \(p_t\)：provenance，如 satellite / UAV / vehicle / inferred；
- \(\tau_t\)：观测时间或状态版本。

这使系统能够区分“看见了什么”和“相信了什么”。

---

## 6. Chunk 的必要性与正确用法

### 6.1 Chunk 不是研究变量

不再在同一个 chunk 内用 \(N_s=1,2,4\) 制造主要任务。

每个 chunk 固定使用足以让 geometry encoder 正常工作的观测量，例如：

- 固定 4–8 个 rig timestamps；
- 固定约 1–2 秒；
- 固定约 5–15 m 的轨迹长度；
- 所有条件使用相同的相机和帧选择策略。

### 6.2 Chunk 是空间测量包

每个 chunk 的作用是：

- 在内部提供多视图基线和几何一致性；
- 被 geometry encoder 转换为一个局部 measurement；
- 作为一次原子 update 写入世界状态。

### 6.3 真正的自变量是车辆已经覆盖多少空间

实验横轴应使用：

- traversed distance；
- assimilated chunk count；
- fraction of route observed；
- fraction of world cells directly supported by ground rays。

不同 chunk 应沿轨迹逐段扩展新的世界区域，避免用高度重叠的相邻 target records 伪造样本数。

---

## 7. 评价区域

在状态更新到第 \(t\) 步时，将世界 tile 划分为三个核心区域。

### 7.1 Visited region

车辆已经经过，并由至少一个历史 chunk 提供直接地面证据：

\[
M_{\mathrm{visited}}^t
=
\bigcup_{i=1}^{t} M_i
\]

这里主要评价：

- ground update 是否细化几何；
- satellite prior 是否被正确纠正；
- 后续更新是否造成 forgetting。

### 7.2 Ahead / not-yet-visited region

车辆未来会经过，但在当前时刻尚未获得地面证据：

\[
M_{\mathrm{ahead}}^t
=
M_{\mathrm{future-route}}
\setminus M_{\mathrm{visited}}^t
\]

这里检验 satellite initialization 能否提供提前可用的静态空间信息。

它不是未来 RGB 视频外推，而是对静态世界状态的空间查询。

### 7.3 Off-route region

车辆整段轨迹都不会直接覆盖的区域：

\[
M_{\mathrm{offroute}}
=
M_{\mathrm{tile}}
\setminus
M_{\mathrm{all-ground-supported}}
\]

这里是 satellite 最不可被车端冗余帧替代的区域，但评价必须受到真实 ground truth 覆盖的限制。没有 UAV、LiDAR 或可靠地图真值时，不能声称恢复了完整 off-route 3D。

---

## 8. 状态空间不能继续只由街景定义

### 8.1 原 Stage A 的逻辑问题

如果状态 teacher 是：

\[
Z^*=E_g(\mathcal G_{\mathrm{dense}})
\]

而所有 decoder 也只在 dense street observations 上训练，那么状态只保证能够表达街景可定义的内容。

这会带来两个限制：

1. 无法声称状态代表整个场景；
2. 卫星看到的屋顶、完整 footprint 等独有信息，可能根本不存在于 decoder 的可读空间中。

此时 satellite writer 最多只能模仿 ground-defined state，而不能向状态加入 ground teacher 从未定义过的信息。

### 8.2 改为 world-defined state

新的 Stage A 应由外部世界几何定义状态及其 readers：

- 多帧累计 LiDAR；
- KITTI-360 静态语义点云/occupancy；
- UAV 彩色点云；
- 在可能情况下使用 mesh、surface 或 DSM。

形式上：

\[
Y_{\mathrm{world}}
\xrightarrow{\text{state fitting / teacher encoding}}
Z_{\mathrm{world}}
\xrightarrow{Q_1,\ldots,Q_m}
\{\text{density,height,depth,semantics,RGB}\}
\]

随后再学习 source-specific writers：

\[
I_{\mathrm{sat}}\rightarrow W_s\rightarrow Z
\]

\[
\mathcal C_t\rightarrow E_g\rightarrow U\rightarrow Z
\]

这样 ground 与 satellite 都只是世界状态的观测来源，而不是由 ground encoder 决定整个表示论。

### 8.3 数据条件下的诚实边界

在 KITTI-360 上，如果只有沿路 LiDAR：

- 可以评价道路周边、立面和 LiDAR 实际支持的静态几何；
- 可以评价道路布局及 BEV occupancy；
- 不能把未被 LiDAR/UAV 覆盖的屋顶和建筑背面当作完整三维真值。

唯一 UAV 区域可以作为高覆盖、三源对应的系统级验证，但不能单独支撑大规模统计结论。

---

## 9. 推荐模型结构

### 9.1 Canonical state grid / field

保留当前 georeferenced BEV latent 的工程优势，但将其升级为：

- fixed tile origin；
- fixed metric XY resolution；
- unified vertical datum；
- latent content + confidence + source mask；
- continuous \((x,y,z)\) query interface。

### 9.2 Satellite writer

Satellite writer 只负责初始化它有证据支持的低频静态结构：

\[
(Z_0,C_0,P_0)=W_s(I_{\mathrm{sat}})
\]

它不应被要求独立恢复精确立面或动态物体。

### 9.3 Ground measurement encoder

固定长度 chunk 输入 VGGT 或其他 geometry encoder，生成：

- local geometry tokens；
- world-coordinate splatted features；
- ray/voxel observation mask；
- geometry confidence。

### 9.4 Evidence-aware updater

Updater 不应只是对 satellite 和 ground 做一次 cross-attention，而应显式实现：

- read old state；
- compare new measurement；
- predict update gate；
- write local residual；
- preserve unsupported regions；
- update confidence/provenance。

可以写成：

\[
\alpha_t
=
g(Z_{t-1},G_t,C_{t-1},C_t^{\mathrm{conf}})
\]

\[
Z_t
=
(1-\alpha_t)\odot Z_{t-1}
+
\alpha_t\odot \widetilde Z_t
\]

其中 \(\alpha_t\) 应受到真实 observation support 约束，而不是完全自由的 attention map。

### 9.5 Frozen readers

至少保留两类互补的 frozen geometry readouts：

- BEV/static：height、occupancy、semantics；
- perspective/continuous：target depth、density、surface query。

RGB 可以作为辅助读出，但不能成为主要几何证据。

---

## 10. 实验设计

### 10.1 数据单位

不再使用“每个 target frame 对应一个 64 m tile”的高度重叠样本作为核心统计单位。

推荐：

- 以 100–200 m 的世界 tile 或明确路段为一个 scene unit；
- 对每个 scene unit 提取一次 satellite tile；
- 将经过该 tile 的车端轨迹切成空间连续 chunks；
- 每个 chunk 内帧数和几何基线固定；
- evaluation 按空间 block 或 scene unit 统计。

### 10.2 主要条件

| 条件 | 要回答的问题 |
|---|---|
| Learned prior / fixed XY | 没有地理图像时网络与坐标先验能知道多少 |
| Satellite only, \(Z_0\) | 车辆到达前 overhead 能初始化多少静态世界状态 |
| Ground streaming only | 仅靠经过的车端 chunks 如何建立状态 |
| Satellite initialization + ground updates | 两种异步证据能否形成更好的持续状态 |
| One-shot satellite-ground fusion | 收益是否只来自普通的一次性 feature fusion |
| Random / shifted satellite | 是否真正使用当前位置的地理内容 |
| Full accumulated ground/LiDAR | 可达到的局部几何 upper bound |

### 10.3 主要指标

#### Current-state query error

对每一步 \(t\) 报告：

\[
E_{\mathrm{visited}}(t),
\quad
E_{\mathrm{ahead}}(t),
\quad
E_{\mathrm{offroute}}(t)
\]

具体可以包含：

- occupancy IoU / F-score；
- height MAE/RMSE；
- LiDAR surface error；
- continuous density/surface query；
- target depth AbsRel/RMSE/\(\delta_1\)。

#### Update gain

衡量车辆到达某区域后，新 measurement 相对旧状态带来的改善：

\[
G_{\mathrm{update}}(t)
=
E(Z_{t-1};M_t)-E(Z_t;M_t)
\]

#### Prior utility before arrival

衡量 satellite 在 ground 尚未到达时，相对 XY/learned prior 的收益：

\[
G_{\mathrm{prior}}(t)
=
E(Z_{\mathrm{XY}};M_{\mathrm{ahead}}^t)
-
E(Z_0;M_{\mathrm{ahead}}^t)
\]

#### Retention / forgetting

对较早已经观察的区域 \(M_i\)，比较经过后续更新后的误差变化：

\[
F_{i\rightarrow t}
=
E(Z_t;M_i)-E(Z_i;M_i)
\]

理想情况下该值接近 0 或继续下降。

#### Reader transfer

如果 writer 训练时未使用 reader \(Q_h\)，最终仍应报告：

\[
E(Q_h(Z_t))
\]

这用于检验信息是否进入共享状态，而不是仅被某个 task head 读取。

### 10.4 因果控制

仍需保留：

- random satellite tile；
- along-road / cross-road shift；
- rotation / scale perturbation；
- matched-capacity XY writer；
- satellite dropped after initialization；
- update order permutation；
- ground measurement dropout。

新增的关键控制是：

> **one-shot fusion 与 persistent update 在相同最终输入集合下的比较。**

如果两者最终输入完全相同，但 persistent state 在 retention、任意时刻查询和 held-out readers 上更好，才能证明状态设计的价值。

---

## 11. 首轮最小验证

在进行完整重构前，可以先用一个小实验判断信号。

### Step 1：选择 scene tile

- 使用一个 100–200 m KITTI-360 路段；
- 聚合该区域全部可用 LiDAR 作为静态几何 reference；
- 提取一个固定 georeferenced satellite tile。

### Step 2：构造顺序 chunks

- 每个 chunk 固定 4–8 个 rig timestamps；
- chunk 内保证 geometry encoder 有足够基线；
- 相邻 chunks 沿轨迹带来新的空间覆盖；
- 不再把 chunk 内删帧作为主实验变量。

### Step 3：建立状态序列

生成：

\[
Z_0,Z_1,\ldots,Z_T
\]

其中 \(Z_0\) 为 satellite initialization，\(Z_t\) 为吸收前 \(t\) 个 ground chunks 后的状态。

### Step 4：画四条核心曲线

横轴为 traversed distance / assimilated chunks：

1. visited geometry error；
2. ahead static-layout error；
3. earlier-region forgetting；
4. satellite 相对 matched XY 的收益。

### Step 5：首轮通过条件

只有同时满足以下条件才扩大实验：

1. Satellite initialization 在车辆到达前改善 ahead 或 off-route 的静态结构；
2. 车辆 chunk 到达后能够进一步改善局部三维几何；
3. ground update 不会显著破坏其他区域；
4. aligned satellite 优于 random/shifted satellite；
5. 同一个状态的收益至少能被两个 frozen readers 读取；
6. 最好有一个 held-out reader 在不参与 writer 训练时仍然改善。

---

## 12. 当前代码的复用与必须修改部分

### 12.1 可以复用

当前 KITTI-360 工程可以继续复用：

- satellite tile 的地理对齐与裁剪；
- 多相机 ground input 构造；
- VGGT cache 和 geometry encoding；
- georeferenced BEV grid；
- satellite writer / heightmap prior 的部分实现；
- frozen RGB/depth/height readers；
- random、shift 和 fixed-XY controls；
- paired scene-level evaluator 的基础结构。

### 12.2 必须修改

#### 数据采样

从 target-centered、彼此高度重叠的局部样本，改为：

- scene-centered world tile；
- 轨迹的顺序 chunks；
- 每个 chunk 的明确新增 observation support；
- scene/block 级统计。

#### Stage A

从 dense-ground latent teacher 改为 world-defined geometry state：

- 固定 vertical datum；
- 使用累计 LiDAR/UAV/occupancy 定义 query target；
- ground encoder 不再决定世界状态的全部语义。

#### Stage B

从一次性 completion：

\[
\hat Z=C(Z_{\mathrm{sparse}},I_{\mathrm{sat}})
\]

改为递归 update：

\[
Z_t=U(Z_{t-1},G_t,M_t,C_t)
\]

#### 评估

从 dense/sparse/full 三分支单点比较，改为：

- state trajectory \(Z_0\rightarrow Z_T\)；
- visited/ahead/off-route 分区；
- update gain；
- retention/forgetting；
- reader transfer；
- one-shot fusion 对照。

### 12.3 对现有实验的定位

当前 dense/sparse/satellite completion 实验不必作废，可以重新定位为：

> **局部 writer/read interface 的预实验。**

它可以回答 satellite feature 是否有可能通过冻结 geometry readers 被读取，但不再负责定义完整论文任务。

---

## 13. 方法贡献应该落在哪里

论文不能只贡献一个新的 cross-attention 模块。更有意义的方法贡献组合是：

1. **Georeferenced persistent state**：独立于当前 camera/chunk 的世界坐标状态；
2. **Source-specific writers**：satellite、UAV、vehicle 通过不同 observation model 写入同一状态；
3. **Evidence-aware recurrent update**：根据 support、confidence、provenance 决定保留、细化或覆盖；
4. **Frozen multi-query interface**：同一状态支持多个 geometry/readout tasks；
5. **State-quality evaluation protocol**：评价提前知识、局部修正、长期保持和 reader transfer。

其中最关键的不是模型规模，而是把“世界状态”从一句表述变成可检验的行为契约。

---

## 14. 与 VGGT-World 的关系

VGGT-World 的核心是：

> 由历史视频预测未来时刻的 geometry latent，并通过冻结 VGGT 后层/heads 读取未来几何。

本工作的对应关系是：

| VGGT-World | 本工作 |
|---|---|
| 时间轴上的未来 latent prediction | 世界坐标中的 state initialization/update |
| 历史帧预测未来帧 geometry state | overhead prior 与 ground chunks 更新同一 spatial state |
| 主要变量是 prediction horizon | 主要变量是 traversed spatial support |
| query 某个未来时刻 | query 任意世界位置和任意状态版本 |
| temporal world model | geospatial state estimator / filter |

因此，本工作并不是 VGGT-World 加一个 satellite condition，而是把“预测可解码几何状态”的思想从短期时间预测迁移到异步空间证据同化。

---

## 15. 推荐论文主张

不建议使用：

> Satellite images recover complete 3D geometry beyond the vehicle trajectory.

也不建议继续使用：

> Satellite images complete the representation lost by removing ground frames.

推荐主张为：

> **We study how asynchronous observations acquired at different altitudes can initialize and continuously update a persistent georeferenced 3D world state. Overhead imagery provides wide-area static structure before vehicle arrival, while sequential ground chunks locally verify, refine, and correct the state through a shared frozen query interface.**

中文：

> **我们研究不同高度、不同时间到达的空间观测如何初始化并持续更新一个持久、地理对齐的三维世界状态。俯视图在车辆到达前提供大范围静态结构，连续车端 chunk 则通过统一的冻结查询接口，对该状态进行局部确认、细化和纠错。**

更短的一句话版本：

> **不是为每个任务重新融合卫星和街景，而是让它们在不同时间共同维护同一个可查询的世界状态。**

---

## 16. 风险与停止条件

### 风险 1：world state 仍然只是某个 head 的隐藏 feature

应通过 frozen multi-readers、held-out reader 和 one-shot fusion 对照排除。

### 风险 2：satellite 只提供坐标模板

应使用 matched-capacity XY、random tile 和跨地理区域 split 排除。

### 风险 3：KITTI-360 无法评价完整 off-route 3D

应限制主张，并使用 UAV 区域做高覆盖三维验证。不能用模型自己的补全结果充当真值。

### 风险 4：persistent updater 只是在做普通 temporal BEV fusion

必须展示：

- satellite-only initialization；
- ground 到达后的纠错；
- 跨 chunk 状态保持；
- world-coordinate arbitrary queries；
- 多 reader 可读性。

### 建议停止的情况

如果首轮实验出现以下结果，不建议继续扩大：

- satellite initialization 与 matched XY 无区别；
- aligned 与 random/shifted satellite 无区别；
- ground update 只能改善当前输出，却不能稳定写入 state；
- 连续更新造成严重 forgetting；
- 收益只存在于训练所用 reader，held-out reader 完全无收益；
- UAV 高覆盖区域也不能显示不同高度证据的互补更新。

---

## 17. 推荐执行顺序

1. 将当前实验作为 writer/read interface probe 跑完，不再继续增加 chunk 内稀疏设置。
2. 修复统一 vertical datum 和世界坐标定义。
3. 选取一个 100–200 m scene tile，构造固定长度的顺序 ground chunks。
4. 用累计 LiDAR 建立第一版 world-defined geometry targets。
5. 将当前 completion 改造成 recurrent, evidence-aware state updater。
6. 实现 visited/ahead/off-route、update gain 和 forgetting 指标。
7. 与 ground-only、satellite-only、matched XY、one-shot fusion、random/shifted satellite 比较。
8. 若 KITTI-360 首轮成立，再在唯一 UAV 区域验证 satellite–UAV–vehicle 三源状态更新。
9. 最后才考虑接入 VLA/planning，作为 persistent state 的下游价值验证，而不是一开始同时训练完整 VLA。

---

## 18. 最终判断

“chunk 内少几帧，卫星能否恢复”是一个可做但牵强的实验问题。

更值得投入的研究问题是：

> **一个世界状态如何在离线俯视先验和在线车端测量之间持续演化，并在任意时刻保持可查询、可纠错、不过度遗忘。**

这里 chunk 的存在是为了让 geometry encoder 获得稳定的局部多视图测量；真正的研究变量是车辆已经覆盖的空间以及状态已经吸收的证据，而不是 chunk 内多一帧还是少一帧。

这条主线也更符合完整的 satellite–UAV–vehicle 研究路径：三者不是为了同时塞入一个 fusion network，而是在不同空间尺度和不同时间点共同维护同一个世界状态。
