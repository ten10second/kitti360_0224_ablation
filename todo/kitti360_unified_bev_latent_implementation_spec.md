# KITTI-360 条件下的卫星—地面统一 BEV Scene Latent：研究叙事、实现与完整证据链

> 文档用途：直接交给 Codex 作为项目实现规格。  
> 当前定位：**以稀疏跨视角 NVS 为任务载体的场景表征学习工作**。  
> 核心约束：不得把项目退化成“卫星特征与地面特征 concat 后端到端生成目标图像”的普通 NVS 网络。

---

## 0. 一页结论

### 0.1 一句话问题定义

给定一张地理配准卫星图和少量带位姿的 KITTI-360 地面图像，恢复一个由充分地面观测定义的、地理配准且可查询的 BEV 场景隐空间，并从中渲染未输入的目标地面视角。

### 0.2 一句话方法

先用密集地面观测学习并固定一个 `ground-generative BEV latent space`；再把卫星图视为该空间的全局俯视先验，把稀疏地面图视为局部真实观测，学习恢复 dense-ground reference latent；最后用冻结的地面视角 decoder 验证恢复结果。

### 0.3 一句话论文定位

这是 **cross-view scene representation learning**，不是通用视觉基础表征，也不只是一个卫星辅助 NVS 网络。

### 0.4 与 Cross-View Splatter 的关键差别

- Cross-View Splatter：从卫星和地面图像直接预测、合并显式 3D Gaussian primitives。
- 本项目：先由充分地面观测定义一个固定的 ground-generative latent space，再研究卫星与稀疏地面观测如何在这个空间中进行 latent inference/completion。
- 差别不能写成“3DGS vs 隐空间”；必须通过 **reference latent、冻结 decoder、source-subset 一致性、跨模态恢复与几何查询**形成证据。

### 0.5 核心变量

密集地面参考表示：

\[
Z^*=E_{\mathrm{gnd}}(\mathcal G_{\mathrm{dense}})
\]

跨视角恢复表示：

\[
\hat Z=C\left(E_{\mathrm{sat}}(I_{\mathrm{sat}}),E_{\mathrm{gnd}}(\mathcal G_{\mathrm{sparse}})\right)
\]

冻结 decoder 查询：

\[
\hat I_t,\hat D_t
=R_{\mathrm{frozen}}(\hat Z,T_t)
\]

研究对象是 \(Z\)，NVS 和 depth/occupancy 是验证接口。

---

## 1. 研究问题、假设与主张边界

## 1.1 研究问题

卫星图与地面图观察同一个地理场景，但可见内容互补：

- 卫星图：道路、建筑 footprint、屋顶、地表、全局空间覆盖；
- 地面图：立面、局部深度、遮挡关系和真实街景外观；
- 卫星图无法直接观察真实立面；
- 稀疏地面图无法覆盖完整区域。

本项目不再研究“卫星图到底有没有作用”。卫星图的角色已固定为：

> 在统一世界坐标下，为 ground-generative latent 提供全局、静态、低频的俯视场景先验。

真正的问题是：

> 两类不完整且异质的观测，能否被映射为同一个由真实地面观测定义的场景状态？

## 1.2 核心假设

1. KITTI-360 的位姿和标定可以把不同地面相机观测连续 lift 到统一公制 BEV 网格。
2. 密集地面观测可以定义一个稳定、可查询、跨 tile 共享 decoder 的 ground-generative latent。
3. 卫星图和稀疏地面图的组合能够比稀疏地面图单独更好地恢复该 reference latent。
4. 恢复质量如果发生在 latent 中，那么冻结 decoder 后仍应反映在 NVS 与几何输出上。
5. 对齐的真实卫星内容应优于只有 XY 坐标、错位卫星图和简单特征拼接。

## 1.3 表征必须满足的行为定义

### Canonical

同一个世界位置无论由哪个时间、相机或模态观察，都写入同一个 BEV cell：

\[
Z[x,y]\leftrightarrow \text{world column }(x,y,\cdot)
\]

### Modality-compatible

卫星、稀疏地面和二者组合产生的是同一个 latent space 的不同完整程度估计，而不是三个互不兼容的 feature space。

### Composable

增加地面 source views 时，仅通过同一个前馈聚合器更新 latent，不做 per-scene test-time optimization：

\[
\hat Z^{(1)}\rightarrow\hat Z^{(2)}\rightarrow\hat Z^{(4)}\rightarrow\hat Z^{(8)}
\]

### Queryable

同一个 latent 至少支持 RGB 和 depth/density 查询：

\[
(\sigma,\mathbf c)=D\left(Z(x,y),z,\mathbf d\right)
\]

### Decoder-independent

Stage B 中 decoder 必须被冻结；否则不能区分“latent 被恢复”与“生成器在末端重新幻觉”。

## 1.4 允许与禁止的论文主张

允许：

- georeferenced cross-view scene representation learning；
- satellite-assisted sparse-ground latent completion；
- ground-generative BEV scene latent；
- KITTI-360 地面可行域中的 3DoF NVS；
- 同一 latent 的 RGB 与几何查询。

禁止或暂不主张：

- 通用 foundation representation；
- 单张卫星图精确恢复真实立面；
- 完整 BEV world state；
- 任意高度、任意姿态的自由 6DoF NVS；
- 通过增加 uncertainty、temporal、semantic 模块宣称新问题；
- 仅以“隐空间不同于3DGS”作为创新。

---

## 2. KITTI-360 数据条件与任务定义

官方入口：<https://www.cvlibs.net/datasets/kitti-360/>  
官方脚本：<https://github.com/autonomousvision/kitti360Scripts>

KITTI-360 具备：

- 前方 90° 透视双目相机，基线约 60 cm；
- 左右各一个 180° 鱼眼相机；
- Velodyne HDL-64E 与 SICK 扫描单元；
- GPS/IMU 与地理配准相机位姿；
- 多段道路序列及 2D/3D 语义标注。

## 2.1 本项目实际使用的数据

### Ground source pool

- 前视左目 RGB：主要透视 source/target；
- 前视右目：优先用于双目/深度辅助，不默认作为与左目同时间的 source；
- 左右鱼眼：转换为多个虚拟 pinhole crops 后作为多方向 source；
- LiDAR：metric depth、occupancy/geometry 监督；
- 官方相机标定与世界位姿。

### Satellite input

KITTI-360 本身不应被假设包含可直接使用的卫星 tile。实现需要用户提供合法来源的地理配准卫星/正射影像，以及：

- 图像路径；
- GSD 或仿射 geotransform；
- CRS/经纬度信息；
- KITTI world frame 到卫星 ENU/grid 的变换。

不得自动抓取有授权风险的地图影像。没有卫星影像与可靠配准时，Stage B 不应启动。

## 2.2 Target pose 的定义

内部渲染使用完整相机位姿：

\[
T_t=[R_t\mid t_t]\in SE(3)
\]

但 KITTI-360 的真实分布是道路上的车辆运动，因此论文任务有效上定义为：

\[
p_t=(x_t,y_t,\mathrm{yaw}_t)
\]

- 固定或近似固定相机高度；
- roll/pitch 使用真实标定值，但不作为自由外推维度；
- 不宣称自由 6DoF；
- 主评测 target 使用原始前视透视图，避免鱼眼重采样质量影响主指标；
- 鱼眼虚拟 crop 可作为补充 target，用于检查 yaw 查询能力。

## 2.3 场景 tile 定义

默认建议，可通过配置修改：

- tile 尺寸：64 m × 64 m；
- BEV 分辨率：0.5 m/cell；
- grid：128 × 128；
- latent channels：128；
- tile 采用固定 ENU/north-up 坐标；
- 同一路段相邻 tile 可有少量 overlap，但 train/val/test 之间必须有空间 buffer。

不要在初版中引入距离 bin 或高度 bin。地面 lift 使用连续深度反投影和 bilinear splatting。

## 2.4 数据划分原则

严禁随机 frame split。必须采用以下之一：

1. sequence-level split；
2. geographic block split；
3. sequence＋geographic 双重 split。

要求：

- train/val/test tile 不共享同一地理建筑；
- 对循环路线按地理坐标去重，防止不同时间重复经过同一区域；
- split 边界设置空间 buffer，例如至少一个 tile 尺寸；
- target timestamp 的所有高度重叠相机必须从 source 中排除；
- target 前后近重复帧设置最小空间间隔，间隔写入配置，不硬编码。

---

## 3. 数据预处理规范

## 3.1 坐标系统一

定义：

- `T_world_rig[t]`：车辆 rig 到 world；
- `T_rig_cam[c]`：相机到 rig；
- `T_world_cam[t,c] = T_world_rig[t] @ T_rig_cam[c]`；
- `T_sat_world`：KITTI world 到卫星 ENU/grid。

所有变换在代码中明确采用 column-vector 或 row-vector 约定，并通过单元测试固定。不得在不同模块中隐式转置。

必须生成 QA 可视化：

1. 车辆轨迹叠加到卫星图；
2. LiDAR 投影到每种相机图像；
3. 相机 frustum 投影到 BEV；
4. 地面 feature splat 与道路/建筑位置叠加。

配准在视觉上未通过时不得训练。

## 3.2 鱼眼转虚拟透视相机

`image_00` 的 rectified 图像约为 1408×376，不能在送入 VGGT 前直接压缩为
160×96。默认沿 calibrated optical axis 使用两个 560 px 宽、72 px overlap
的水平 crop；两个 crop 共享 `image_00` 的物理光心与外参，但分别使用
`c_x' = c_x - x_0` 后再按输出尺寸缩放的内参。

不要把 180° 鱼眼一次性展开成单张超宽 pinhole 图。每个鱼眼生成多个 tangent perspective crops。

建议默认：

- 每个鱼眼 3 个 crop；
- crop FOV：90°；
- 相对鱼眼光轴 yaw offsets：`[-45°, 0°, +45°]`；
- 输出统一分辨率，例如 384 × 640；
- 保存 valid mask；
- crop 配置必须可修改。

默认 source layout 为 `front2 + left3 + right3 = 8 views/frame`。用于 VGGT
多帧 metric scale 的真实基线必须来自 `T_world_rig`/车辆 pose；virtual crop
数量不得改变 metric anchor 的权重。

主前视 target 同样使用这两个 calibrated crops，分别携带 crop-specific `K`，
由同一 latent/renderer 同时查询。禁止把 1408×376 整图直接压缩到 160×96
作为 RGB/depth 监督；该操作会把 3.74:1 横向视场扭曲为 1.67:1。

对于虚拟相机 `v`：

\[
T_{w\leftarrow v}(t)=T_{w\leftarrow rig}(t)T_{rig\leftarrow f}T_{f\leftarrow v}
\]

实现必须使用官方鱼眼相机模型做 inverse warping：虚拟 pinhole pixel → 3D ray → 鱼眼 projection → raw fisheye sampling。

每个虚拟 crop 需要保存：

- RGB；
- `K_virtual`；
- `T_world_virtual`；
- valid mask；
- 原始相机、时间戳、crop rotation 元数据。

## 3.3 LiDAR 深度

初版优先使用 metric LiDAR，而不是未经校准的单目深度。

对每张 source/target perspective image：

1. 将同时间或运动补偿后的 LiDAR 转到相机坐标；
2. 投影并 z-buffer；
3. 输出 sparse depth 与 validity mask；
4. 可选地用 LiDAR 监督 depth completion；
5. 动态物体区域不做多帧静态聚合。

地面像素连续反投影：

\[
X_{iu}=T_{w\leftarrow c_i}\left(d_{iu}K_i^{-1}\bar u\right)
=(x_{iu},y_{iu},z_{iu})
\]

再按连续 \((x,y)\) 坐标 bilinear splat 到 BEV。这里不使用 depth bins。

## 3.4 动态区域

统一 scene latent 初版定义为静态场景表示。车辆、行人、骑行者等动态类：

- 训练 photometric loss 时 masking；
- geometry aggregation 时 masking；
- 主 NVS 指标同时报告 `static-only` 与 `all-pixel`，以 static-only 为核心；
- 不把动态建模加入初版创新。

## 3.5 Tile manifest

每个 tile 生成一个 manifest，至少包括：

```json
{
  "tile_id": "...",
  "split": "train|val|test",
  "world_bounds": [0, 0, 64, 64],
  "bev_resolution_m": 0.5,
  "satellite": {
    "path": "...",
    "world_to_pixel": [[0,0,0],[0,0,0],[0,0,1]],
    "valid_mask": "..."
  },
  "views": [
    {
      "view_id": "...",
      "timestamp": "...",
      "camera_type": "front|fisheye_virtual",
      "rgb": "...",
      "depth": "...",
      "dynamic_mask": "...",
      "K": [[0,0,0],[0,0,0],[0,0,1]],
      "T_world_cam": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
    }
  ]
}
```

---

## 4. 模型：两阶段表征学习

## 4.1 总体原则

必须严格区分：

- Stage A：定义 ground-generative latent space；
- Stage B：从 satellite＋sparse ground 恢复该 latent；
- Stage B 的 renderer 和 reference latent encoder 默认冻结。

如果直接联合训练所有模块，项目将失去“恢复预先定义场景隐空间”的证据。

---

## 4.2 Stage A：Ground-Generative Reference Latent

### 输入

每个 tile 的密集地面 reference view set：

\[
\mathcal G_{\mathrm{dense}}=\{I_i,K_i,T_i,D_i\}_{i=1}^{N_d}
\]

建议 `N_d` 初始为 16 或 32，按空间覆盖采样而非连续帧堆叠。

对每次训练采样 target view `t`：

- target 不进入 dense source set；
- target 同时间的高重叠相机不进入 source；
- 必要时排除与 target 距离过近的帧。

### Ground encoder 与连续 lift

```text
RGB -> shared 2D encoder/FPN -> pixel features
pixel features + metric depth + camera pose -> continuous 3D backprojection
3D points -> bilinear splat on XY -> BEV feature/count/height statistics
BEV context network -> Z_star
```

建议输出：

- `bev_feature_sum`；
- `bev_feature_weight`；
- `bev_observation_mask`；
- 可选 `mean_height`、`height_variance`，只作为输入通道，不做高度 bin；
- `Z_star: [C,H,W]`。

多视图聚合必须 permutation invariant，初版使用 weighted sum/mean 即可，不需要复杂时序模块。

### Queryable decoder

把二维 BEV latent 解释为世界垂直 column 的隐式代码。对目标 ray 上的三维点：

\[
(\sigma,\mathbf c)=D_\theta\left(\operatorname{bilinear}(Z^*,x,y),z,\mathbf d\right)
\]

其中：

- `z` 使用连续高度；
- `d` 是 view direction；
- 使用标准体渲染得到 RGB 与 expected depth；
- ray marching 的连续数值采样不属于 ground lift 的 distance bins；
- 初版不加入 diffusion 或 target-view image generator。

### Stage A 损失

\[
\mathcal L_A=
\lambda_{rgb}\mathcal L_{rgb}
+\lambda_{perc}\mathcal L_{LPIPS}
+\lambda_{depth}\mathcal L_{depth}
+\lambda_{height}\mathcal L_{frozen-height}
\]

初始实现至少需要：

- static-region RGB L1/Charbonnier；
- LPIPS；
- LiDAR-valid pixel depth loss；
- 由 `Z_star` 预测 local-ground-relative height 的共享几何 readout loss。

该几何 readout 是一个 BEV 卷积 coarse-to-fine decoder，不称为 DPT；它只在
Stage A 训练一次，并与 renderer 一起冻结。所有 Stage-B 变体必须由同一组权重读取。

### Stage A 验收

只有满足以下条件才能进入 Stage B：

1. 单 tile overfit 成功；
2. 多 tile 共享 encoder/decoder 后能渲染 held-out target；
3. dense-ground 明显优于 sparse-ground；
4. rendered depth 与 LiDAR 对齐；
5. decoder 在 val/test tile 上不做 per-scene optimization；
6. reference latent 可缓存并由相同 decoder 重放。

完成后冻结：

- ground reference encoder，或至少冻结其输出定义；
- renderer；
- geometry head。

导出每个训练 tile 的 reference latent、valid mask 与 coverage map。

---

## 4.3 Stage B：Satellite-Assisted Latent Recovery

### 输入

\[
I_{sat},\qquad
\mathcal G_{sparse}\subset\mathcal G_{dense}
\]

建议训练时随机采样非恒等分支：

\[
N_s\in\{1,2,4\}
\]

`N_s=8` 时 completion 按契约逐位等于 frozen `Z_gnd`，没有 Stage-B
梯度，因此只进入 identity 回归测试与 C1 评估，不占用训练 step。

### Satellite encoder

卫星图已经与公制 XY 网格配准：

```text
satellite RGB -> CNN/ViT/FPN -> resample to canonical BEV -> Z_sat
```

必须包含 satellite valid mask。可使用 ImageNet/DINO 初始化，但 target renderer 不得直接 cross-attend 原始 satellite DINO tokens；所有信息需先进入 canonical latent。

### Sparse ground encoder

复用 Stage A 的 ground encoder/lifter，得到：

\[
Z_{gnd}^{sparse},M_{gnd}^{sparse}
\]

默认冻结 ground encoder，先只训练 satellite encoder 和 completion/fusion network；若性能不足，再做小学习率联合微调，并保留冻结版本作为对照。

### Fusion/completion

最小推荐形式：

\[
\hat Z=Z_{gnd}^{sparse}+\alpha(N_s)\,G_{write}\odot\Delta Z
\]

\[
\alpha(N_s)=1-\frac{N_s}{N_d},\qquad
G_{write}=\sigma\left(G([Z_{sat},Z_{gnd}^{sparse},M_{gnd}])\right)
\]

解释：

- `Z_gnd_sparse` 是局部真实观测锚点，satellite/XY 先验进入 `Delta Z`；
- `G_write` 是可审计的写入 gate，不称为 uncertainty；
- 在 `N_s=N_d` 时 `alpha=0`，必须逐位返回 ground latent；
- observed/fill 区域由确定性 support mask 定义，不由 gate 猜测；
- `M_gnd` 是确定性观测 mask/count，不把 uncertainty modeling 作为贡献。

必须实现以下替代融合以供消融：

- concat＋conv；
- simple add；
- ground-only；
- satellite-only；
- coordinate-only；
- proposed residual update。

其中 `coordinate-only`（Evidence B3）的定义必须严格固定：用 canonical
BEV 每个 pixel-center 的公制相对坐标生成确定性的 XY/Fourier positional
encoding，替换 `Z_sat` 后进入与 proposed 完全相同的 write-gate/delta
通路。该位置编码不接收梯度，不得使用可学习的 `C×H×W` 参数表，也不含
tile ID 或绝对 GPS。可学习的逐 cell 空间模板若作为额外诊断，必须单独命名
为 `learned-template`，不能参与 `aligned satellite > coordinate-only` gate。

### Stage B 损失

\[
\mathcal L_B=
\lambda_{anchor}\mathcal L_{anchor}^{obs}
+\lambda_{geo}\mathcal L_{frozen-geo}^{fill}
+\lambda_{low}\mathcal L_{RGB-lowfreq}
+\lambda_{app}\mathcal L_{appearance}^{supported}
+\epsilon\mathcal L_{latent-diag}
\]

其中：

#### Observation anchor

\[
M_{obs}=M_{gnd}^{sparse},\qquad
M_{fill}=M_{dense-geometry}\land\neg M_{obs}
\]

- `M_obs` 上约束 `Z_hat` 不偏离 `Z_gnd_sparse`；
- `M_fill` 上通过同一个 frozen height readout 监督几何补全；
- 全 tensor `Z_hat-Z_star` 只保留为很小的 regularizer/诊断，不作为主目标。

#### Frozen render loss

\[
\hat I_t=R_{frozen}(\hat Z,T_t)
\]

全图只强约束低频布局；高频外观只在 target pixel 反投影到 `M_obs` 的区域
约束。反投影 depth 由 frozen dense-ground renderer 稠密提供，并在 target
LiDAR-valid pixel 处用公制真值覆盖。卫星不可见立面的真实高频纹理不作强监督。

#### Frozen geometry loss

使用 Stage A 唯一的 frozen relative-height decoder，在 `M_fill` 与 LiDAR
relative-height target 对齐。Satellite 自带 height auxiliary 默认权重为 0，
不能作为主几何证据旁路。

### Stage B 验收

1. `satellite+sparse ground` 在冻结 decoder 下优于 `sparse ground only`；
2. aligned satellite 优于 coordinate-only 和 misaligned satellite；
3. 增加 source views 时性能总体单调趋近 dense-ground upper bound；
4. 改善同时体现在至少一种几何指标，而不只是在 LPIPS；
5. 不允许依赖 per-scene optimization。

---

## 5. 完整实验与证据链

实验分为四层。每层对应一个明确的科学问题，不能只报告最终 NVS 表格。

---

## 5.1 Evidence A：Ground-generative latent 是否成立

### A1. Dense-ground NVS upper bound

输入 dense ground source set，渲染 held-out front targets。

比较：

- single source ground；
- 4-view ground；
- 8-view ground；
- dense ground reference。

指标：

- PSNR；
- SSIM；
- LPIPS；
- static-region PSNR/SSIM/LPIPS；
- LiDAR depth AbsRel、RMSE、δ1。

成功条件：dense-ground reference 显著优于稀疏输入，并能在多个未见 tile 上工作。

### A2. Geometry query

从同一 `Z_star` 输出 depth 或 occupancy：

- target-view depth；
- BEV occupancy/height（如果实现）；
- 可选从 density 提取点云/mesh，与聚合 LiDAR 比较 Chamfer/height error。

目的：证明 latent 不是只保存二维生成纹理。

### A3. Frozen replay

缓存 `Z_star`，重新加载后仅调用 frozen renderer。输出必须与在线 encoder 输出一致到数值容差。

目的：证明 latent 是独立、稳定的场景载体，而不是 encoder/decoder 之间的隐式 side channel。

---

## 5.2 Evidence B：跨视角输入能否恢复 reference latent

固定同一 target 集合，比较：

| ID | 输入/方法 | 验证问题 |
|---|---|---|
| B0 | Dense ground `Z_star` | upper bound |
| B1 | Sparse ground only | 无卫星时的下界 |
| B2 | Satellite only | 卫星能够恢复多少低频/布局 |
| B3 | Sparse ground + XY positional encoding | 公制坐标本身是否足够 |
| B4 | Sparse ground + satellite, naive concat | 简单融合是否已经足够 |
| B5 | Sparse ground + satellite, simple add | 残差设计是否必要 |
| B6 | Proposed aligned latent recovery | 完整方法 |
| B7 | Proposed + shifted/rotated satellite | 是否真正利用空间对应 |
| B8 | Proposed + satellite from another tile | 是否利用场景内容而非图像统计 |
| B9 | Direct target-conditioned cross-attention | 统一 latent 是否优于直接条件生成 |
| B10 | Cross-View Splatter/可复现3DGS基线 | 与最近工作比较；若代码不可得需明确 |

所有 B1–B10 使用：

- 相同 source views；
- 相同 target views；
- 相同训练/测试空间 split；
- 相同动态 mask；
- 可比较的输入分辨率。

报告三类结果：

1. latent recovery；
2. frozen NVS；
3. frozen geometry。

---

## 5.3 Evidence C：表征性质是否成立

### C1. Source-count composability

对同一 tile 与 target，使用：

\[
N_s\in\{1,2,4,8\}
\]

记录：

- latent distance 到 `Z_star`；
- NVS LPIPS/PSNR；
- depth RMSE；
- observed/unobserved cell 分区结果。

理想趋势：

\[
\hat Z^{(1)}\rightarrow\hat Z^{(2)}\rightarrow\hat Z^{(4)}\rightarrow Z^*
\]

不强制每个样本严格单调，但总体均值及置信区间应呈改善趋势。

### C2. Disjoint subset consistency

从同一 tile 采样两个不重叠 source subsets `A` 和 `B`：

\[
\hat Z_A=C(I_{sat},\mathcal G_A),\qquad
\hat Z_B=C(I_{sat},\mathcal G_B)
\]

在两者共同可观测区域和 shared geometry 输出上比较：

- normalized latent cosine/L1；
- density/depth consistency；
- 对同一 target 的 render consistency。

不要强制所有高频 appearance channels 完全相等；核心报告 geometry/shared output consistency。

### C3. Modality recovery

分别从以下输入恢复同一个 reference latent：

- satellite only；
- ground only；
- satellite＋ground。

验证三者确实位于同一个 frozen decoder 可接受的 latent space，而不是各自调用不同 decoder。

### C4. Satellite alignment sensitivity

对卫星图施加受控扰动：

- 平移：例如 1、2、5、10 m；
- 旋转：例如 2°、5°、10°；
- 替换为相邻或随机 tile。

观察 latent/NVS/geometry 的退化曲线。

目的不是做鲁棒定位，而是证明方法使用了地理对应关系。初版不训练 alignment correction 模块。

### C5. Decoder freezing test

至少报告：

- decoder frozen；
- decoder jointly finetuned。

核心结论必须来自 frozen 版本。joint finetune 只作为性能上限，不作为表征证据。

### C6. Cross-tile generalization

在从未参与训练的地理 tile 上前馈生成 latent，不允许：

- per-scene latent optimization；
- test-time pose refinement；
- 使用 test target 图像调整 latent。

---

## 5.4 Evidence D：KITTI-360 特有的数据控制实验

### D1. Camera-source composition

比较：

- front only；
- fisheye virtual crops only；
- front＋fisheye；
- front＋fisheye＋satellite。

目的：区分“更大地面 FOV”与“卫星俯视先验”的贡献。

### D2. Stereo leakage control

当左前视为 target：

- 同时间右前视不得默认作为 source；
- 单独报告允许/不允许 stereo mate 的结果；
- 核心表格使用无同时间 stereo leakage 的协议。

### D3. Fisheye preprocessing control

比较少量样本：

- 单张超宽 pinhole；
- 多 tangent crops；
- 原始 fisheye 模型（如果后续实现）。

主系统使用多 tangent crops；该实验主要确认重采样方案没有导致异常。

### D4. Static-only evaluation

分别报告：

- all pixels；
- static pixels；
- dynamic pixels（仅作诊断）。

核心结论以 static scene latent 为准。

### D5. Spatial leakage audit

输出：

- train/val/test 轨迹卫星叠加图；
- tile 边界；
- 最近 train-test 地理距离分布；
- 重复路线/loop 检测报告。

没有该审计，不得发布最终结果。

---

## 6. 主要指标与统计规范

## 6.1 NVS

- PSNR ↑；
- SSIM ↑；
- LPIPS ↓；
- static-region 同组指标；
- 可选 DISTS；
- 不以 FID 作为确定性、场景对应 NVS 的核心指标。

## 6.2 Geometry

- depth AbsRel ↓；
- depth RMSE ↓；
- δ1 ↑；
- occupancy IoU ↑（若实现）；
- point/mesh Chamfer ↓（若从 density 提取）；
- BEV height MAE/RMSE ↓（若有可靠 reference）。

## 6.3 Latent

仅在 Stage A 定义被冻结后计算：

- masked normalized L1；
- cosine distance；
- observed/unobserved cell 分区；
- disjoint-source shared-region consistency；
- 可选 CKA 作为诊断，不作为主要优化目标。

## 6.4 统计

- 至少 3 个随机种子用于核心 B/C 实验；
- 报告均值和标准差/95% bootstrap CI；
- tile 为统计单位，不能把相邻帧当作独立样本夸大显著性；
- 同一 target 上做 paired comparison。

---

## 7. 最小成功判据与停止条件

本项目不预设卫星必然大幅提高所有 RGB 指标。最低证据链要求方向性同时成立：

1. `dense ground reference` 能构成稳定 upper bound；
2. `satellite+sparse ground` 在冻结 decoder 下优于 `sparse ground only`；
3. `aligned satellite` 优于 `XY-only`；
4. `aligned satellite` 优于 `misaligned/random satellite`；
5. 增加 source views 后总体趋近 `Z_star`；
6. 改善至少同时出现在 NVS 和一种 geometry/latent 指标；
7. 所有结论在地理隔离 test tiles 上成立。

停止/回退条件：

- Stage A dense ground latent 无法可靠渲染：停止 Stage B，先修正表示/renderer；
- `satellite+ground` 与 `ground+XY` 无差别：说明卫星视觉内容未被利用；
- aligned 与 misaligned 卫星无差别：说明模型忽略卫星空间对应；
- 只有 joint decoder finetune 有提升、frozen decoder 无提升：不能定位为表征恢复；
- 只有 RGB 改善、geometry/latent 恶化：只能定位为生成/NVS方法；
- 若卫星作用有限，应如实报告，不通过增加无关模块掩盖。

---

## 8. 推荐代码结构

```text
project/
├── configs/
│   ├── data_kitti360.yaml
│   ├── stage_a_ground_latent.yaml
│   ├── stage_b_latent_recovery.yaml
│   └── experiments/
├── data/
│   ├── manifests/
│   ├── satellite/
│   ├── virtual_cameras/
│   └── cache/
├── src/
│   ├── datasets/
│   │   ├── kitti360_dataset.py
│   │   ├── tile_sampler.py
│   │   └── source_target_sampler.py
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── fisheye_virtual_camera.py
│   │   ├── lidar_projection.py
│   │   ├── continuous_splat.py
│   │   └── ray_builder.py
│   ├── models/
│   │   ├── image_encoder.py
│   │   ├── satellite_encoder.py
│   │   ├── ground_bev_encoder.py
│   │   ├── latent_completion.py
│   │   ├── column_field_decoder.py
│   │   └── geometry_head.py
│   ├── losses/
│   │   ├── rendering.py
│   │   ├── latent.py
│   │   └── geometry.py
│   └── evaluation/
│       ├── nvs_metrics.py
│       ├── geometry_metrics.py
│       ├── latent_metrics.py
│       └── leakage_audit.py
├── scripts/
│   ├── preprocess_kitti360.py
│   ├── build_virtual_cameras.py
│   ├── align_satellite.py
│   ├── build_tile_manifests.py
│   ├── train_stage_a.py
│   ├── export_reference_latents.py
│   ├── train_stage_b.py
│   ├── eval_evidence_a.py
│   ├── eval_evidence_b.py
│   ├── eval_evidence_c.py
│   └── eval_evidence_d.py
├── tests/
│   ├── test_transforms.py
│   ├── test_virtual_camera.py
│   ├── test_lidar_projection.py
│   ├── test_continuous_splat.py
│   ├── test_target_leakage.py
│   └── test_frozen_decoder.py
└── README.md
```

---

## 9. 建议配置草案

```yaml
data:
  tile_size_m: 64.0
  bev_resolution_m: 0.5
  primary_target_camera: front_left
  source_counts: [1, 2, 4, 8]
  dense_source_count: 24
  exclude_target_timestamp_all_cameras: true
  dynamic_mask: true
  split_mode: geographic_sequence
  split_buffer_m: 64.0

virtual_camera:
  enabled: true
  output_height: 384
  output_width: 640
  fov_deg: 90.0
  yaw_offsets_deg: [-45.0, 0.0, 45.0]

latent:
  channels: 128
  height_min_m: -4.0
  height_max_m: 20.0
  use_depth_bins_for_lift: false
  aggregation: weighted_mean

renderer:
  type: implicit_column_field
  use_full_se3_rays: true
  predict_rgb: true
  predict_depth: true
  target_pose_claim: ground_3dof

stage_a:
  freeze_after_training: true
  export_reference_latents: true

stage_b:
  freeze_ground_encoder: true
  freeze_renderer: true
  fusion: gated_residual
  satellite_modality_dropout: 0.1
```

具体数值可根据显存和数据覆盖修改，但任何改变都写入实验配置并版本化。

---

## 10. 训练伪代码

## 10.1 Stage A

```python
for tile in loader:
    target = sample_target(tile.views)
    dense_sources = sample_dense_sources(
        tile.views,
        exclude_target=target,
        exclude_same_timestamp=True,
    )

    z_star, coverage = ground_encoder(dense_sources)
    rgb_pred, depth_pred = frozen_candidate_renderer(
        z_star,
        target.K,
        target.T_world_cam,
    )

    loss = rgb_loss(rgb_pred, target.rgb, target.static_mask)
    loss += perceptual_loss(rgb_pred, target.rgb, target.static_mask)
    loss += depth_loss(depth_pred, target.depth, target.depth_mask)
    loss.backward()
```

Stage A 收敛后冻结 encoder/renderer，并为训练 tile 导出 `Z_star`。validation/test 的 `Z_star` 仅用于 upper bound 和评估，不可作为 Stage B 输入。

## 10.2 Stage B

```python
for tile in loader:
    target = sample_target(tile.views)
    dense_sources = sample_dense_sources_excluding_target(tile, target)
    sparse_sources = sample_subset(dense_sources, n=sample([1, 2, 4, 8]))

    with torch.no_grad():
        z_star, ref_mask = frozen_ground_encoder(dense_sources)

    z_sat = satellite_encoder(tile.satellite)
    with torch.no_grad():
        z_sparse, sparse_mask = frozen_ground_encoder(sparse_sources)

    z_hat = latent_completion(z_sat, z_sparse, sparse_mask)

    rgb_pred, depth_pred = frozen_renderer(
        z_hat,
        target.K,
        target.T_world_cam,
    )
    height_pred = frozen_height_decoder(z_hat)
    m_obs = sparse_mask
    m_fill = dense_geometry_mask & ~m_obs
    teacher_depth = frozen_renderer(z_star, target.K, target.T_world_cam).depth
    teacher_depth[target.depth_mask] = target.depth[target.depth_mask]
    supported_pixels = backproject_and_sample(teacher_depth, m_obs)

    loss = latent_anchor(z_hat, z_sparse, m_obs)
    loss += geometry_fill_loss(height_pred, relative_height, m_fill)
    loss += low_frequency_rgb_loss(rgb_pred, target.rgb)
    loss += supported_high_frequency_loss(rgb_pred, target.rgb, supported_pixels)
    loss += 0.01 * latent_loss_diagnostic(z_hat, z_star)
    loss.backward()
```

---

## 11. 必须实现的单元测试与可视化

### Geometry tests

- SE(3) compose/inverse round-trip；
- world point → camera → pixel → ray 的一致性；
- virtual perspective center ray 与配置朝向一致；
- fisheye inverse warp valid mask 正确；
- LiDAR z-buffer 不产生负深度；
- continuous splat 的权重和守恒；
- satellite/world affine round-trip。

### Leakage tests

- target view 不在 source list；
- target timestamp 的 stereo/fisheye views 按配置排除；
- test tile 与 train tile 的距离不低于 buffer；
- loop route 的同地理位置不会跨 split。

### Representation tests

- cached `Z_star` 与在线输出重放一致；
- Stage B optimizer 不包含 frozen renderer 参数；
- decoder freezing 前后参数 checksum 不变；
- source 输入顺序置换不改变聚合结果；
- ground observation mask 与真实 splat coverage 一致。

### Required figures

1. 卫星图上的 train/val/test 轨迹与 tile；
2. 前视、鱼眼虚拟 crop、LiDAR depth；
3. ground BEV coverage；
4. `Z_star`、`Z_sparse`、`Z_sat`、`Z_hat` 的 PCA 可视化；
5. aligned/misaligned satellite 的 latent 和 NVS 对比；
6. source count 增加时 latent/NVS/depth 的变化；
7. frozen decoder 的 dense upper bound 与恢复结果。

---

## 12. 实施里程碑

## M0：数据和坐标 QA

交付：

- KITTI-360 loader；
- 鱼眼虚拟 pinhole；
- LiDAR depth；
- satellite/trajectory overlay；
- tile manifests；
- split leakage audit。

Gate：所有投影与叠加可视化正确。

## M1：Ground-only renderer

交付：

- continuous lift-splat；
- ground BEV latent；
- implicit column decoder；
- 单 tile overfit；
- held-out target NVS/depth。

Gate：dense ground 明显优于 sparse ground。

## M2：Reference latent 固化

交付：

- Stage A 多 tile 训练；
- frozen encoder/decoder；
- reference latent cache；
- Evidence A 全部实验。

Gate：缓存 latent 可稳定重放，geometry 有效。

## M3：Satellite latent recovery

交付：

- satellite encoder；
- coordinate-only baseline；
- naive/add/residual fusion；
- Stage B frozen-decoder 训练。

Gate：full aligned 至少优于 ground-only、XY-only 与 misaligned satellite。

## M4：完整证据链

交付：

- Evidence B/C/D；
- 3 seeds；
- 统计与可视化；
- Cross-View Splatter 或最接近基线比较；
- failure cases。

## M5：写作材料

交付：

- 主结果表；
- 四组关键图；
- 数据协议；
- 方法图；
- limitations。

---

## 13. 实现优先级：初版不要加入的内容

初版不要实现：

- diffusion/video diffusion；
- 显式 uncertainty distribution；
- temporal memory；
- semantic 分层 latent；
- 高度/距离离散 bins；
- UAV/6DoF 扩展；
- 动态物体生成；
- target-view-specific satellite cross-attention；
- per-scene NeRF/latent optimization；
- 自动 pose refinement。

这些内容只会模糊“reference latent → cross-view recovery → frozen decoding”的主线。

---

## 14. 论文展开模板

## Introduction 逻辑

1. 稀疏地面图包含真实街景，但空间覆盖不足；卫星图覆盖全局，但无法看到立面。
2. 直接跨视角生成或直接预测场景 primitives，没有显式回答异质观测是否能形成同一个场景状态。
3. 关键观察：地面图连续 lift 后与地理配准卫星图共享 metric XY 索引。
4. 提出 ground-generative BEV latent：先由充分地面观测定义，再由卫星＋稀疏地面观测恢复。
5. 使用冻结 decoder 的 NVS 与几何查询证明改善发生在表示层。

## Contribution 草案

1. 定义一个由充分地面观测学习、在统一世界坐标中可查询的 ground-generative BEV scene latent。
2. 提出卫星全局先验与稀疏地面证据联合恢复该 latent 的跨视角 inference/completion 方法。
3. 在 KITTI-360 上建立冻结解码、source-subset、错位卫星、coordinate-only、NVS 与几何组成的完整表征证据链。

## 一句话摘要

> We learn a ground-generative, georeferenced BEV scene latent from dense street-level observations, and infer this latent from a satellite image and sparse ground views, enabling frozen-decoder novel-view and geometry prediction without per-scene optimization.

---

## 15. 最终交付清单

Codex 完成实现时必须交付：

- [ ] 数据下载/路径说明，不自动下载未授权卫星图；
- [ ] KITTI-360 与卫星配准 QA；
- [ ] 鱼眼虚拟 pinhole 数据；
- [ ] LiDAR depth 与动态 mask；
- [ ] 地理隔离 split 与 leakage audit；
- [ ] Stage A ground-generative latent；
- [ ] frozen encoder/renderer 与 reference latent cache；
- [ ] Stage B satellite-assisted latent recovery；
- [ ] ground-only、satellite-only、XY-only、naive、misaligned、full baselines；
- [ ] Evidence A/B/C/D 实验脚本；
- [ ] NVS、geometry、latent 指标；
- [ ] source-count 与 subset-consistency 实验；
- [ ] 至少 3 seeds 的核心结果；
- [ ] failure cases；
- [ ] README、配置、运行命令和单元测试。

---

## 16. Codex 开工顺序

Codex 不应一次性实现全模型。严格按以下顺序：

1. 检查本地数据目录和现有代码，不覆盖用户已有修改；
2. 实现并测试坐标、虚拟相机、LiDAR 投影；
3. 生成 5–10 个 tile 的可视化 QA；
4. 实现 ground-only 单 tile overfit；
5. 实现多 tile Stage A 并冻结；
6. 导出 reference latent；
7. 先训练 coordinate-only 和 ground-only Stage B；
8. 加 satellite encoder 与 naive fusion；
9. 最后实现 residual completion；
10. 运行最小证据链，确认方向成立后再扩展全量实验。

任何阶段遇到以下情况应停止并报告，而不是增加模块：

- 缺少卫星数据/配准；
- Stage A upper bound 不成立；
- split 存在地理泄漏；
- frozen decoder 无法复现在线结果；
- aligned satellite 没有超过 coordinate-only。
