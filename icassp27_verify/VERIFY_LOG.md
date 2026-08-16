# ICASSP27 改造 · Day-0 验证记录（KITTI-raw 先行）

> 日期：2026-08-16
> 依据文档：`/media/shizhm/sda1/proposal/MM26_submit_review/` 下
> ① `ICASSP27_method_framework.md`（改代码对照文档）
> ② `Satellite_Geometric_Anchor_Sparse_Ground_Checklist.md`（falsification probe 清单）
> ③ `Satellite_Prior_Sparse_Street_Views_Cross_Position_3D_Reconstruction.md`（总方案）
> 代码库：`/media/shizhm/Lenovo/kitti360_0224_ablation`（MM26 AR 框架）
> 数据：`/media/shizhm/Lenovo/KITTI_RAW`（按指示先用 KITTI-raw）
> 验证脚本：本目录 `v1_pose_lidar.py`、`v2_sat_align.py`、`v2b_sat_fit.py`、`v3_tuple_split.py`、`kitti_raw_io.py`，输出在 `out/`
> 环境：conda `maskgit`（torch 2.4.1+cu124，与项目 requirements 一致），GPU RTX 4090

---

## 0. 总结论（TL;DR）

1. **管线基础全部打通**：oxts→世界位姿、LiDAR→前向相机投影、米制 tuple 采样、地理切分在 KITTI-raw 上全部可行，指标满足 checklist Day-0 交付物要求。
2. **发现一个数据级 bug**：KITTI-raw 的 1280×1280 卫星图是**各向异性**的（东西 0.196 m/px、南北 0.128 m/px = 东西×cos(49°)，Web Mercator 下载时纬度轴被压缩）。"0.2 m/px" 只对东西轴近似成立。各向异性校正后配准残差 ~1px，车心居中 + north-up 假设成立；不校正则南北向系统残差 7–42px。**用前必须重采样或改 world→pixel 映射。**
3. **KITTI-360 卫星图（512×512）验证为各向同性** 0.196 m/px（覆盖 ~100m），与计划文档"0.194–0.199 m/px、残差 ~1px"一致（本机 8 条 drive 可用）。KITTI-360 适合作为卫星支路主数据，KITTI-raw 适合按总方案 §4 做"外部验证/工程链路"。
4. **三个缺口**（不阻塞 probe，但必须排进 Day-1）：
   - KITTI-raw tracklet 标注全部为空 → 动态实例屏蔽需换方案（时序一致性 或 KITTI-360 3D 标注）；
   - MASt3R/DUSt3R/VGGT/MVSplat 本机均未安装 → Gate B 外部 baseline 尚未"能跑"；
   - 现有 `prepare_train_test_split.py` 仍是**时域 95/5 切分**（MM26 killer 2 的泄漏源），改造清单要求的新地理划分完全没写。

---

## 1. 数据盘点

### 1.1 KITTI-raw（`/media/shizhm/Lenovo/KITTI_RAW`）

- 5 个日期、**151 条 drive**，全部有 calib 三件套（`calib_cam_to_cam / calib_velo_to_cam / calib_imu_to_velo`；注意 2011_09_28 的 calib 在 `<date>_calib/<date>/` 嵌套目录）。
- 147 条有 velodyne；**66 条有 satellite**（逐帧 1280×1280 PNG，与帧数一致）；image_02 彩色 1242×375（个别 drive 1241×376——**图像尺寸 per-drive 不一，dataloader 不能硬编码**）。
- **有效集合（image+oxts+velodyne+satellite 帧数一致）：57 条 drive，33,319 帧**，分布 09_26:42 / 09_28:5 / 09_29:3 / 09_30:4 / 10_03:3。
- 36 条 drive 有 `tracklet_labels.xml` 但**解析后全部 0 个 tracklet**（空文件）→ 无可用动态 3D 标注。
- 结论：**规模满足两周 MVP**（总方案 §9：5k–10k tuple、500–1000 地理独立 target），见 §4 tuple 统计。

### 1.2 KITTI-360（`/media/shizhm/Lenovo/KITTI-360`，未在本次指示范围，顺手盘点）

- 本机可用：8 条 drive（0000/0002/0003/0005/0006/0007/0009/0010），每条含 `calibration/ image_00/02/03 oxts/ poses.txt cam0_to_world.txt satellite/ train_frames.txt test_frames.txt`；drive_0000 有 11,518 张卫星图。
- **configs 里写的 data_root（`/media/user/...`、`/media/zhimiao/...`）本机不存在**，训练配置迁移时需统一改成本机路径。
- 其 `train_test_split_config.yaml` 也是 test_ratio=0.05 的时域切分 → 若后续用 KITTI-360，泄漏问题同样要重做划分。

### 1.3 KITTI-raw 与 KITTI-360 的管线差异（影响改造清单"保留卫星数据管线"项）

| 项 | KITTI-360（现有代码） | KITTI-raw（本次验证） |
|---|---|---|
| 位姿来源 | `poses.txt`（imu→world，官方 SLAM 结果）+ `calib_cam_to_pose.txt` | `oxts`（lat/lon/roll/pitch/yaw）→ ENU 位姿（KITTI devkit 公式，本次实现于 `kitti_raw_io.py`） |
| 前向相机 | image_00（rectified 1408×376，P_rect_00） | image_02（1242×375 彩色，P_rect_02；尺寸 per-drive 有差异） |
| 卫星 | 512×512 @ **0.196** m/px 各向同性，车心居中 north-up | 1280×1280 @ **(0.196, 0.128) m/px 各向异性**，车心居中 north-up 成立（校正后残差 ~1px） |
| 动态标注 | 3D semantic/instance 完整 | tracklet 全空，需替代方案 |

---

## 2. 代码盘点：对照 ICASSP27 §6 改造清单逐项定位

（行号基于当前代码库；`models/stage2/` 目录 ≠ 文档说的"Stage 2 两阶段训练"，后者指 anchor-view stage2 训练器，勿误删目录。）

### 2.1 保留项（均已确认存在、接口可用）

| 项 | 位置 | 状态 |
|---|---|---|
| VQ tokenizer（ImageNet f16/1024） | `models/stage1/maskgit/tokenizer.py:11`（encode/decode/decode_tokens）；分发器 `models/stage1/vqmodel.py:17`；权重 `ckpts/`（taming f16-1024/16384、maskgit imagenet、llamagen） | ✅ 直接复用；Emu3 Phase B 换码本走 `make_vqmodel` 扩展 |
| GPT decoder 骨架 + cross-attn + teacher-forcing | `models/stage2/simplified_token_predictor.py:1019`（`SimplifiedTokenPredictor`，`nn.TransformerDecoderLayer`×N + `CacheableAttention`:79 KV-cache）；训练循环 `world3d/train/trainer_ar.py:603-804` | ✅ 骨架保留；注意 `CacheableAttention` 同时承担 KV-cache，删 RayRoPE 不能整删该类 |
| e_pose 13 维 pose vector + MLP | 构造 `world3d/train/pose_ar.py:10`（rot6d 6 + t_cam−t_imu 3 + 归一化内参 4）；MLP `models/stage2/vanilla_components.py:44`（`VanillaPoseProjector`，输出单 token 拼在 memory 首位 `simplified_token_predictor.py:1552-1560`） | ✅ 与文档 §3.4 一致 |
| Fourier 坐标编码器 | `models/stage2/simplified_token_predictor.py:367`（`FourierCoordEncoder`，num_freqs 默认 10） | ✅ 存在但当前消费 IPM coords（与 BEV warp 同链路）——拆除 warp 时需把 coords 来源与编码器解耦，编码器本体保留 |
| 卫星检索/对齐 | `world3d/io/kitti360d_dataloader.py:583`（`<drive>/satellite/{frame}.png`）；`world3d/geo/sat_alignment.py`；`utils/geometry/kitti_transforms.py`、`camera_to_sat_projection.py` | ✅ 逻辑可复用；mpp 常量需按 §3 结论改 |
| 训练循环/配置体系 | `world3d/config.py` + `configs/*.yaml` + `trainer_ar.py:335-440`（data_root/drives/frames_file） | ✅ 框架保留，数据路径全部要改 |

### 2.2 删除项（范围已圈定）

| 项 | 位置 | 备注 |
|---|---|---|
| anchor routing（极坐标 anchors + 高斯位置偏置 + RBF 广播） | **活体**：`models/stage2/pose_aware_anchor_query.py`（整文件：learnable_anchors:56、polar 初始化:118、RBF broadcast:274-290）；接线 `simplified_token_predictor.py:1169-1183, 1529-1544`（hybrid）、`:1921-1936`（MaskGIT） | 注意 `simplified_token_predictor.py:1943-1956` 的 `PoseRouteCrossAttn`/`AnchorBasedSpatialFiLM` 调用是**不可达死代码**（mode 已限制），活体在 pose_aware_anchor_query.py；连带 `world3d/config.py:125-126` 的 `n_pose_queries/hybrid_memory_source`、trainer 可视化钩子、3 个 infer_* 脚本 |
| RayRoPE | `simplified_token_predictor.py:458`（`RayRoPEEncoder`）；注入点 `CacheableAttention:146-170`、self-attn 调用 `:1594-1607`、MaskGIT `:1961-1972`；数据依赖 `condition_tokens["K"]/["T_cam_to_world"]` | 文档提醒的"删后必补 1D PE"：`world3d/config.py:127` 已有 `use_explicit_token_pos` 备用开关可参考 |
| BEV unproject + F_bev 双线性采样 | 入口 `world3d/train/geometry_ar.py:11`（`compute_inverse_projection_view`）→ `utils/geometry/bev_to_camera_warp.py:203`；grid_sample 于 `simplified_token_predictor.py:1441-1450/1494-1501/1842-1848`；配套 `conditioning_ar.py`、`ar_pipeline.py:19`（BEV 可见性 mask）、`SemanticEncoder` | 删除时连同 trainer_ar.py:634-714 的 IPM warp 数据链一起拆 |
| Stage 2（mutual-NN + L_align + L_nce + 两阶段） | 训练器 `world3d/train/train_anchor_view_stage2.py`；损失 `consistency_loss.py:22`（InfoNCE:185-197）、`anchor_view_consistency_loss.py`；条件 `anchor_view_conditioning.py`；配对 `view_pairing.py`；配置 `configs/ar_anchor_view.yaml` 等 3 个 | trainer_ar.py:761-804 的 `use_consistency_loss` 挂钩也要关 |
| SatMAE（主通路移除） | `models/multiscale_vit_encoder.py:91`（fMoW ViT-L，权重 `ckpts/fmow_pretrain.pth` 3.95GB）；接线 `simplified_token_predictor.py:1189-1196`、`trainer_ar.py:247,700` | 保留为消融分支（文档 §7），只从主通路摘除 |

### 2.3 新增项（当前状态）

| 项 | 现状 |
|---|---|
| DINOv2 双分支 | **无**。DINOv2 仅存在于评测指标 `metrics/dino_similarity.py`（评测用，与编码器无关）→ 需新增 |
| 变长 source 拼接 + per-camera pose MLP | **无**。现有输入=单目标视图+单卫星图，pose 仅 1 个全局 token → 需新增 |
| Tuple 采样器（米制距离+Δyaw+window 共享 crop） | **无**。现有采样全部是同帧多视角（`FixedFiveViewDataset` 等），无跨帧米制采样 → 需新增（本次 V3 已验证可行性） |
| LiDAR depth 评测 | **无**。现有深度指标用 MiDaS 相对深度（`metrics/depth_consistency.py:43`）→ 需新增 velodyne 投影评测（本次 V1 的 `kitti_raw_io.py` 可作底子） |
| Cross-position warp 一致性评测 | **部分**。`metrics/multiview_consistency.py` 有 GT 深度 warp + masked DINO，但属图像对指标，非"真 pose+LiDAR depth warp 相邻 target"协议 → 需按文档 §5 重写 |
| 消融开关 `--sat_encoder/--geo/--use_sat/--use_src` | **无** → 新增 |
| **地理划分** | **无，且现状是时域 95/5**：`scripts/prepare_train_test_split.py:13`（每 drive 末 5% 作 test），本地 KITTI-360 的 split config 同样如此。这正是 review.txt 主题 B 指出的泄漏 → **必须重做**，V3 已给出 KITTI-raw 的地理连通分量方案 |

---

## 3. Day-0 验证结果（KITTI-raw）

### V1 oxts→位姿 + LiDAR→前向相机投影 —— ✅ PASS

4 条代表性 drive（0002/0009/0022/0034，覆盖 4 个采集日）：

- **位姿/尺度**：步长均值 0.59–1.04 m/帧（10 Hz），折算 21–38 km/h，p99 ≤ 1.29 m，**无 >30 m/s 跳变**（GPS 无毛刺）；由位姿矩阵反解的相对 heading 与 oxts yaw 差**均值 0.003–0.038°、最大 0.115°**——ENU 公式与 devkit 一致。
- **LiDAR 投影**（`P_rect_02 @ R_rect_00 @ velo2cam` 标准链）：像素覆盖率（5px 膨胀）44.8–53.9%，中位深度 8.5–16 m。**目检叠加图（`out/v1_lidar_*.jpg`）：点云边缘与车辆/道路/建筑轮廓精确锁定，深度渐变平滑，无系统错位** → Gate A 的"几何用 LiDAR 验证"在 KITTI-raw 上成立。
- 已知缺口：2011_09_26_drive_0009 velodyne 少 4 帧（447 vs 443），sampler 需容缺帧。

### V2/V2b 卫星对齐 —— ⚠️ 发现各向异性 bug（重要）

方法：对每条 drive 取 30–40 对帧（位移 15–120 m），相位相关测卫星图内容位移，与 oxts ENU 位移预测比。

| 模型 | KITTI-raw 1280px | KITTI-360 512px |
|---|---|---|
| 各向同性 0.2 m/px | 残差中位 6.9–28.0 px，p90 ≤ 44 px ❌ | 残差中位 1.55–1.84 px（可用但非最优） |
| 拟合各向同性 | m_u=m_v 矛盾 ❌ | m≈0.196，残差 0.68–0.73 px ✅ |
| 拟合各向异性 | **m_u=0.1958–0.1962, m_v=0.1272–0.1290，残差中位 0.90–1.14 px，p90≤1.82 px** ✅ | ratio=0.998（即各向同性）|

- KITTI-raw 各 drive 一致给出 **m_v/m_u = 0.6496–0.6583 ≈ cos(49.01°)=0.656**；m_u≈0.196 = Web Mercator zoom19 在 49°N 的地面分辨率。诊断：**下载管线把纬度方向跨度误乘了 cos(φ)**（典型 bbox 换算 bug），图像实际覆盖约 **251 m（东西）× 164 m（南北）**，不是正方形米制区域。
- 独立佐证：轨迹点 HSV 采样显示路面特征（d0034 高速路：轨迹处饱和度 36.7% 单调升至 +10m 处 81.5%，典型沥青→植被剖面）。
- **处置**：(a) world→pixel 映射按 (0.196, 0.128) 各向异性写（残差即 ~1px，零成本）；或 (b) 预处理把南北轴拉伸 ×1/0.656 重采样成各向同性再入库。**不建议**直接沿用 dataloader 硬编码的 512/0.2 假设。
- KITTI-360 侧：建议把代码常量从 0.2 修到 0.196（残差 1.8→0.7 px）。
- 覆盖检查：60 m window + 20 m target 外推 = ~80 m 范围，在 KITTI-raw 164–251 m 覆盖内、KITTI-360 ~100 m 内均放得下（后者余量小，crop 必须以 window 中心而非逐帧裁剪——与文档防泄漏要求一致）。

### V3 tuple 采样 + 地理切分 —— ✅ PASS（规模足够，切分必须按连通分量）

- **Tuple 规模**（K=3 source、间隔≥2 m、每 anchor 每 bin 1 个 target 的保守计数）：
  - extrapolation：[2,5)=9,249 / [5,10)=8,908 / [10,20]=8,374
  - interpolation：三 bin 各 ~8.3–9.2k
  - **Δyaw>20° 过滤只丢 9.1%**（转弯段少）→ 过滤规则可行
- **Window**：60 m 滑窗共 **388 个**（轨迹总长 24.8 km，中位 3 窗/drive）→ 支撑 window 共享卫星 crop 的采样单元。
- **地理切分**：以 drive 的 ENU bbox（+100 m margin）做重叠图 → **9 个连通分量**，最大 22 条 drive。**按日期切分不安全**（4 个大连通分量都横跨多个日期，同日多 drive 共享街道）。可行方案示例：c0(22 drive,12.5k tuples)+小分量→train；c2(9,17.7k)→val；c3(8,16.5k)→test，train/test 卫星 footprint 零重叠。
- 注意：bbox+margin 是保守判定（真实街道共享少于 bbox 交叠），实际泄漏风险只会更低。

---

## 4. Checklist §8 数据检查清单 · 逐项打勾（KITTI-raw 现状）

- [x] 卫星 north-up、固定 mpp —— north-up ✅、车心居中 ✅；**但 mpp 各向异性 (0.196, 0.128) ≠ 0.2，需修正后统一**（§3 V2b）
- [x] 每帧 pose→satellite pixel 变换核查 —— 相位相关残差 ~1px + 轨迹 HSV 路面检验通过
- [x] window 共享 crop —— 几何上可行（覆盖检查通过），dataloader 需按 window 分组（改造清单新增项）
- [x] train/val/test 按地理 footprint 去重 —— 连通分量方案给出（9 分量），**代码尚无，需新写**（现状 95/5 时域切分必须废弃）
- [x] source/target 不重复 —— 采样器设计约束，V3 计数时已保证 target 在 source 前方
- [ ] **动态车辆/行人屏蔽 —— KITTI-raw tracklet 全空 ❌**。替代：(a) LiDAR 时序一致性检测动态点；(b) 该检查项改在 KITTI-360（有 3D 实例标注）执行
- [x] 首轮 source 只用前向 perspective —— KITTI-raw 天然只有前向，无鱼眼捷径风险

### MM26 三个死因的对照现状

| 死因 | 现状 |
|---|---|
| Gate A（LiDAR 验证几何） | ✅ V1 已打通 KITTI-raw 投影链；评测工具（AbsRel/warp）待写 |
| Gate B（外部 baseline） | ❌ MASt3R/DUSt3R/VGGT/MVSplat 本机均无（无 repo、无 env、无权重）→ **Day-1 第一优先级：装 MASt3R + 权重，跑通 1 条 drive** |
| Gate C（盲区增益） | 协议就绪（V3 的 inter/extrapolation × 距离 bin 分层采样已验证可行），待 G0/G0+sat 训练 |

---

## 5. 结论与下一步（按 Day 排期对齐 checklist §7）

**判死/判活条件未触发任何一条（probe 尚未开始训练）；Day-0 管线复核完成，发现的都是可修问题：**

1. **[Day-1a] 装外部 baseline**：clone MASt3R（+ DUSt3R 备选）与权重，用 V3 的同一 tuple 协议在 1 条 drive 上跑通稀疏重建 → Gate B 前置。
2. **[Day-1b] 卫星预处理**：KITTI-raw 各向异性修正（推荐预处理重采样成各向同性 0.196，window 级 512×512 crop 重新出库）；KITTI-360 常量 0.2→0.196。
3. **[Day-1c] 地理切分落地**：把 V3 的 9 连通分量固化为 split yaml（train/val/test），替换 `prepare_train_test_split.py` 的 95/5。
4. **[Day-2] G0（ground-only）**：VGGT/MASt3R 稀疏重建 + ego-motion 尺度（ego-motion 尺度恢复本身 ego pose 已有，V1 位姿即可供），跑三 bin × inter/extra 几何指标。
5. **[Day-2b] 动态屏蔽替代方案**：LiDAR 时序一致性（KITTI-raw）或 KITTI-360 3D 标注投影，先在 10 个 window 目检。
6. **[决策点提醒]**：KITTI-raw 与 KITTI-360 的角色按总方案 §4 执行——KITTI-360（本机 8 drive + 3D 标注 + 各向同性卫星）做主验证集，KITTI-raw（57 drive）做外部验证/泛化集；两者 Day-0 管线本次均已打通或定位。

---

## 6. 第二轮（2026-08-16 下午）：数据决策变更 + 代码改造落地

用户指示：主数据改用 KITTI-360 前视；切分沿用 `~/codespace/CS2S_pose_environment/dataset/kitti_raw_sat_lidar_geofence_test2_buffer30` 的方法；按代码盘点执行改造。详见 `REFACTOR_NOTES.md`。此处只记结论与修正。

### 6.1 对第一轮结论的两处修正

1. **KITTI-raw tracklet 并非全空**：第一轮解析路径写错（XML 根为 `boost_serialization`，正确路径 `tracklets/item`）。实际 **36/36 条含 tracklet，共 1,212 个动态目标**（Car 913 / Van 95 / Pedestrian 84 / Cyclist 41 / Truck 22 / Misc 33 / Person(sitting) 16 / Tram 8）。动态实例屏蔽的缺口消除（若在 KITTI-raw 上评测）。
2. KITTI-360 侧补充：**8 条 drive 共享全局世界系**（poses.txt 直接跨 drive 比距离），地理簇 {0002,0003}(33.8m) / {0005,0006}(0.3m) / {0009,0010}(1.3m) / {0000} / {0007}；poses.txt 稀疏（~91% 帧覆盖）。

### 6.2 geofence 切分（替换 95/5 时域切分）

`scripts/make_geofence_split_kitti360.py` 复刻参考方法（test 路线 + 30m buffer 剔除）：
**train {0000,0005,0006,0009,0010}=42,261 帧 / val {0007}=2,890 / test {0002,0003}=14,678**；
buffer 剔除 0 帧（簇间距 1.9km），保留 train→test 最小距离 **1,892.9 m**。仅收录「精确位姿+卫星图存在」帧。产物在 `dataset_splits/kitti360_geofence_buffer30/`。

### 6.3 改造落地与验证（详见 REFACTOR_NOTES.md）

新增：tuple 数据集（28,738 基础 tuples，Δyaw>20° 剔 2,501≈8%）、`ICASSP27Predictor`（DINOv2 vitb14 冻结双分支 + metric PE + rel-pose add + 1D learned PE decoder；B0/B1/B2 开关可用；36.6M 可训练）、单阶段 CE 训练器 + `configs/icassp27_pilot.yaml`。「删除」项通过新 import 链不含旧模块实现（旧文件保留供复现与消融回接）。

验证结果：
- 冒烟测试全过（几何一致性 / 确定性 / memory token 账目 1+1369+2×666 / B0/B1 / generate）
- 真实训练 400 步：loss 7.03→6.35 单调降（ppl 1135→574），0.46 s/it @bs8（4090）
- **Phase A tokenizer oracle 上限：VQ-GT 重建 PSNR 仅 15–16 dB**（ImageNet VQGAN 在街景域）——doc 预期的"解读 pilot 数字天花板"已量化，数值上印证 Phase B 换 Emu3 并做域 finetune 的必要性
- DINOv2 hub 加载加了离线回退（网络抖动不再阻塞）

### 6.4 待办（Day-1+）

1. 评测工具：分 bin 推理脚本、KITTI-360 LiDAR→cam0 depth AbsRel、cross-position warp 一致性（doc §5 缺口）
2. 消融接线：`geo=rayrope/ipm`、`sat_encoder=satmae`（占位已留）
3. MASt3R/DUSt3R 外部 baseline 安装（Gate B 前置，仍未做）
4. B0/B1/B2 三配置正式 pilot 与 sanity 检查（B0 曲线应平、B1 近 bin 应优于 B0）
5. KITTI-360 动态实例屏蔽（3D 标注投影 + 10 window 目检）

---
### 附：验证产物索引
- `out/v1_lidar_*.jpg`：LiDAR 深度叠加目检图（12 张，4 drive × 3 帧）
- `out/v2b_overlay_d*.jpg`：卫星轨迹叠加图（KITTI-raw 各向异性模型）
- `smoke_test.py`：数据集+模型冒烟测试；`runs/icassp27_smoke400/`：400 步训练 ckpt 与可视化
- 脚本：`v1_pose_lidar.py`、`v2_sat_align.py`/`v2b_sat_fit.py`、`v3_tuple_split.py`、`kitti_raw_io.py`（KITTI-raw IO 底子）
- 第二轮改造：`REFACTOR_NOTES.md`（改造对照文档）、`scripts/make_geofence_split_kitti360.py`、`world3d/data/kitti360_tuple_dataset.py`、`world3d/models/icassp27_predictor.py`、`world3d/train/train_icassp27.py`、`configs/icassp27_pilot.yaml`

