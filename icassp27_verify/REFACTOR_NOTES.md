# ICASSP27 代码改造落地记录（对照 ICASSP27_method_framework.md §6）

> 日期：2026-08-16
> 数据决策（用户指示）：主数据改用 **KITTI-360 前视**（本机 `/media/shizhm/Lenovo/KITTI-360`，8 drive）；
> train/test 切分沿用 `~/codespace/CS2S_pose_environment/dataset/kitti_raw_sat_lidar_geofence_test2_buffer30`
> 的方法（`remove_train_frames_within_gps_buffer_of_test_route`，buffer 30 m）。
> 环境同前（conda `maskgit`）。

---

## 0. 新增文件总览

| 文件 | 角色 |
|---|---|
| `scripts/make_geofence_split_kitti360.py` | geofence 切分脚本（复刻参考方法） |
| `dataset_splits/kitti360_geofence_buffer30/{train,val,test}_manifest.jsonl + manifest_stats.json` | 切分产物 |
| `world3d/data/kitti360_tuple_dataset.py` | Tuple 采样数据集（新增组件） |
| `world3d/models/icassp27_predictor.py` | 新模型（DINOv2 双分支 + decoder） |
| `world3d/train/train_icassp27.py` | 单阶段 teacher-forcing 训练器 |
| `configs/icassp27_pilot.yaml` | Phase A pilot 配置（B2；B0/B1 用命令行开关） |
| `icassp27_verify/smoke_test.py` | 数据集+模型冒烟测试 |

**改造策略说明**：新框架以**新文件**落地，不复用 `trainer_ar.py`/`simplified_token_predictor.py` 的旧文件改写——旧路径（anchor routing、RayRoPE、BEV warp、Stage-2 损失、SatMAE 主通路）**不进入新 import 链**即为"删除"；旧文件保留以复现 MM26 旧实验与消融行复用（`geo=rayrope/ipm`、`sat_encoder=satmae` 需要回接旧模块时再接线）。这避免了破坏 `runs/` 下旧结果的复现能力。

---

## 1. 切分（新增，替换 95/5 时域切分）

`scripts/make_geofence_split_kitti360.py` 复刻参考实现的语义：指定 test 路线 → train/val 中剔除距 test 路线 **30 m** 内的帧 → 输出逐帧 manifest + stats。

KITTI-360 8 drive 共享全局世界系（Day-0 已验证），地理簇：{0002,0003}（33.8m）、{0005,0006}（0.3m）、{0009,0010}（1.3m）、{0000}、{0007}。指派：

- **test = {0002, 0003}**（14,678 帧可用，连续城市区域）
- **val = {0007}**（2,890 帧，孤立区域，与 test 距离 1.8km+）
- **train = {0000, 0005, 0006, 0009, 0010}**（42,261 帧）
- buffer 剔除 0 帧（簇间距 1.9km ≫ 30m），**保留 train→test 最小距离 1892.9 m**；机制与参考实现一致，若换 test 路线自动生效。
- 仅收录「poses.txt 精确位姿 + 卫星图存在」的帧（KITTI-360 poses.txt 稀疏，~91% 覆盖）。

产物 schema 与参考一致（drive / frame_index / world_x / world_y / split / 各传感器路径 / nearest_test_route_distance_m / geofence_removed）。

## 2. Tuple 采样器（新增，doc §4）

`world3d/data/kitti360_tuple_dataset.py`：

- **窗口**：每 drive 弧长上 60m 滑窗（步进 30m）；窗口中心帧的卫星图（512×512@0.196，车心 north-up）作为**该窗口所有 tuple 共享的 crop**（防 crop-shift 泄漏）；窗口中心世界坐标 = window 局部系原点。
- **anchor**：窗内每 4m 一个候选 anchor；K∈{1,2,3} 个 source，相邻 source 弧长间距 ≥2m（chord 校验）。
- **target**：距最后一个 source 沿弧长 [2,20]m **均匀采样**（训练随机、评测按 bin 中值 3.5/7.5/15m 确定性）；extrapolation 构造（target 恒在所有 source 前方）。
- **Δyaw 过滤**：anchor→target heading 差 >20° 剔除（构建期剔除 2501/31239 ≈ 8%）。
- **位姿**：优先 `cam0_to_world.txt`（cam0=前视 rectified），fallback `poses.txt @ calib_cam_to_pose[image_00]`；e_pose 13 维沿用 `build_pose_vec`（保留项）。
- **rel-pose**：每 source 的 `Δt(3)+rot6d(6)`（target→source，世界系平移即可，窗口系只差原点平移）。
- 确定性：`make_rng(seed, epoch, idx, salt)` 复用现有机制；DDP/worker 安全。
- 规模：train split 构建 **28,738** 个 (anchor×bin) 基础 tuple（训练时每条随机 K 和距离，实际采样空间更大）。

冒烟验证（`smoke_test.py`）：5 个随机样本 req_d vs 实测距离差 ≤0.4m，source span ≥2m，dyaw<2°；确定性重复调用 bit-exact。

## 3. 模型（新增，doc §2/§3）

`world3d/models/icassp27_predictor.py`：

| doc 条目 | 实现 |
|---|---|
| §3.1 tokenizer | 不入模型，trainer 持有 `PretrainedTokenizer`（f16/1024，冻结）——Phase A 权重路径 `ckpts/maskgit-vqgan-imagenet-f16-256.bin`（保留项原样） |
| §3.2 卫星分支 | `DinoV2Encoder`（vitb14，冻结）→ `sat_proj` + `MetricPE`（Fourier num_freqs=10 + MLP，**逐 token 相加**；patch 中心世界 (x,y) 由窗口原点+网格+mpp 0.196 直接算出）。无 BEV 采样、无显式对齐 |
| §3.3 街景分支 | 同一 DINOv2 权重（共享）；`src_proj` + `RelPoseProjector`（9→D）加到该 source 全部 tokens；变长 K 顺序拼接，padding 由 `src_mask` → cross-attn `memory_key_padding_mask` |
| §3.4 e_pose | 复用 `VanillaPoseProjector`（13 维 → 1 token，memory 首位） |
| §3.5 decoder | `nn.TransformerDecoderLayer` ×N（norm_first, batch_first），causal self-attn + cross-attn to M；**标准 1D learned PE**（`nn.Parameter`，RayRoPE 删除后按 doc 坑位#1 补上）；teacher forcing `make_teacher_forcing`；`generate()` 温度+top-p 采样写死策略 |
| 消融开关 | `use_sat`/`use_src`（B0/B1/B2）、`sat_encoder ∈ {dino, satmae}`（satmae 抛 NotImplemented，留接线点）、`geo`（pose_add 已实现；rayrope/ipm 为消融行，未接线） |

**与文档的一处偏差（记录在案）**：DINOv2 官方只有 patch-14（无 /16）。卫星 512→518×518（37×37=1369 tokens），街景 640×256→518×252（37×18=666 tokens/视图）。token 数与 doc 的 1024/16-patch 假设不同但量级一致，cross-attn 可承受（K=3 时 memory = 1+1369+1998 = 3368）。

## 4. 训练器与配置（新增，doc §4）

`world3d/train/train_icassp27.py` + `configs/icassp27_pilot.yaml`：

- **单阶段端到端**，Loss = token CE（label_smoothing 可配），无第二损失。
- 可训练：decoder blocks、token/pos embed、sat/src_proj、MetricPE、RelPose、e_pose MLP（共 36.6M）；DINOv2 与 VQ 冻结（123.2M total）。
- AdamW + warmup-cosine + grad clip；每 epoch `dataset.epoch += 1` 重采样 K/距离。
- 可视化：teacher-forced argmax 重建 vs VQ-GT decode 拼图（沿用 legacy 约定）。
- 运行方式：
  - B2（主）：`python -m world3d.train.train_icassp27 --config configs/icassp27_pilot.yaml`
  - B1：`--use_sat false`；B0：`--use_src false`
- 实测吞吐：bs=8，0.50 s/it（RTX 4090，含 DINOv2 前向）。

## 5. 「删除」项的落地方式（对照 §6 删除清单，已实际执行）

git：快照 `a32d6a8`（改造前完整状态）→ 删除提交 `adf92fc`（**62 files，-19,730 行 / +186 行**）。任何删除可 `git checkout a32d6a8 -- <path>` 找回。

### 5.1 整文件删除（51 个）

| 删除项 | 文件 |
|---|---|
| anchor routing | `models/stage2/pose_aware_anchor_query.py`、`models/stage2/simplified_token_predictor_bk.py`（含活体 PoseRouteCrossAttn/FiLM 的备份） |
| Stage-2 全部 | `world3d/train/train_anchor_view_stage2.py`、`consistency_loss.py`（mutual-NN + L_align + L_nce）、`anchor_view_consistency_loss.py`、`anchor_view_conditioning.py`、`view_pairing.py`、`scripts/infer_anchor_view_stage2.py`、configs `ar_anchor_view / ar_direct_consistency / ar_hybrid_enhanced_consistency` |
| BEV 地面 unproject + F_bev 通路 | `world3d/train/geometry_ar.py`、`world3d/train/conditioning_ar.py`、`utils/geometry/bev_to_camera_warp.py`、`camera_to_sat_projection.py`、`camera_to_camera_ground.py`、`differentiable_projection.py`、`pose_loss.py`、`homography.py`、`utils/losses/`（geometric_loss/dcn_loss，无引用者）、`world3d/models/bev_downsample.py` + IPM 可视化工具（`tools/vis_warp_from_dataloader.py`、`vis_kitti360d_bev_fov.py`、`vis_fixed_five_views_ipm.py`、根目录 `debug_ipm_alignment*.py`） |
| RayRoPE 备份/死代码 | （类本体在 predictor 重写中删除）`world3d/models/ray_coordinate_encoder.py`、`direct_predictor_modules.py`（均无引用者） |
| MM26 训练/推理栈 | `scripts/infer_{yaw_sweep,yaw_sweep_batched,direct_yaw_sweep,vanilla_ar,vanilla_yaw_sweep,test_frames,anchor_view_stage2}.py`、`scripts/evaluate_metrics{,_optimized}.py`（依赖已删模块）、trainer/train `{maskgit,oneslot}` 四件、configs `ar / ar_direct / ar_hybrid{,_anchor,_enhanced} / ar_oneslot{,_warmstart} / maskgit` |
| 替换掉的 95/5 切分 | `scripts/prepare_train_test_split.py`（MM26 killer 2 泄漏源；由 `make_geofence_split_kitti360.py` 替代） |

### 5.2 文件内删除/重写

| 文件 | 改动 |
|---|---|
| `models/stage2/simplified_token_predictor.py` | **重写为 vanilla-only**（~2000→~330 行）：删 RayRoPEEncoder/RayDirectionEncoder/PoseRouteCrossAttn/AnchorBasedSpatialFiLM/PoseAwareAnchorQuery 接线/SemanticEncoder/BEV grid_sample/direct+hybrid 模式；保留 GPT 骨架（causal self-attn + cross-attn）、CacheableAttention（仅 KV-cache，RoPE 分支删除）、topleft/bottomup teacher-forcing、e_pose（VanillaPoseProjector）；mode≠vanilla 直接抛错 |
| `world3d/train/trainer_ar.py` | 剥离：IPM warp 块、BEV 编码/SatMAE 构建、BEV 可见性 mask、一致性损失块（use_consistency_loss）、anchor-view resume 容错与可视化钩子；condition_tokens 只剩 pose；loss_history 3 元组 |
| `world3d/config.py` | 删字段：mode/fourier_freqs/train_bev_encoder/no_bev_pretrain/n_pose_queries/hybrid_memory_source/use_ipm_semantic/use_explicit_token_pos/semantic_dim、pair-consistency 块、consistency 块、anchor-view 块（16 字段）、MaskGIT/OneSlot 块 |
| `world3d/data/ar_pipeline.py` | 删 BEV mask 对 geometry_ar/conditioning_ar 的依赖；`compute_bev_visibility_mask` 重写为自包含 helper（diffusion 分支仍用，不属 AR 删除目标） |
| `configs/ar_vanilla.yaml` | drives 改为 geofence-safe（剔除 test {0002,0003} / val {0007}）、data_root 指本机、num_workers=0（本机 cv2-fork 规避） |
| `world3d/train/train_ar.py` | 默认 config `configs/ar.yaml`→`ar_vanilla.yaml` |
| 各 `__init__.py` | `models/stage2`（去 PoseAwareAnchorQuery）、`world3d/models`（清空死导出）、`utils/geometry`（去 6 个 BEV 族模块导出） |
| `scripts/run_metrics_eval.py` | SegAnyConsistency 改可选 import（**改造前即坏**，git 快照验证；顺手修复） |

### 5.3 按清单"保留为消融分支"的

- `models/multiscale_vit_encoder.py`（SatMAE/fMoW ViT-L）+ `ckpts/fmow_pretrain.pth`：文件保留，主通路已无引用；未来 `--sat_encoder satmae` 消融行回接。
- diffusion 分支（`trainer_diffusion.py`、`diffusion_model.py` 等）：不在删除清单，原样保留并验证可 import。

### 5.4 删除后验证

- 全部存活入口 import OK（新栈 3 + legacy vanilla 5 + diffusion 3 + 共享模块 8 + metrics 2）
- 新栈 30 步真实训练 OK（28,738 tuples，36.6M 参数）
- legacy vanilla 2 步真实训练 OK（geofence-safe 5 drives，201,160 五视图样本）
- vanilla predictor 单测：logits 形状、backward、KV-cache 增量步、bottomup 序、mode 守卫（direct 拒绝）
- 环境补装 tensorboard/pytorch-fid（requirements 内列出但缺失）

### 5.5 已知行为差异（非缺陷）

- legacy trainer 多 worker（cv2-fork）在本机 abort——**改造前即存在**（数据管线未动），ar_vanilla.yaml 已固化 num_workers=0。
- 旧 MM26 runs/ 的 ckpt 与新 predictor 不兼容（state_dict 键不同）——旧结果复现请用快照 `a32d6a8`。

## 6. 已知限制 / 待办（进入 Day-1+）

1. **评测工具未写**：分 bin 评测脚本、LiDAR depth AbsRel（KITTI-360 velodyne 投影到 cam0 需按 KITTI-360 标定链）、cross-position warp 一致性——doc §5 的代码缺口仍开放。
2. `geo=rayrope/ipm` 消融行未接线（需回接 `RayRoPEEncoder`/BEV warp 旧模块）。
3. `sat_encoder=satmae` 未接线（权重在 `ckpts/fmow_pretrain.pth`）。
4. 推理/评测入口（AR 采样 → 分 bin 图像 + 指标）未写；`generate()` 已具备单 batch 采样能力。
5. KITTI-360 动态实例屏蔽（3D 标注投影）未实现——评测阶段做，先在 10 个 window 目检（doc 坑位#5）。
6. Phase B（Emu3 tokenizer）按 doc 走接口不变换权重的路线，未开始（属计划内后置）。

## 7. 验证记录

- `icassp27_verify/smoke_test.py`：数据集几何一致性（5 样本）、确定性、batch collate、模型 forward/backward、memory token 账目（1+1369+2×666=2702 精确）、B0/B1 开关、`generate()` —— **全部通过**。
- 60 步真实训练：loss 7.03（warmup 中 lr 5e-6），无 NaN，0.5s/it。
- 400 步趋势验证：`runs/icassp27_smoke400`（见 VERIFY_LOG 更新）。

---

## 8. 追加（2026-08-16 晚）：target raymap 条件化（geo 注入修正）

**动机（架构审查发现的真实缺陷）**：e_pose 的 t_cam−t_imu 是 MM26"车心转 yaw"时代的安装偏移（~1m），不携带沿路线位置；source rel-pose 只有相对量；卫星 MetricPE 只标记卫星 token。三者组合对"整条 tuple 在 60m 窗口内平移"**不变**，而窗口共享同一张卫星 crop → B0 原理上无法知道目标在窗口哪里，B2 只能靠隐式地标匹配。这是与主流做法（LVSM/CameraCtrl 的 Plücker raymap；3DiM 的相对位姿 attention）对照后确认的设计缺口。

**改动**（对照 ICASSP27 文档 §3.5/§7 的 geo 消融行，把已删除的 rayrope/ipm 替换为主流选项）：

| 文件 | 改动 |
|---|---|
| `world3d/models/icassp27_predictor.py` | 新增 `TargetRayPE`（每 token 6 维 (o,d)→MLP→D，加到 target token 嵌入；BOS 位 PE 置零）；`_target_rays()`：o = 相机中心在**窗口局部系**的坐标（t_cam − origin_xyz），d = R·K⁻¹(u,v,1)/‖·‖（40×16 token 网格的像素中心）；`geo` 开关 `{raymap, pose_add, proj}`，默认 raymap，proj 为 PVSM 式占位 |
| `world3d/data/kitti360_tuple_dataset.py` | `window_origin_xy`(2D) → `window_origin_xyz`(3D，窗口中心帧 IMU 全坐标)，collate 同步 |
| `world3d/train/train_icassp27.py` + `configs/icassp27_pilot.yaml` | 传 `tgt_K`/`tgt_T_cam`/`window_origin_xyz`；配置 `model.geo: raymap` |
| （顺手）`DinoV2Encoder` | hub 加载改为**缓存优先**（此前网络慢时 try 分支可挂起数分钟） |

**为什么用窗口局部原点**：① 卫星 MetricPE 同为窗口系，query/key 同规范可直接互查（"我的射线打到卫星哪个 patch"）；② 避开 PVSM（CVPR 2026）指出的绝对世界 gauge 问题；③ 数值恒在 ±50m，对 MLP 友好。

**验证**：
- 平移不变性破坏测试：目标 +30m → ray 原点均值偏移 30.0m，ray PE 逐 token |diff| = 521（≫0）——B0 条件化不再平移不变 ✅
- 冒烟全过（B0/B1×raymap、pose_add 消融行、generate）；30 步训练回归 OK（36.7M 可训练）
- 400 步趋势（同 warmup=50 条件）：raymap 6.35→**6.44** vs pose_add **6.35**；400 步属 warmup 后极早期，不做优劣结论，仅确认可训练、无 NaN、吞吐持平（0.46 vs 0.46 s/it）。真正的 geo 消融判断留给正式 pilot。
