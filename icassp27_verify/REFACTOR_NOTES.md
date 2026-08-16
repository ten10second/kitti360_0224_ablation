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

## 5. 「删除」项的落地方式（对照 §6 删除清单）

| 删除项 | 新框架中的状态 |
|---|---|
| anchor routing（极坐标 anchors/高斯偏置/RBF 广播） | **不在新 import 链**（`icassp27_predictor.py` 不 import `pose_aware_anchor_query`） |
| RayRoPE | 同上；替换为标准 1D learned PE |
| BEV unproject + F_bev 双线性采样 | 同上（`compute_inverse_projection_view`/`warp_bev_to_camera*` 不再被训练路径调用） |
| Stage 2（mutual-NN/L_align/L_nce/两阶段） | 训练器只有单循环 CE，`consistency_loss`/`anchor_view_*` 不被 import |
| SatMAE 主通路 | `sat_encoder=satmae` 未接线（NotImplementedError 占位），DINOv2 为主 |

旧文件（`trainer_ar.py`、`simplified_token_predictor.py`、`pose_aware_anchor_query.py`、`train_anchor_view_stage2.py` 等）原样保留——供旧 runs 复现与后续 `geo=rayrope/ipm`、`sat_encoder=satmae` 消融行回接。

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
