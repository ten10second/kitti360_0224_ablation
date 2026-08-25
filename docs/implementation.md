# Implementation Guide — Dense-Ground-Defined Cross-Modal BEV Recovery
> 生成时间：2026-08-24 | 策略：基于现有项目改写 | 状态：APPROVED
> 依据：用户已批准的“首轮修改清单”与 `todo/kitti360_unified_bev_latent_implementation_spec.md`

## 1 改写范围

```text
world3d/unified_bev/
├── checkpoints.py                  # version/fingerprint/architecture/grid contract
├── geometry.py                     # relative height、support masks、target→BEV support
├── losses.py                       # masked/low-frequency/high-frequency losses
├── models.py                       # auditable completion contract；satellite prior
└── readouts.py                     # 唯一的 BEVHeightDecoder
scripts/
├── train_unified_bev_stage_a.py    # 联合训练并保存 geometry readout
├── train_unified_bev_stage_b.py    # observation-aware recovery losses
├── eval_unified_bev_probe.py       # observed/fill frozen geometry + appearance metrics
├── consistency_unified_bev_multichain.py # frozen-query C2
└── run_unified_bev_claim_probe.sh  # 唯一首轮运行链
tests/test_unified_bev.py           # height/support/freeze/output/loss 回归测试
```

| 文件 | 输入 | 输出 | 核心约束 |
|---|---|---|---|
| `geometry.py` | world points、BEV masks、target depth | relative height、obs/fill masks、target supported pixels | 空格不参与 quantile/loss；整体 Z 平移不改变相对高度 |
| `readouts.py` | `Z:[B,C,H,W]` | relative height `[B,1,H,W]` | 只在 Stage A 的 `Z*` 上训练，Stage B 永久冻结 |
| `models.py` | satellite prior、`Z_gnd`、coverage、`Ns` | `CompletionOutput` | `latent = Z_gnd + correction`；无 fake uncertainty |
| Stage A | dense-ground sources、target RGB/depth、LiDAR height | `ground/decoder/geometry_decoder` checkpoint | 删除无梯度 coverage loss |
| Stage B | frozen Stage A、satellite、sparse ground | satellite/completion checkpoint | 主损失按 observed/fill 与低/高频拆分 |
| evaluator | 同一 Stage A head 读取所有 latent 分支 | tile-level JSONL | 几何 headline 在 `M_fill`，整体 PSNR 为次要指标 |
| C2 | 两个互斥 source subsets | frozen height/depth/render consistency | raw latent L1 仅诊断 |

## 2 数据流与张量契约

```text
dense ground ── frozen/Stage-A encoder ──> Z* [B,64,128,128]
                    ├── ColumnFieldDecoder(query pose) ──> RGB + camera-z depth
                    └── BEVHeightDecoder ────────────────> relative height

sparse ground (front2 + left3 + right3 per frame)
              ── frozen encoder ──> Z_gnd, M_obs
satellite ────── satellite encoder ──> Z_sat
Z_sat + Z_gnd + M_obs ── completion ──> CompletionOutput
                                           ├── latent Z_hat
                                           ├── write_gate
                                           ├── correction
                                           └── ground_support

M_dense_height = M_dense_lift AND M_LiDAR_height_valid
M_fill = M_dense_height AND NOT M_obs
target_supported_pixels = project(target teacher depth to world XY)
                          then sample M_obs
```

## 3 函数级实现

### 3.1 `world3d/unified_bev/geometry.py`

- `relative_height_map(points_world, points_valid, origin_xy, resolution_m, height, width, *, quantile=0.1, min_height_m=-2.0, max_height_m=30.0) -> tuple[Tensor, Tensor, Tensor]`
  - 双线性聚合 world-Z；只在 covered cells 上逐 batch 求低分位 ground-Z；减去 ground-Z 后裁剪；返回 `[B,1,H,W]` relative height、bool valid mask、`[B,1,1,1]` ground-Z。
- `observation_partition(sparse_support, dense_geometry_support) -> tuple[Tensor, Tensor]`
  - 返回互斥 `M_obs` 与 `M_fill`；两者均为 bool。
- `target_pixels_supported_by_bev(depth_z, depth_valid, K, T_world_cam, origin_xy, tile_size_m, bev_support) -> Tensor`
  - 将 camera-z teacher depth 反投影到 world XY，以 nearest sampling 查询 `bev_support`；无效/出 tile 像素为 false。

### 3.2 `world3d/unified_bev/readouts.py`

- `BEVHeightDecoder(nn.Module)`：卷积金字塔 + coarse-to-fine fusion；名称不冒充 Transformer-token DPT。
- `freeze_module(module) -> nn.Module`：`eval()` 且所有参数 `requires_grad=False`，供 Stage B/eval/C2 共用。

### 3.3 `world3d/unified_bev/losses.py`

- `masked_smooth_l1(pred, target, mask) -> Tensor`：空 mask 返回与 `pred` 相连的零标量。
- `low_frequency(x, scale) -> Tensor`：平均池化得到布局/颜色低频。
- `low_frequency_l1(pred, target, scale) -> Tensor`。
- `high_frequency_masked_l1(pred, target, mask, scale) -> Tensor`：仅在 ground-supported target pixels 约束高频残差。

### 3.4 `world3d/unified_bev/models.py`

- Ground sparse/dense encoder 的高度均值通道统一改为 `relative_height_map`，variance 保持平移不变。
- `HeightMapSatellitePrior.forward(...) -> tuple[prior, h_pred]`，删除第三个全零 fake uncertainty；`h_pred` 使用统一 relative-height target，aux 默认权重 0。
- `CompletionOutput` dataclass 字段：`latent`、`write_gate`、`correction`、`ground_support`。
- `LatentCompletion.forward(...) -> CompletionOutput`；`conf` 改为直接可解释的 `write_gate`，记录实际 applied correction；`Ns == N_dense` 必须逐位返回 `Z_gnd`。

### 3.5 Stage A

- 构建 `BEVHeightDecoder` 并加入 optimizer。
- `geometry_loss = masked_smooth_l1(geometry_decoder(Z*), h_rel, h_valid)`。
- checkpoint 必须包含 `ground`、`decoder`、`geometry_decoder`，完整 architecture/grid
  config、schema version 与 SHA-256 fingerprint。
- 删除 `coverage_loss`；coverage 只记录日志。

### 3.6 Stage B

- 加载并冻结 Stage A 的 ground encoder、renderer、geometry decoder；三者架构只从
  Stage-A checkpoint 构建，不能由 Stage-B CLI 静默覆盖。
- Stage-B checkpoint 保存 Stage-A fingerprint；eval/C2 若收到另一份 Stage A 立即拒绝。
- 主损失：
  - `L_anchor`: `M_obs` 上 `Z_hat` 不偏离 `Z_gnd`；
  - `L_geo_fill`: `M_fill` 上 frozen height readout 对 relative LiDAR height；
  - `L_rgb_lowfreq`: 全图低频布局；
- `L_rgb_observed`: target supported pixels 上高频外观；support 使用 frozen
  `Z*` renderer 的稠密 teacher depth，并在 LiDAR-valid pixel 处以真值覆盖；
  - 全 tensor latent loss 仅保留为默认小权重 regularizer/诊断。
- Stage-B 训练随机采样 `Ns={1,2,4}`；`Ns=N_dense` 是无梯度的严格 identity，仅用于回归测试和 C1 评估。
- 删除 Stage-B `nadir_weight` 训练分支；direct satellite height auxiliary 默认 0。

### 3.7 Eval 与 C2

- 每个 latent 分支都通过同一 `geometry_decoder`，输出 all/observed/fill MAE、RMSE。
- `M_fill = M_dense-lift ∧ M_LiDAR-height ∧ ¬M_obs`；既不把无 metric height label 的 VGGT cell 用作监督，也不把 dense lift 未支持的 LiDAR-only cell 计入 headline。
- 输出 low-frequency PSNR、supported RGB PSNR/LPIPS、correction norm、write-gate mean、mask coverage 与 satellite condition。
- C2 比较同一冻结 height decoder、同一 target pose renderer 的 geometry/depth/low-frequency consistency；A-only/B-only/common/neither 分区；raw latent L1 仅使用 `_diag` 后缀。
- VGGT cache/eval/C2 逐 subset 输出 metric scale、scale source、有效 baseline 数、relative MAD、pose-alignment RMSE 与 reliability label。`Ns=1` 明确是 camera-rig fallback；只有多帧 subset 才称 vehicle-motion scale。
- VGGT cache v6 逐 tile 保存 `view_layout_version` 与 `(drive,target_fid,source_fids)`；每帧固定为 `front2+left3+right3=8 views`。多帧 scale 的 metric anchor 直接使用 source vehicle poses，predicted views 先按三台物理相机聚合；单帧 camera-rig scale 也先平均同一物理相机的 virtual views。主 target 为两个 calibrated front crops，renderer 输出 `(B,2,C,H,W)`，不再监督被压扁的整幅 `image_00`。Stage A/B attach、eval 与 C2 在读取张量前校验，禁止旧 source/target layout 或错误 split/tile 静默混入。

## 4 结果文件

- Stage A：`runs/.../stage_a.pt`，含三组权重、schema、完整架构/网格配置、geometry-target version 与 fingerprint。
- Stage B：`runs/.../stage_b.pt`，含 `satellite_encoder`、`completion`、Stage-A 路径/fingerprint 与 observation-aware loss config。
- Eval JSONL：每行一个 `(drive,target_fid)`；包含 condition、Ns、RGB/depth、frozen height all/obs/fill、support、gate/correction 指标。
- C2 JSONL：每行一个 tile；包含每个 Ns/family/partition 的 frozen height consistency、target depth consistency、low-frequency render consistency及 latent diagnostic。

## 5 实现与验证顺序

```text
relative height/support helpers + tests
→ readout/completion contract + tests
→ Stage A checkpoint
→ Stage B losses
→ evaluator
→ C2
→ single claim-probe run script
→ full unit tests + CPU smoke + optional GPU one-step smoke
```

## 6 设计校验

- ✅ 实验要求覆盖：首轮清单的 geometry/support/shared-interface/causal-control 均有对应代码路径。
- ✅ 逻辑一致性：所有 BEV tensors 使用 `[B,C,H,W]` south-up pixel-center；height/support 使用 `[B,1,H,W]`。
- ✅ 完整性：所有计划修改文件均有函数级职责；旧四-head DPT、adapted decoder、post-hoc probe、nadir training 与 date-stamped alternate chains 已从仓库删除。
- ✅ 兼容性：旧 checkpoint 被显式拒绝；ground family、网格与 Stage-A/Stage-B 配对均有 fail-fast 校验。
