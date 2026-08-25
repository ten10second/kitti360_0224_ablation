# 开发日志 — Unified BEV Claim-Aligned Refactor
> 创建时间：2026-08-24 | 最后更新：2026-08-24
> 关联实现指南：`docs/implementation.md`
> 本文件只追加，不删除。

## 项目概览

| 项目 | 内容 |
|---|---|
| 研究方向 | dense-ground-defined frozen world interface + satellite/sparse-ground recovery |
| 实现策略 | 基于现有 KITTI-360 unified BEV probe 原位改写 |
| 框架 | PyTorch |
| 数据 | KITTI-360 + registered satellite + LiDAR/VGGT geometry |
| 运行策略 | 自动跑快速测试；长训练在代码 gate 后运行 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|---|---|---|---|---|
| 实现契约 | `docs/*.md` | ✅ Done | 2026-08-24 | 用户已批准首轮清单 |
| 几何/support | `geometry.py` | 🔄 WIP | — | relative height 与 target support |
| 冻结 readout | `readouts.py` | ⬜ TODO | — | Stage-A-only geometry interface |
| Completion | `models.py` | ⬜ TODO | — | auditable output，无 fake uncertainty |
| Stage A | `train_unified_bev_stage_a.py` | ⬜ TODO | — | 保存 geometry decoder |
| Stage B | `train_unified_bev_stage_b.py` | ⬜ TODO | — | observation-aware losses |
| Evaluator | `eval_unified_bev_probe.py` | ⬜ TODO | — | observed/fill frozen metrics |
| C2 | `consistency_unified_bev_multichain.py` | ⬜ TODO | — | frozen-query consistency |
| 主运行链 | `run_unified_bev_claim_probe.sh` | ⬜ TODO | — | held-out Ns=1,2 causal probe |
| Tests | `tests/test_unified_bev.py` | ⬜ TODO | — | targeted + smoke |

## 开发日志

### 2026-08-24 — 固化首轮实现契约

- **完成内容**：将用户批准的修改清单固化为函数级实现、数据流、结果字段与验证顺序。
- **遇到的问题**：项目没有 ResearchPilot 的 `docs/implementation.md` 与 `docs/user_requirements.md`，且当前工作树已有大量相关未提交修改。
- **解决方案**：把现有工作树作为不可回退基线；只追加 claim-aligned 改动，不覆盖 VGGT、双 crop、普通 splat 等已确认修改。

## 已知问题

- [ ] 旧 Stage-A checkpoint 不含 `geometry_decoder`，不能用于新的主证据链，需重训。
- [ ] 旧 Stage-B checkpoint 使用 tensor-only completion 输出与 `conf` 参数名，只作为历史结果保留。
- [ ] 完整 20k/10k 训练尚未运行。

## 运行说明

### 快速验证

```bash
conda run --no-capture-output -n maskgit python -m pytest tests/test_unified_bev.py -q
```

- 验证 south-up/pixel-center、relative height、support partition、冻结 geometry head、completion identity/output 与 observation-aware loss。

### 首轮 claim probe

```bash
bash scripts/run_unified_bev_claim_probe.sh
```

- 在 VGGT joint-view geometry cache 上训练新的 Stage A 与 residual/coordinate-only Stage B，并在 held-out drive 上运行 aligned、random-tile、cross-road 5m 与 C2 frozen-query 评估。

### 2026-08-24 — 建立统一几何与冻结 readout 基础接口

- **完成内容**：新增 `relative_height_map`、observed/fill partition、target-pixel support 投影、mask-aware low/high-frequency losses、`BEVHeightDecoder` 与 `freeze_module`；ground encoder 高度均值输入改为 local relative height。
- **完成内容**：`HeightMapSatellitePrior` 删除全零 uncertainty 占位输出；`LatentCompletion` 改为 `CompletionOutput(latent, write_gate, correction, ground_support)`，并保留 dense-source bitwise identity。
- **遇到的问题**：系统 PATH 没有裸 `python` 命令。
- **解决方案**：后续统一使用 `conda run --no-capture-output -n maskgit python`；四个基础模块已通过 `py_compile`。

### 2026-08-24 — 完成首轮 claim-aligned 改写

- **完成内容**：Stage A 现在联合训练唯一的 `BEVHeightDecoder`，checkpoint 保存完整架构、网格、geometry-target version 与 SHA-256 fingerprint；Stage B 只能从该 checkpoint 重建并冻结 ground encoder、RGB/depth renderer 和 geometry readout，且 checkpoint 与 exact Stage A fingerprint 绑定。
- **完成内容**：Stage B 改为 observed latent anchor、fill-region frozen geometry、全图低频 RGB、ground-supported 高频 RGB 与小权重 latent regularizer；删除无梯度 coverage loss、训练期 nadir loss、fake uncertainty 和可学习逐格 coordinate table。
- **完成内容**：evaluator 与 C2 改用同一 frozen geometry/render interface；headline 几何指标按 observed/fill 分区，raw latent 只保留 `_diag`；新增 ground/fixed-XY/aligned/random/cross-road controls。
- **完成内容**：VGGT 使用官方 joint forward 与 exact-subset cache；修复把真实 VGGT confidence 全部压成零覆盖的问题，改为逐视图 log-evidence q10/q90 归一化；普通 bilinear mean splat 保持不变。
- **尺度边界**：`Ns=1` 不存在帧间车辆运动，只能使用 calibrated camera rig，标记为 `single_frame_camera_rig_fallback`；`Ns=2` 只有一个 motion baseline。eval/C2 每条记录均保存 scale source、reliability、pair count、relative MAD 与 pose RMSE。
- **退役内容**：旧 date-stamped 主链与 post-hoc 四-head DPT、adapted decoder、nadir-training、raw-latent C2 runner 已从主证据链退出；历史 Python 文件保留时会 fail-fast 提示，避免误跑。

### 2026-08-24 — 验证结果

- **单元测试**：当前环境未安装 `pytest`，改用等价函数 runner 执行全部 42 项，42/42 通过；本机已有 LPIPS AlexNet 与 v0.1 权重，identity test 也已通过。
- **静态检查**：`compileall` 覆盖 `scripts/`、`world3d/unified_bev/` 和测试文件；7 条相关 shell 链通过 `bash -n`；`git diff --check` 通过。
- **真实数据 smoke**：在 KITTI-360 drive 0003、真实 `/home/shizhm/Downloads/vggt.pt` 与一 tile exact-subset cache 上完成 Stage-A/Stage-B one-step、eval 与 C2。审查修正后的 evaluator 中 dense coverage 为 `0.4843`、dense-lift∧height-label support 为 `0.3389`、Ns=1 sparse coverage 为 `0.0170`、fill fraction 为 `0.3220`、target-supported fraction 为 `0.3049`；这些数字只证明数据流和非空 mask，不作为效果结论。
- **尺度审计 smoke**：dense Ns=8 使用 28 个 vehicle-motion baselines，scale=`19.8788`、relative MAD=`0.0951`；sparse Ns=1 明确使用 camera-rig fallback，scale=`2.7649`、relative MAD=`0.3077`。eval 与 C2 JSONL 均已验证字段真实落盘。
- **checkpoint smoke**：Stage-A replay 为逐位一致；Stage-B/评估/C2 的 fingerprint mismatch 会 fail-fast；旧 schema checkpoint 被拒绝。

### 2026-08-24 — 独立审查修正

- **M_fill 契约**：修正 Stage B/eval 中仅使用 LiDAR-valid mask 的偏差；现在统一为 `dense-lift support ∧ LiDAR-height-valid ∧ ¬sparse-support`，并新增 dense lift 与 height label 不一致时的回归测试。
- **训练计步**：`Ns=N_dense` 是 bitwise identity 且没有 Stage-B 梯度，现从训练 source choices 删除；Stage B 训练 `{1,2,4}`，C1/identity 评估仍覆盖 `{1,2,4,8}`。
- **配对评估**：默认指标切换为 claim-aligned fill geometry/low-frequency/supported RGB；缺失请求字段默认 fail-fast，只有显式 `--allow_missing` 才兼容历史记录。
- **cache 身份**：VGGT cache 升级为 v4，并保存 `(drive,target_fid,source_fids)`；训练 attach、eval 与 C2 均在使用 geometry 前 fail-fast 校验，旧无身份 cache 必须重建。
- **v4 真实 smoke**：一 tile 独立构建 `0:{1,2,4,8}` 与 `4:{1,2}` 六个 VGGT subsets，随后 Stage-B one-step、evaluator 和 C2 `Ns={1,2}` 全部通过身份校验并输出 scale QA。
- **最终复审**：无 blocker/high issue；修正 comparator 对“字段存在但无 finite pair”仍返回成功的问题，并把 lower-is-better 指标的 win count 统一改为 `A_better` 语义。

## 最终状态

| 模块 | 状态 | 验证 |
|---|---|---|
| relative height / support | ✅ Done | translation、empty-mask、真实 KITTI 样本 |
| shared frozen geometry readout | ✅ Done | freeze/schema/replay tests |
| Stage A / Stage B | ✅ Done | one-step real-data smoke |
| evaluator / C2 | ✅ Done | one-tile real-data JSONL smoke |
| VGGT exact-subset geometry | ✅ Done | GPU forward、cache augment、scale/confidence audit |
| claim probe launcher | ✅ Ready | `bash -n`；20k/10k 长训练未启动 |

## 已知剩余项

- 完整 Stage A 20k + Stage B 10k、32-tile held-out probe 尚未运行；因此当前只能判定“实现 gate 通过”，不能判定论文 claim 已通过。
- `Ns=1` 的 VGGT geometry 不是 vehicle-motion-scaled；它只能作为 camera-rig-scaled extreme-sparsity diagnostic。若论文要求所有设置都由车辆运动定标，headline 应从 `Ns>=2` 开始。
- `Ns=2` 只有一个 motion baseline，MAD 为形式上的零而非稳健性证据；可靠性判断必须同时查看 pair count。
- LPIPS/SSIM 会显著增加评估耗时；可用 `WITH_PERCEPTUAL=0` 跳过，不影响 frozen geometry headline。

## 运行说明（最终）

快速离线回归（当前环境未安装 pytest，因此使用函数 runner）：

```bash
conda run --no-capture-output -n maskgit python - <<'PY'
import runpy
namespace = runpy.run_path('tests/test_unified_bev.py')
for name in sorted(namespace):
    if name.startswith('test_'):
        namespace[name]()
PY
```

完整首轮 held-out claim probe：

```bash
WITH_PERCEPTUAL=0 bash scripts/run_unified_bev_claim_probe.sh
```

启用 LPIPS/SSIM（需要对应依赖与权重）：

```bash
WITH_PERCEPTUAL=1 bash scripts/run_unified_bev_claim_probe.sh
```

### 2026-08-25 — 删除非主链历史代码

- **删除范围**：本轮移除 10 条旧 date-stamped/day/night shell chains、仅服务旧 ICASSP pilot 的 checkpoint watcher，以及 free-latent、旧 VGGT aspect、decoder-gain、nadir-roundtrip、旧 render、四-head false-DPT、post-hoc height probe、vertical stratification、adapted decoder 共 9 个 superseded Python runners；计入此前已删除的 dense-consistency chain，当前 Git diff 共删除 21 个脚本、2606 行。
- **保留范围**：唯一主链 `run_unified_bev_claim_probe.sh`；Stage A/B、eval/C2；VGGT cache 与当前 geometry gate；数据/split/卫星检查、south-up alignment QA、smoke/replay；Metric3D builder 作为仍可用的几何 baseline。
- **fallback 审计**：旧 shell 中 retry 属于历史 alternate path 内部逻辑；由于整条路径的参数/checkpoint 契约已失效，整体按 dead alternate execution path 删除。主链没有新增 fallback、兼容层或依赖。
- **恢复性**：删除尚未 commit，可从 Git 历史/当前基线恢复；未删除 `runs/` 中历史实验结果。

## 运行说明（清理后）

```bash
WITH_PERCEPTUAL=0 bash scripts/run_unified_bev_claim_probe.sh
```

### 2026-08-25 — 恢复 VGGT front2 与车辆运动尺度锚点

- **发现问题**：已验证有效的 `image_00` 双 crop 只存在于 Metric3D builder；VGGT 主链仍把 1408×376 整图直接压成 160×96，且多帧 metric scale 使用所有 virtual-view 光心均值，没有直接使用车辆位移。
- **修正内容**：source layout 固定为 `front2_left3_right3_v1`，即每帧 2 个共享 cam0 光心的前视 crop、左/右鱼眼各 3 个共享各自物理光心的 tangent views，共 8 views。前视 crop 使用各自平移后的 principal point，左右鱼眼继续使用 `calib_cam_to_pose.txt` 的独立外参。
- **尺度契约**：多帧 VGGT 的 predicted centers 先按三台物理相机聚合，metric baseline 直接来自每个 source frame 的 `T_world_imu`；`Ns=1` 仍明确使用 calibrated camera-rig fallback。
- **cache 迁移**：VGGT cache 升级为 v6；旧 7-view sample/VGGT cache fail-fast，必须重建。
- **验证**：44/44 单元测试通过；真实 KITTI-360 tile 得到 64 source views，两个 front crop 共享 cam0 光心，left/right triplet 分别共享各自光心。真实 VGGT forward 中 `Ns=1` 标记为 camera-rig scale，`Ns=2/8` 标记为 vehicle-motion scale；新 v6 cache 已完成 Stage-A one-step、Stage-B one-step 与 evaluator smoke。

### 2026-08-25 — 修复 target RGB 的超宽图压缩

- **发现问题**：VGGT source 已改为双 crop，但 Stage-A/B 的 `target_rgb` 仍沿用旧 `_view()`，把 1408×376 整图直接压成 160×96；因此此前可视化中的“原始 RGB”已经发生横向几何扭曲，训练目标也不成立。
- **修复**：target 改为与 cam0 source 一致的两个 calibrated front crops，分别保留 crop-specific `K`，renderer、target support、RGB losses、depth metrics、SSIM/LPIPS 全部支持 `(B,2,...)`；删除旧整图 target 路径，Stage-A/B schema 升至 v3，sample cache 增加 target-layout gate。
- **验证**：45/45 回归测试通过；真实 tile 得到 `target_rgb=(2,3,96,160)`；Stage-A/Stage-B/evaluator 双-target smoke 通过。单 tile Stage-A 5k QA：dense RGB PSNR 16.766 dB、LiDAR depth AbsRel 0.073、RMSE 2.883 m、VGGT lift coverage 0.421。

### 2026-08-25 — 实验单位重构：frame → route chunk（spatial hole completion）

- **动机**：用户裁定主实验单位不应是帧而应是"a geometry-bearing ground observation chunk"。连续帧高度重复使 Ns 逐帧稀疏混淆了两种问题：encoder 少视图退化（诊断）与车辆未采集该空间（主张）。主张升级为 *overhead-conditioned recovery of missing ground-chunk geometry states under sparse trajectory coverage*，指标从"几帧"改为 ground-equivalent acquisition length（米）。
- **chunks.py**：`build_route_chunks`（12 m 弧长切分、>5 m 跳变切段）、窗口（4 chunk ≤48 m）、hole=中间连续块（missing_chunks，K∈{1,2,3}）、guard 单侧化（chunk0 护右侧、其余护左侧——hole 永远是内部块，外侧不必 guard；两侧 guard 会吃掉 12 m 中的 8 m 导致快行路段无帧可用）、lift 帧（guard 安全带内均匀取）与 geometry 帧（lift∪上下文，≤8 帧封顶 VGGT 视图预算）。
- **ChunkedUnifiedBEVDataset**：复用全部视图构建；source=每 chunk 2 个 lift 帧（4×2×8 视图=64，与 v6 预算一致）；query=各 chunk 弧中点帧（front2）；meta 带 chunk 表。0003 真实数据 22 窗口。
- **cache v7**：每 chunk 一次独立联合 forward（64 视图/次，成本与 v6 单 subset 相同）；chunk 级 exactness（条件间只在 chunk 成员上不同，K-chunk 条件拼装条目零额外推理）；identity 按窗口 chunk 表；主链不再出现 camera-rig fallback。实测 1 tile：4 chunk 全 vehicle_motion 多 baseline，MAD 0.028–0.112，18 s/tile。
- **Stage A/B/eval/C2/主链**：A 多 query 渲染（render_multi_view，同时修复旧多视图 target 的潜在广播问题）；B 的 sparse_source_choices 变为保留 chunk 数 K、alpha=1−K/Nc、K=Nc 逐位恒等；新 evaluator `eval_unified_bev_chunk_probe.py`（hole/hole_core 分区= kept support 腐蚀 guard、per-query-chunk 指标、逐 chunk scale QA）；新 C2 `consistency_unified_bev_chunk.py`（{c0,c2} vs {c1,c3}）；`l_equiv_analysis.py` 输出 L_equiv；主链 `run_unified_bev_chunk_chain.sh`。schema 升 3，fingerprint 纳入 chunk_config。
- **验证**：50/50 单测（新增 5 项：弧长切分/跳变、hole 模式与 guard、lift⊆geometry∧全 K guard、identity 幂等、chunk 计数 alpha 表）；1-tile 真实 GPU smoke 全链通过（cache→A→B→eval→C2→L_equiv）。
- **已知事实**：K=2 时 M_hole≈6.5%、hole_core≈3.4%——kept chunk 相机的远距可见性仍覆盖多数 BEV，hole 大小由 conf 门控与 K 共同决定，属实验结果而非缺陷；完整 20k/10k 训练未启动。

## 运行说明（chunk 链）

```bash
WITH_PERCEPTUAL=0 bash scripts/run_unified_bev_chunk_chain.sh
```
