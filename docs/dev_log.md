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

### 2026-08-25 — 主线改为 persistent georeferenced world state

- **动机**：chunk 内删帧不能构成卫星不可替代的空间缺口；新主问题是异步 overhead/ground 观测如何初始化并持续更新同一个地理对齐世界状态。讨论稿 `todo/persistent_georeferenced_world_state.md`，计划 `.omx/plans/persistent-georeferenced-world-state.md`。
- **新增模块**：`world_state.py`（契约/provenance/ModelInputs 隔离）、`world_targets.py`（固定 datum + georeferenced satellite resample）、`world_data.py`（100 m scene tile）、`state_models.py`（world encoder、卫星/XY initializer、updater、one-shot）、`world_checkpoints.py`（`world_state_v1`）。
- **新主链**：`scripts/run_world_state_probe.sh` → target build → Stage A interface → assimilation 四分支 → trajectory eval → paired bootstrap。
- **v1 边界**：无 UAV/VLA；off-route 只作 diagnostic；measurement 在未接 VGGT cache 时用 support-masked world teacher 作为 updater 输入（几何证据来自累计 LiDAR support，不是自由 attention）。
- **不 commit**：按用户约束本轮不自动 commit/push。

### 2026-08-26 — 删除 spatial hole-probe

- **删除**：`run_unified_bev_chunk_chain.sh`、`eval_unified_bev_chunk_probe.py`、`consistency_unified_bev_chunk.py`、`l_equiv_analysis.py`；`ChunkedUnifiedBEVDataset` 与 chunk cache v7 attach；Stage A/B / VGGT builder 的 `--chunked` 挖洞路径；`missing_chunks` / `guard_keep_mask` / `build_chunk_windows`。
- **保留**：`chunks.py` 的弧长切段与 `select_chunk_frames`，仅供 world-state 作为测量包。旧 hole checkpoint 若带 `chunk_config` 会被 Stage-A loader 拒绝。
- **文档**：README / user_requirements 不再把 hole-probe 标为可运行预实验。

### 2026-08-26 — 固化实验方案

- 写入 `docs/experiment_plan.md`：四实验（ahead 先验、优于 XY、到达写入、不遗忘）；静态真值用 `data_3d_semantics`，当前包去前景用 `data_3d_bboxes`；只训 \(W_s\) 与 \(U\)，VGGT 冻结。

### 2026-08-26 — 卫星骨干改为冻结 DINOv2

- `SatelliteInitializer` 不再从零训 ViT。默认 `dinov2_vitb14` 冻住抽特征，只训 `write_head`（DINO token 插值到 200×200 + 固定 XY 编码）。
- `FixedXYInitializer` 不跑 DINO，零特征走同一 write head。checkpoint 不保存 DINO 权重。
- 单测用 `backbone='tiny'`，避免拉取 hub。

### 2026-08-26 — 实验方案重梳

- `docs/experiment_plan.md` 与当前实现对齐：DINOv2 冻特征 + write head 初始化 \(Z_0\)；VGGT 冻测量；四实验 E1–E4 为全部主验证。

### 2026-08-26 — 修复 world-target 高度的绝对海拔 clip 顺序 bug

- **发现**：`accumulate_lidar_surface` 把物理高度护栏 `np.clip(height, -2, 40)` 作用在**绝对 world-Z p90** 上（KITTI-360 绝对海拔 ~115–125 m），`height_minus_datum` 减 datum 后全图恒等 −85.30 m（0003 smoke blob 实测 std=0、unique=1）——静态 height 监督与 E1–E4 全部 height 指标失义；G_t 测量路径（`state_models.py` 的 `h_rel=z_abs−datum`）本来就正确，因此 updater 会被训练成忽略正确测量。同类 bug 第二次（第一次=08-23 Exp1 的 clamp(30) 压平 DEM）。
- **修复**：clip 移入 `height_minus_datum`（datum 相对域，−2/40 m，语义对齐 `geometry.relative_height_map`）；`accumulate_lidar_surface` 返回未 clip 的绝对 p90；模块 docstring 写明绝对海拔陷阱。
- **测试**：新增 `test_world_height_targets_survive_absolute_elevation`（绝对海拔 120/128 m 合成点 + datum 121.8：道路 −1.8、立面 +6.2、500 m 噪声→相对上限 40、60 m→相对下限 −2、"非恒定"断言）；58/58 通过。
- **重建验证**：0003 同 scene 重建；valid/density/chunk_lidar_support/datum/satellite/origin 全部逐位不变（外科手术式改动），world_target_hash 更新（旧 checkpoint 按设计 fail-fast）。新高度：mode=−1.94 m（≈datum 125.30 − 传感器高 1.8 m 的路面水平）、p50=−0.87、p99=6.75、max=8.78、15257 unique。
- **可视化验证**：新脚本 `scripts/qa_world_height_targets.py`：卫星图 / 修复前后高度图（同色标）/ 前后直方图（红尖峰 −85 vs 蓝展开）/ 2 m、5 m 等高线叠卫星灰度（与建筑 footprint 含太阳能屋顶对齐）/ 3D 高度表面。图：`runs/world_state_targets_smoke/qa/height_fix_verification.png`。
- **已知观察（非缺陷）**：37% 有效格落在 −2 m 地板——scene 内地形起伏超过固定 datum 以下 2 m，是 v1 固定 datum 政策（禁逐 chunk quantile）的代价；若影响 E1–E4 可评估把 floor 放宽到 −5 m（设计参数，非正确性问题）。
- **运维**：sda2 又未挂载，`udisksctl mount -b /dev/sda83` 恢复。

### 2026-08-26 — world_target_v2：版本契约 + 因果 datum + 地板放宽（外部审查三点）

- **核实（三点全部属实）**：①`WORLD_TARGET_VERSION` 未随 clip 修复 bump，且 interface fingerprint 只含字符串常量、不含 scene 的 `world_target_hash`——buggy-math 时代的 checkpoint 仍能通过验证；`WorldStateSceneDataset` 也不检查 blob 的 `world_target_version`。②−2 m floor 在真实 0003 上压掉 37% 有效格。③datum 用整段轨迹光心中位数，t=0 时读取未来车辆位置（非因果）。
- **v2 契约**：`WORLD_TARGET_VERSION="surface_p90_relative_height_clipped_v2"`、`Z_DATUM_POLICY="first_chunk_lidar_optical_center_world_z_median_v1"`；`MIN/MAX_RELATIVE_HEIGHT_M=-8/40` 提为 `world_targets` 模块常量。
- **因果 datum**：新增 `first_chunk_datum_z(window, by_fid)`（仅第一个 chunk 的 LiDAR 光心 world-Z 中位数，建图开始即可得，全 scene 固定）；`build_scene_blob` 弃用全场中位数。
- **版本/manifest 绑定**：`WorldStateSceneDataset` 在 `__init__`/`__getitem__` fail-fast 校验每个 blob 的 `world_target_version`，并暴露 `manifest_hash`（scene_id+world_target_hash 排序 sha256）；Stage A checkpoint 保存 `scenes_manifest_hash` 且计入 interface fingerprint；assimilation 训练前 `validate_scenes_manifest`（eval 用 held-out scenes 不做 manifest 等值校验，只走版本契约）。
- **QA gate**：build 期 `floor_frac ≤ 2%`、`ceil_frac ≤ 1%`，超限 skip 该 scene；scene 行打印两个比值。
- **测试**：新增版本契约/因果 datum/版本拒绝+manifest 绑定（含 fingerprint 敏感性）三测试，回归测试断言改用共享常量；61/61 通过。
- **重建验证（0003 同 scene）**：datum 125.30→128.73（该路段为下坡：起点比全场中位数高 3.4 m——这正是旧 −2 m floor 37% 压积的根因）；floor_frac 37%→**0.38%**、ceil 0%；高度 [−8, 5.35]、24,066 unique；v2 dataset 加载 OK，旧 v1 blob 目录端到端被拒。QA 图 `runs/world_state_targets_smoke/qa/height_target_v2_qa.png`（前后高度图/直方图/等高线叠卫星/3D 表面，视觉模型复核通过）。
- **语义注记**：因果 datum 使高度分布整体下移（p50=−4.31：下坡路远端路面在 datum 下 4~5 m），这是固定 datum 的诚实几何而非缺陷；地板 −8 m 对应"起点以下约 6 m 下坡"容限，QA gate 会在更陡 scene 上显式暴露。

### 2026-08-26 — 接入正式 VGGT chunk measurement（主链最大实现缺口闭合）

- **缺口核实**：assimilation/eval 的 G_t 都是 `world_enc(height*support, density*support, support)`——把 LiDAR 监督裁到当前 chunk 再编码，E3 实际验证的是"裁真值写入"；`GroundMeasurementEncoder` 在 optimizer 里却从未被 forward。
- **新增 `world3d/unified_bev/world_vggt.py`**：`load_world_vggt_cache`（按 scene_id/world_target_version/world_target_hash 三重身份 fail-fast，重建 targets 自动作废旧缓存）；`chunk_measurement_from_cache`（fp16 缓存条目 → GroundMeasurementEncoder → GroundMeasurement，最近 unroll 步保留 meas_enc 梯度、prefix replay detach）；`teacher_measurement`（显式降级 fallback）。
- **新增 `scripts/build_world_vggt_cache.py`**：每 scene 每 chunk（blob `chunk_table.geometry_fids` ≤8 帧 × 8 视图=64）一次独立联合 VGGT 前向，复用 v6 的 `run_joint_subset`（车辆运动定标、conf 归一化、depth/conf 回投到 96×160 视图分辨率）；缓存存 rgb/K/T_world_cam/T_world_imu/depth/conf + scale QA，逐 chunk 增量落盘可续建。
- **assimilation/eval/probe 链**：两者接 `--vggt_cache`；eval 现在从 assimilation checkpoint 加载 `measurement_encoder`（teacher 时代从不加载）；行记录 `measurement_source`；**E3/E4 区域语义修正**——g_update/outside_latent_max/forget 改用测量自身 support（VGGT support 远于 LiDAR chunk mask，用后者判"support 外"会误报合法写入）。probe 链插入 train/test 两个缓存构建步骤。
- **真实 smoke（0003，8 chunks）**：缓存 0.6 min（8×64 视图，全 `vehicle_motion` 定标，MAD 0.024–0.114，pose RMSE 0.23–0.67 m，conf>0.3 占 0.61–1.00）；interface 20 步 → assimilation(sat_ground, 真测量) 20 步 loss 12.7→4.6 → eval aligned 9 行 `measurement_source=vggt_cache`；**VGGT support 27,585 格 vs LiDAR chunk mask 11,141 格、Jaccard 0.389**（测量确实来自 VGGT 门控而非监督 mask）；测量-support 语义下 `outside_latent_max=0.0`（精确保持契约在真测量上成立）；teacher fallback 冒烟通过并带显式 WARNING。
- **顺带修复**：DINOv2 hub 加载优先本地缓存（`source='local'`）——`_parse_repo_info` 在缓存命中前就会探测 GitHub main 分支，断网即挂（本机 repo+权重已在 `~/.cache/torch/hub`）。
- **测试**：新增 meas_enc 门控（conf/depth 越界 → support 清空、support 外 confidence=0）与缓存身份/组装两测试；63/63 通过。

### 2026-08-26 — P0 修复：VGGT 测量 support 超出 LiDAR 标签的伪负监督

- **核实（外部审查属实）**：assimilation 的 chunk 损失用 `meas.support`（VGGT 门控，chunk1=27,585 格）做 mask，但 `height/density` 在 `world_valid` 外是占位 0（`height_minus_datum`/`log_normalize_density` 构造）——「VGGT 看见、无标签」的格被监督为高度 0/密度 0，是伪负标签而非 masked supervision；eval 的 `g_update`/`forget` 同样用未交 `valid` 的测量 support，指标被 0 值污染。
- **量化（新诊断字段）**：chunk1 overlap=0.727（7,527 伪负格，27%）、chunk4/8 overlap≈0.84–0.86（约 15%）——修复前 15–27% 的监督区域是伪负标签，此前 20-step loss 12.7→4.6 不能解读为几何学习。
- **修复**：新增 `world_state.supervised_region(measurement_support, world_valid)`（契约函数+回归测试）；训练 chunk 损失与 one_shot final 损失改用 `supervised = meas.support & valid`；eval `g_update_height`/`forget_1_to_t_height` 改在 `测量support & valid` 上计算，新增行字段 `measurement_support_cells`/`supervised_support_cells`/`measurement_target_overlap`；`outside_latent_max` 保持对完整测量 support（保持契约关心的是写入区域本身）。
- **同类第三处（自发现）**：`distill = F.smooth_l1_loss(state.latent, z_world)` 是全图无 mask——experiment_plan §6 明确是 "masked distill"；无 mask 版把卫星初始化的 Z_0 在 ahead/未知区域拉向 world teacher 的「未知」外推，直接压制 E1。改为 `masked_smooth_l1(state.latent, z_world, valid)`（valid 经 `_expanded_mask` 广播到 64 通道）。
- **验证**：64/64 单测；重跑 smoke（interface 复用 → assim 20 步 loss 4.18 → eval aligned）`outside_latent_max=0.0` 不变，overlap 字段落盘。

### 2026-08-26 — one-shot mask 语义修正 + VGGT 测量帧/查询帧严格隔离（cache v2）

- **one-shot 遗漏（外部审查指出）**：final 损失区域原用 `sup.final_support`（累计 LiDAR support），与 one-shot 聚合器实际写入区域不一致——会在"LiDAR 有标签、VGGT 没写入"处计算损失、漏掉"VGGT 写入且有效"的格。新增 `state_models.one_shot_support(measurements)`（测量 support 并集）+ `test_one_shot_support_matches_aggregate_write_region`（与 `aggregate_measurements` 写入区域逐位相等）；训练改 `supervised_region(one_shot_support(measurements), valid)`。
- **查询隔离（此前 depth_absrel 非严格 held-out）**：旧契约 `geometry_fids`（lift+ctx 全成员）含 core 查询帧——实测 0003 smoke **8 chunk 中 3 个（37.5%）query 帧在测量帧内**。v2 修复：构建端剔除 `core_fid`、从同 chunk 未用帧按弧距最近补足（仍 8 帧 64 视图）、entry 存 `measurement_fids`+`query_fid` 并断言 disjoint；加载端 `assert_query_isolation`（缺字段=query 隔离前旧缓存、query_fid 不匹配、泄漏三种 fail-fast）；`WORLD_VGGT_CACHE_VERSION`→`world_vggt_chunk_measurement_v2`；v1 缓存已删除重建。
- **验证**：65/65 单测（缓存测试扩三断言：身份不匹配/泄漏/旧 schema）；重建 0.6 min 全 vehicle_motion（MAD 0.022–0.114）；链路 smoke（sat_ground 20 步 + one_shot 2 步 + eval aligned）通过，`outside_latent_max=0.0` 保持，`depth_absrel≈0.74–0.81`（严格 held-out 下的 20-step smoke 值，不作效果结论）。

### 2026-08-26 — 四分支收敛为 shared_assimilation（E2 的结构化落实）

- **动机**：旧链四次独立训练（sat_ground/xy_ground/ground_only/one_shot）各自产出 updater/meas_enc——E2 的"同一 updater：sat-init vs XY-init"对比被 updater 权重差异混淆。
- **新训练形态**：同 batch 内两条链 `Z_t^sat=U(Z_{t-1}^sat,G_t)`、`Z_t^xy=U(Z_{t-1}^xy,G_t)` 消费**同一条测量流**（prefix detach 一次计算、recent 带梯度供两链共享）、同一个 updater、同一组冻结 readers；训练参数=updater+meas_enc+两个 write head（satellite/XY 各一，容量对齐）；loss=两链之和（updater 从两链收梯度，学的是与初始化无关的同化算子）。ground_only/one_shot 降为纯评测期变体（--control 改 init/聚合方式），不再训练。
- **checkpoint**：`branch="shared_assimilation"`，单文件含 sat/xy 初始化器+updater+measurement_encoder；eval 全部 8 个 control 共用它；probe 链从 4 次训练变 1 次。
- **保留语义**：chunk 损失仍在 `supervised_region(meas.support, valid)`；distill 仍 masked valid；retention 仍 visited_mask；查询隔离/缓存 v2 不变。
- **smoke（0003，20 步）**：两链 loss sat 13.28→3.07 / xy 17.86→3.15；四 control 单 ckpt 评测——t=0 ahead MAE 排序已符合 E1 预期方向：satellite 0.898 < XY 1.067 < 空初始化 2.544（one_shot t=0 与 aligned 逐位一致=共享初始化验证）；recurrent control 末行 g_update 为正（0.02–0.03，smoke 量级）、outside=0.0 保持。
- 65/65 单测通过。

### 2026-08-26 — shared_assimilation 定稿：测量单次计算确认 + plan 对齐 + 测试裁剪

- **用户裁定**：保留共享联合训练（meas_enc 单实例由两链损失共同训练，不改为随机冻结）；确认实现中每 chunk 的 measurement 恰好计算一次（`prefix`/`recent` 列表在 `_run_chain` 外构建一次，两链闭包消费同一对象；meas_enc 每 chunk 只前向一次）。
- **experiment_plan 对齐**：§2 训/冻表新增 `GroundMeasurementEncoder`（训、单实例共享）与 XY write head（训、容量对齐）两行；表后写明 shared_assimilation 双链共享 G_t/M_t/updater/readers、G_t 每 chunk 只算一次；§6 分支段改为"不再有独立训练分支，ground_only/one_shot 仅评测期 --control"。
- **测试裁剪（65→39）**：test_unified_bev.py 48→22——删除退役 frame-completion 路径的全部契约（LatentCompletion/alpha 恒等/coordinate_only completion、卫星 ViT/heightmap prior、nadir 往返、M3D 双 crop、observation RGB 损失、B7 道路系控制、LPIPS/SSIM、frame Stage-A/B checkpoint schema、v6 frame-cache attach 身份）；三个测试缩减为只测存活部分（ColumnFieldDecoder 渲染、fixed_relative_xy_encoding、masked_smooth_l1 空 mask）；保留 world 链活代码契约（几何约定/splat/height_statistics/unproject/GroundDenseBEVEncoder/front2 内参/VGGT 定标四件/chunks/readers 冻结）。裁剪后 39/39 通过。

### 2026-08-27 — E0 冻结 DINOv2 卫星特征探针：信息上限判决（用户问"这个能不能学出来"）

- **设计**：绕开状态机，直接测卫星通路信息上限——冻结 DINOv2-ViT-B/14 对与 `SatelliteInitializer` 完全相同的 200×200 south-up 卫星 raster 抽 16×16 token，小卷积头直接回归 LiDAR height/density（valid 监督）。四臂同头同训练：dino / xy（零特征+固定XY，安慰剂）/ shuffle（逐 scene token 置换，布局绑定性检查）/ scratch（从零小 CNN，可训 encoder 参照）+ 常数基线。训练 22 scene（drives 0000/0005/0009，含 QA gate 跳 2），评测 0003 地理隔离 scene。
- **结果（held-out 0003）**：dino r=+0.600（train 0.785）≫ xy r=−0.347；shuffle r=+0.441（布局绑定贡献 0.6→0.44 的下降）；scratch r=+0.259（22 scene 下从零 encoder 过拟合，held-out 崩）；hMAE：dino 3.23 < scratch 3.16 ≈ shuffle 3.39 < xy 3.63 < 常数 3.70。**关键分解：全部臂 bias≈+2.5~3.1 m**——eval scene 是下坡路（高度中位 −4.3）而训练 scene 全平地，绝对标定不迁移；bias 校正后 dino MAE 3.39→2.15。视觉复核（e0_vis.png）：预测与真值的道路走廊/建筑 footprint 结构对应，残差以系统性偏置为主。
- **判决**：①**信息存在**——冻结 DINO 特征携带可跨 scene 迁移的布局信息（r=0.60 且 shuffle/xy 对照干净），E1 有东西可测；②**冻结骨干的选择在当前数据量下是对的**——从零 CNN 显著更差；③**绝对高度标定不可从卫星学**（地形分布依赖），这本来就是状态的分工：Z_0 给布局（低频），updater 的地面测量负责绝对标定（E3）——E1 判读必须看 bias 分解/Pearson，纯 MAE headline 会低估先验价值。density 各臂几乎无分化（0.176–0.201），layout 信号主要在 height。
- 产物：`scripts/probe_world_satellite_prior.py`、`runs/world_state_e0/{summary.json, e0_vis.png}`；targets_train 22 scene（可复用）。

### 2026-08-27 — E0 勘误与定版：数字矛盾修正 + centered MAE 入代码（外部审查指出）

- **修正 1（不等式反向）**：上一条记录写 "dino 3.23 < scratch 3.16" ——不等号方向错了。正确排序（首次运行）：scratch 3.158 < dino 3.230 < shuffle 3.387 < xy 3.631；**hMAE 上 dino 并不优于 scratch**，dino 的优势在 Pearson（0.600 vs 0.259）与 centered MAE（2.15 vs ~2.26）。
- **修正 2（数字来源混淆）**：3.230 = 探针首次运行（held-out 0003 的 valid 区域；该 scene 上 valid 与 route 数值逐位一致）；3.39→2.15 = 可视化时的**另一次独立重训**（未定 RNG 的 scene 顺序导致两 run 结果不同）。两数不同源，不应混排在同一句里。
- **根因修复**：scene 顺序此前用 numpy 全局 RNG、未随 `--seed` 播种——已加 `np.random.seed(args.seed)`；`evaluate()` 正式新增 `height_centered_mae`（去均值偏置）与 `height_median_centered_mae`（去中位偏置，更稳健），每次运行自动落盘 summary.json。
- **定版数字（fully-seeded 单次运行，held-out 0003，valid=route）**：
  | arm | hMAE | bias | centered | med-centered | r |
  |---|---|---|---|---|---|
  | dino | 3.402 | +2.901 | **2.202** | 2.186 | **+0.469** |
  | scratch | 3.412 | +3.087* | 2.272 | 2.268 | +0.403 |
  | shuffle | 3.540 | +2.965* | 2.459 | 2.452 | −0.140 |
  | xy | 3.571 | +3.061 | 2.525 | 2.509 | −0.328 |
  （*为同 run 内数值；mean 常数基线 hMAE 3.702）
- **三次运行的稳定性（诚实记录单 scene 评测的方差）**：dino held-out r ∈ [+0.469, +0.600]（三次均最高且恒正）；xy 恒负 [−0.368, −0.328]；dino centered MAE 恒 < xy（2.15–2.20 vs 2.52–2.54）；全臂 bias 恒 ≈ +2.9~3.1。**结论对 run-to-run 方差稳健：信息存在/安慰剂干净/绝对标定不可学；但精确数字必须多 scene × 多 seed 才能上 headline（与 experiment_plan 的 32 scene × 3 seed 协议一致）。**

### 2026-08-27 — 高度校正探针：整移 vs 坡度 + oracle/VGGT 校正条件（scripts/probe_height_correction.py）

- **Part 1（残差形状诊断，held-out 0003）**：卫星预测（E0 dino 臂）减 LiDAR 的残差，常数 vs 最小二乘倾斜平面：valid 区 raw 3.402 → 常数(中位) 2.186 → **平面 0.906**；ahead 区 raw 3.671 → 常数 3.178 → **平面 0.924**；拟合坡度 75–81 m/km。**判决：误差主体是倾斜平面（下坡趋势没被预测），不是整体抬升**——状态机若要加全局标定，需要缓变地形坡度，单个标量修正的上限就是常数行（ahead 只能 3.67→3.2）。
- **Part 2（校正条件，ahead=4,147 格）**：oracle 标量（LiDAR 定偏，用答案）：+3.414 m → ahead MAE 3.671→3.238（几乎无改善，与 Part 1 一致）；**VGGT-chunk1 标量：−3.762 m，符号反**（差 oracle 7.2 m），校正后 MAE 6.622 比不校还差。
- **符号反转根因（诊断三连）**：VGGT 测量 BEV 高度（门控 unproject + z splat 列均值 − datum）在 overlap 上 `h_meas−gt` 中位 **+6.87 m**；分层：抬升格(gt>1)≈**+0.02**、近(<20m)路面 +1.7、**远(>20m)路面 +8.6**；corr(err, 距离)=+0.66。即误差随距离增长（远距/下坡像素深度欠估→点落在更近更高处，远格被高空内容填充），非整图常数。近限校正（<15/20/30m，只用可部署信息）仍为 −0.6~−1.4：无标签近处 support 格（天空线/立面列均值）继续污染中位。
- **判决**：①"整体高度修正值"**不能**按原设想进状态机——标量无上限增益且 VGGT 列均值 z 在远处/无标签格不可信；②正确顺序是先修测量侧统计量：每格 z 改用**低分位数（地面包络）**而非列均值 + 标定用格限近距，之后再考虑从已驶过轨迹的近格拟合**沿路坡度（1D）**而非全局标量；③本 scene 是下坡最坏情形，平地 scene 的远距偏差待多 scene 验证——但失败模式真实存在，标定设计必须先过它。
- 产物：`runs/world_state_e0/height_correction.json`（Part1/Part2 全量）。

### 2026-08-27 — 用户裁定：两条代码禁令落档

- **不做 1**：不给状态加单一全局高度 offset（坡度主导的残差让标量没有上限增益）。
- **不做 2**：不用当前 VGGT BEV 列均值高度做全局校准（远距被高空内容污染、offset 符号会反）。
- 已写入 `docs/experiment_plan.md` §8「不做」清单（各附一行实测依据）；`state_models.py` 列均值 z splat 处加 CAUTION 注释（含 +8.6 m/>20 m/corr=0.66 数字与"低分位数替代"方向），防止未来误用为标定源。测量侧低分位数地面包络为后续另议项，本轮不动代码。

### 2026-08-27 — 测量高度统计量重设计：地面包络 + 可靠距离门 + 车辆位姿锚定（用户五点方案）

- **定性（用户分析）**："VGGT 标量校正"本是拿第一包局部观测修整图——下坡路上一个数字本就修不好；车端其实带来了局部高度信息，是列均值把它算坏了（抬升 +0.02/近 +1.7/远 +8.6 分层为证）。正确过程：卫星给布局（前方坡度未知）→ 每到达一段写准一段 → 前方等到达。不是第一包校准整图。
- **实现**：①`geometry.ground_height_quantile`（硬 bin + 双趟稳定排序取每格 z 低分位=地面包络，替代 bilinear 列均值）；②`GroundMeasurementEncoder` 新增 `reliable_range_m=25`（超距像素整门控剔除——support 只含可靠域，前方等到达）、`ground_quantile=0.15`、`camera_height_above_road_m=1.75`（从 21 个训练 scene 数据定标的中位，与 KITTI-360 文档 1.73 m 吻合；非 eval 调参）；③新 `ground_field()` 方法：门控→unproject→低分位→**相机位姿 z − 1.75 作 chunk 局部路面基准**、残差中位偏移在测量内部扣除（绝不外溢为状态级全局 offset，遵守 §8 禁令）。
- **实测（0003 chunk1，旧→新）**：support 28,127→7,831（只写可靠域）；写入格内中位偏差 **+6.87→+1.09 m**、MAE **7.70→2.26 m**；近(<20m)中位 +0.67/MAE 1.21；anchor=−0.79 m（本 chunk VGGT 地面整体偏低被扣除）。与 LiDAR 标签 overlap 0.73→0.84（可靠域与标签对齐更好）。
- **链路**：assimilation 20 步 + eval aligned 全通；`outside_latent_max=0.0` 精确保持不变；`update_support_cells` 7,831 落盘可审计。
- **测试**：新增 ground_height_quantile 包络（均值会读 125、分位数读 122）、ground_field 锚定（齐次位姿取 z 的 [3,3]→[2,3] bug 在测试中暴露并修复）、可靠距离门（depth=30>25m → support 空）三测试；41/41 通过。

### 2026-08-27 — world_target_v3：官方语义语义云接管静态真值（用户拍板直接上 V3）

- **前提确认**：官方 static PLY 与我们的世界系完全一致——0003 tile 直接套 origin bin 命中 166 万点；17,831 个共性格 v2-p90 vs 官方-p90 中位差 −0.033 m、MAD 0.09 m（无镜像/单位/datum 错位）；instance 编码 `semantic*1000+classInstanceID` 在 17M 点上 100% 成立（stuff 类 XX000、thing 类带个体号）。
- **P0 审计（分歧按成分分类）**：ground-dominated 格（top<15%）v2 与官方几乎完美（median +1.2cm、偏低>1m 仅 0.02%）；**top-mixed 格 42.6% 偏低 >1m**（raw 视角遮挡采不全顶面）；车辆类污染 2.2–2.6%。选层规则由此实证：ground 主导取地面 p50，含 TOP(≥15%) 取 TOP 点 p95。
- **实现（world3d/unified_bev/semantics.py 新模块）**：PLY 解析器（static/dynamic 两套 header 自动识别）、LABEL_POLICY（GROUND/TOP/IGNORE/DYNAMIC 四组常量 + label_policy_hash + 版本串 inferred_geometry_colour_v1，待 labels.py 校验）、质量过滤（conf≥0.5 ∧ visible=1 ∧ 白名单类）、`select_surface_height` 成分感知选层、`bin_semantic_surface` 出 height_world_z/semantic_top/count 三图。
- **builder 重构**：`build_scene_blob` 真值源换官方 static 云（按 anchor_fid+margin 选覆盖段），raw scan 只保留 chunk_support 的职责（时间轴证据）；blob 新增 `semantics{label_policy_hash,conf_threshold,root}` 与 `semantic_top` 图；`WORLD_TARGET_VERSION="official_semantics_surface_v3"`。
- **抓到并修复一个回归**：初版把 valid 判断放在 `height_minus_datum` 之后（NaN 已被重置为 0）→ valid=全图 4 万、25k 空格假高度 0（伪负回归）。改为 `valid = ~isnan(packed.height_world_z)` 后：valid=14,827（v2 的 61%，监督变严变少是预期代价）、共性格 median diff +0.012/MAD 0.084、car 残渣全部消失（v3 HIGHER>1m 即"raw 污染被删"10.7% + "真屋顶恢复"<1m 占 2.2%）。
- **验证**：42/42 单测（新增 surface selection/instance coding/policy hash 测试）；dataset 加载 OK（高度 [−7.74,9.20] 连续分布、12,107 unique）；旧 v2 缓存按版本绑定自动被拒；22 train + 1 test scene 全量重建（train 0000×21+0006×1，skip 2；valid 中位 9,216 ≫ E 协议 256 门槛）；VGGT 缓存全重建（176+8 chunks、100% vehicle_motion、0 泄漏）；端到端 smoke（interface→assim→eval aligned）全通，outside_latent_max=0.0 保持。
- **待办**：labels.py 到手后校验 LABEL_POLICY（重点：static 云实测无一 car/person 点出现，26 号在 dynamic 是 carrier）→ 视需要 bump LABEL_POLICY_VERSION 重跑 targets；正式长训练（interface 5000 步 + shared_assimilation）在此 target 上启动。

### 2026-08-28 — LABEL_POLICY 官方校验完成：三处错判修正（v2 名单）

- **labels.py 已拉取校验**（46 个官方标签）。大类全对：7=road/8=sidewalk/9=parking/22=terrain/11=building/21=vegetation/17=pole/24=person/**26=car**（此前悬案定案：static 云实测无 26 点，停车从未进入语义云）。
- **三处错判修正**：①id 12 实为 **wall**（非"低矮地面"）→ 从 GROUND 移入 TOP（竖直结构投顶面票，原配置会让墙点拉高地面中位）；②id 6 实为官方特有 **ground** 类（非 fence）→ 从 TOP 移入 GROUND；③id 34 实为 **garage**（位置对，名字更正）、13=fence。补充归类：10=rail track、14=guard rail 进 TOP；20=traffic sign、16=tunnel 进 IGNORE；DYNAMIC_CARRIERS 扩为官方全动态族 {24,25,26,27,29,30,32,33}。
- `LABEL_POLICY_VERSION → official_labels_verified_v2`（label_policy_hash 随之变化 → 现有 v3 blob 按身份链自动可判失效，需用新名单重建 targets+重训——当前 overnight 链跑的是 v1 名单产物，其结果定位为"pre-verification 基线"，校验后差异若显著再决定是否重跑）。
- 42/42 测试通过；formal 链健康（interface step 1680/5000，loss 2.66→0.36，GPU 96%）。

### 2026-08-28 — E3/E4 不过的机制诊断（diag_e3_e4_mechanism.py，train 22 scene vs held-out 0003）

- **判决先行：E3/E4 失败是 held-out 域差，不是没学会写**。训练集上 updater 正常工作：g_update 中位 **+0.104**（130/176 步为正，t>4 后 +0.224），err_before 0.93→err_after 0.55，recurrent 在 **22/22** 训练 scene 上优于 one-shot（中位差 −0.255）；held-out 0003 上同一组权重 g_update 中位塌到 **+0.028**（4/8 为正），recurrent 反而输 one-shot +0.113。
- **held-out 是已知最坏情形（下坡 scene）**：①测量沿程劣化——meas_readout_err 2.77→6.45（train 中位 1.94 平稳），08-27 校正探针已证明 75–81 m/km 坡度在 25 m 包络内造成 ~2 m 系统性倾斜，标量中位锚定去不掉；②到达区域被陈旧远距写入占据——err_before 沿程单调爬升 3.18→5.84，逐步 ±0.1 的修正量永远修不掉前一 chunk 的远距偏差；③训练集几乎全是平地（0000×21+0006×1），该失效模式在训练中未出现，meas_enc/updater 对它零适应。
- **次级机制（两侧都在、held-out 放大）**：写入方向落在 reader 零空间——latent_delta_in 很大（train 1.19/held-out 0.64）但读出变化 ~0（t=1 时测量读出 2.77 明明优于状态 3.18，写入后反而 3.26）；64 维 latent 只有少数方向被 height reader 读出，无 depth 约束使 depth reader 方向自由漂移（depth_absrel 0.50→0.85 的恶化是免费旁证）。retention_overlap train 0.87 / held-out 0.98：保持项的锚定区几乎覆盖整个写入区，对弱写入态是额外拖累。
- **E4 勘误（上一轮判读有误）**：forget₁→ₜ 全为负（train −0.53 / held-out −0.62，负=改善），**遗忘门实际通过**；E4 只输在 one-shot 对比条款（held-out persistent 4.76 vs one-shot 4.41，差 7.9%>2%）。forget 恶化之说作废。
- **对下一步的含义**：当前"held-out 判死"建立在 n=1 且是对抗性 scene 上；先补 v2 名单 targets + 多个 held-out scene（含平地）再定生死；训练侧要加地形多样性或让测量对倾斜鲁棒（§8 禁的是全局标量 offset，不禁止局部/1D 坡度）；同化期写入需约束到 reader 可见方向（加 depth 读出一致性）。
- 产物：`scripts/diag_e3_e4_mechanism.py`、`runs/world_state_v3_formal_20260828/diag_e3_e4_{train,heldout}.jsonl`。

### 2026-08-28 — E3 修复第一轮：查询帧深度一致性 + retention 排除当前写入区（用户批准执行）

- **文档先行**：experiment_plan §2 Stage-A readers 行改写（reader 权重不更新，但 depth 读出一致性梯度穿过冻结 reader 到 latent）；§6 Loss 增补两项并附机制诊断依据，"无 depth loss"表述废除（RGB 仍不作主监督，深度只用 LiDAR 几何）。
- **代码**：①`world_state.retention_mask(chunk_support, chunk_index, current_supervised)` 新契约函数（`visited_mask(t-1) & ¬当前supervised`）；②`train_world_state_assimilation.py` 新增 `--depth_weight`（默认 0.1）：每个 recent 步把 `state.latent` 经冻结 `depth_r` 用 `render_multi_view` 渲到该 chunk 查询相机位姿，与 `sup.query_depth`（LiDAR 真值，仅入损失不入前向，无泄漏）masked smooth-L1；retention 改用 `retention_mask`。
- **测试**：新增 `test_retention_mask_excludes_current_write_region`（锚定区排除当前写入、老格保持、未访不锚、chunk1 拒绝）；43/43 通过（21+22）。
- **smoke（22 train scene，20 步，held-out 0003 评估）**：**g_update 8/8 全为正**（+0.12~+0.49；对照旧 8000 步模型中位 +0.028、半数为负）；**depth_absrel 0.40→0.26–0.40 不再单调恶化**（旧模型 0.50→0.85）；`outside_latent_max=0.0` 精确保持契约不变；height_visited_mae t=8 达 4.41（旧 4.76，与 one_shot 持平）。20 步 smoke 只证机制方向，不作效果结论。
- **遗留**：下坡"弯尺子"（测量侧倾斜）未治——训练集地形多样性或测量内 1D 坡度估计另议；正式长训练待 v2 名单 targets 重建后启动。

### 2026-08-29 — DGM1 地形锚定：对齐验证 + 双层测量锚定接入（用户拍板直接开工）

- **对齐验证（三步，脚本入 scripts/，数据入 runs/dgm_alignment_check/）**：①drive 级 LiDAR-vs-DGM voxel 检查（qa_dgm_alignment.py，修过一处 velodyne 外参错用 cam_to_pose 的 bug——曾造成 31% 点"沉入"DTM）；②600 帧窗垂直漂移分析（qa_dgm_vertical_drift.py）：0003 漂移 ≤0.25 m、0000/0009 有 ±0.2–0.5 m 局部伪影、0010 连续 6 窗 +1.7–2.9 m（疑似桥/源错误，需门控）；③**scene 级终检（qa_dgm_scene_check.py）：23/23 scene 路面格 MAD 中位 3.5 cm、最差 15.3 cm（逐 scene 一个常数 offset 后）——DGM1 与静态真值厘米级吻合，判决强 GO**。已知语义：semantic_top 存原始 ID（7/8/9/22=路/人行道/停车/地形），非 ×1000 编码。
- **实现**：①`world3d/unified_bev/dgm.py`：DgmTileSet（zip 内 100 万行 xyz 快速加载）、`fit_world_to_utm`（逐 scene 仿射，实测 RMSE 0.03–0.49 m）、`DgmAnchor.from_blob`（blob 的 geometry_fids 上拟合）+ `anchor_tile_tensor`；②`state_models.ground_field` 双层锚定：近距（≤15 m trusted）格保 VGGT 包络（中心改用 DGM 一致性中位 Δ）、**远距格（仅 15–25 m 点）写 `dgm + Δ`——勘测裸地形状替代坡度偏置的远距包络**；Δ 逐 chunk 由近距格中位数因果估计，只存在于测量内（§8 合规）；③QA 门控：trusted 层 vs DGM 残差 MAD 超门 → 整 chunk 回退 legacy 中位锚定（失败方向安全）；④`GroundMeasurement` 增 `dgm_qa` 审计字段，eval 行落盘 `dgm_status/dgm_mad_m/dgm_tier2_cells`；训练/eval/diag 三脚本接 `--dgm_tiles`（训练与评估必须同开关，否则输入分布错配）。
- **门控标定（calibrate_dgm_gate.py，184 chunk 实测）**：trusted 层"包络噪声"本底 MAD 0.35–1.23（train p99=1.15），伪影带 2–3 m；初版 0.5 m 门（按静态真值一致性定）导致全部 chunk 保守回退——改默认 **1.5 m**（p99 的 1.3 倍、伪影下界的 0.75 倍）。
- **验证**：46/46 单测（新增 tile 采样/仿射拟合精确性、双层锚定、MAD 门控回退三测试）；0003 真实 smoke 全链通（20 scene anchor 拟合、8/8 chunk `dgm` 状态生效、**每 chunk 4505 个远格接勘测地形**、g_update 8/8 为正、outside_latent_max=0.0 保持）。
- **输入层判决（diag_dgm_input_quality.py，held-out 0003 全部 tier2 远格，与训练无关）**：legacy 中位误差 **2.578 m → DGM 1.403 m（−46%）**，MAE 3.352→2.692；chunk 5/6 的中位误差压到 **0.24/0.15 m**——弯尺子在源头被扳直，方案 A（1D 坡度）正式作废，方案 B 降级为可选。
- **诚实边界**：20 步 smoke 的端点指标与无 DGM 持平（visited 4.402 vs 4.409）——收益在"到达时刻的陈旧写入质量"，需要正式长训练让 meas_enc/updater 学会利用干净输入；tier2 写裸地高程，在目标为建筑顶的格上与真值固有差异（到达后由测量纠正，非新增问题）；0010 类伪影段依赖门控拦截（本 smoke 未覆盖该 drive）。未 commit。

### 2026-08-29 — 退役 probe_height_correction.py（用户裁定）

- **删除**：`scripts/probe_height_correction.py`（08-27 弯尺子诊断探针）。动机：DGM 锚定上线后其历史使命结束（方案 A 作废、§8 两条禁令的实验依据已固化在 experiment_plan 与本日志）。
- **保留**：结果 JSON `runs/world_state_e0/height_correction.json`（untracked 数据）；dev_log 08-27 的诊断记录（含坡度 75–81 m/km、标量校正符号反转等数字）。按 E0 探针先例的差别：E0 的信息上限判决仍是 E1 判读工具故保留探针，本探针的结论已完全被后续工程取代。
- 验证：compileall 通过，46/46 单测通过；无任何代码引用残留。

### 2026-08-29 — v4 正式链：v2 名单 + DGM 锚定 + 深度一致性长训练（E3/E4 首次过门）

- **链**：`run_world_state_formal_v4.sh` 一键完成 targets 重建 → VGGT 缓存重建 → interface 5000 → shared assimilation 8000（`--dgm_tiles`）→ 9 control 评估 → paired → SUMMARY；产物 `runs/world_state_v4_formal_20260829/`（run_metadata 记录 git commit/开关）。
- **训练集扩容**：v2 名单下 max_scenes=64 构建出 **64 train scene（0000×39、0005×21、0006×4，skip 2）**——地形多样性顺带改善（0005 首次进入训练）；test 仍为 0003 单 scene。
- **E1–E4 判定（held-out 0003，对 v3 pre-verification 基线）**：
  | 实验 | v3 基线 | v4 | 门控 |
  |---|---|---|---|
  | E1 ahead t0 aligned vs xy vs shift | 4.43 < 5.25，shift 裕量 0.07–0.11 | **4.14 < 4.73**，shift 裕量 0.09–0.11 | 方向 ✓（裕量仍薄，n=1） |
  | E2 同 updater sat vs xy | +0.82 | **+0.60**（t8 收敛 3.799 vs 3.806）| 方向 ✓（n=1） |
  | E3 g_update 中位 | +0.028（4/8 正）| **+0.303（6/8 正）** | **过门** |
  | E4 persistent vs one-shot | 4.76 输 4.41（差 7.9%）| **3.799 优于 4.123（7.9%）** | **过门（反超）** |
  | E4 遗忘 forget₁→ₜ | 全负 ✓ | 全负（−0.40→−0.01）✓ | 过门 |
  | depth_absrel 终值 | 0.851（恶化）| **0.528**（t0 0.503，不再恶化）| 佐证修复 |
  | outside_latent_max | 0.0 | 0.0 全程 | 硬契约保持 |
- **归并诚实声明**：本轮同时引入四个变化（v2 名单、64 scene 扩容、DGM 锚定、深度一致性+retention 修复），E3/E4 翻转是合并效果，未做单因子消融；ground_only 终值 visited 3.615 略优于 aligned 3.799（单 scene 噪声量级，不动摇"卫星管 ahead、ground 管校正"的分工叙事，留观）。
- 64 scene 下 assimilation 终损 8.50（sat 3.24/xy 5.52）与 v3 的 2.86 不可比（深度项入损+数据集变了）。下一步按协议扩 32 scene × 3 seed bootstrap 前先补多 held-out scene（单 scene CI 无意义）。

### 2026-08-28 — 清理第一、二组退役代码（frame 链 + ICASSP27 试点）

- **第一组（旧 frame unified-BEV 链，13 文件）**：`run_unified_bev_claim_probe.sh`、`train_unified_bev_stage_a/b.py`、`eval_unified_bev_probe.py`、`consistency_unified_bev_multichain.py`、`build_unified_bev_cache.py`、`check_unified_bev_replay.py`、`compare_unified_bev_paired.py`、`smoke_unified_bev.py`、`diag_vggt_gate.py`、`export_vggt_metric_pointmap.py`、`render_vggt_pointmap_views.py`、`visualize_dense_reconstruction.py`，及 `configs/unified_bev_stage_a/b.yaml`（无 reader）。
- **重要修正**：`build_vggt_street_cache.py` 从删除清单撤回——`build_world_vggt_cache.py`（主链）import 其 `run_joint_subset`（VGGT 联合前向），`tests/test_unified_bev.py` 4 处 import `estimate_motion_metric_scale`；它是活的共享工具模块。
- **第二组（ICASSP27 试点）**：`icassp27_verify/` 整目录（VERIFY_LOG/REFACTOR_NOTES/out 可视化）、`world3d/models/icassp27_predictor.py`、`world3d/train/train_icassp27.py`、`world3d/data/kitti360_tuple_dataset.py`、`configs/icassp27_*.yaml` 两个。各 `__init__.py` 均为纯注释，无 import 断裂。
- **验证**：`compileall`（scripts/、world3d/unified_bev/、tests/）通过；余下 2 条 shell 链 `bash -n` 通过；14 个活脚本模块级 import 全 OK（CPU，隐藏 GPU 不干扰 formal 链）；42/42 单测通过（test_unified_bev 21 + test_world_state 21）。
- **规模**：共删 64 个 tracked 文件、5965 行；未 commit，可从 git 历史恢复。`models/`、`metrics/`、`utils/`、`datasets/`、`ckpts/`、`world3d` 非核心子包等第三、四组候选未动。
