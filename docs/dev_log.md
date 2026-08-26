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
