# 评估指标参考手册（persistent world state）

> 数据来源：`scripts/eval_world_state_trajectory.py` 每行 JSONL 记录的全部字段。
> 区域定义：**visited** = 截至时刻 t 车端 LiDAR 扫过的格子；**ahead** = 全轨迹最终扫过 ∧ 尚未 visited；**off-route** = 有真值标签但整条轨迹都不经过（仅诊断）。
> 有效格子 < 256 的区域不报数值（JSONL 里为 null）。

## 一、主判定指标（E1–E4 门控）

| 指标 | 定义 | 回答什么 | 出现行 |
|---|---|---|---|
| `height_ahead_mae` | ahead 区高度读出 vs 静态真值的平均绝对误差 | **E1/E2**：先验在到达前给的布局质量 | 每步 |
| `height_visited_mae` | visited 区同上 | **E3/E4**：到达写入与持久质量 | 每步 |
| `density_visited/ahead_mae` | 占据密度同上 | 辅助（非门控主指标） | 每步 |
| `g_update_height` | 本步写入前→后，在当前测量写入区（∧有效标签）的 MAE 差 | **E3**：到达这一步把地图改善了多少（>0 即写入有效） | 每步 t≥1 |
| `forget_1_to_t_height` | 当前读出 vs chunk1 刚写入后的读出，在 chunk1 写入区的 MAE 差 | **E4**：老区域是否被后续更新破坏（≤0 = 无遗忘甚至继续改善） | 每步 t≥2 |
| `version`（=t） | 已同化的 chunk 数；`traversed_m` 为对应里程 | 查询时刻轴 | 每步 |

对照（control）即在这些指标上横向比较：aligned / xy / random / shift_cross / shift_road / sat_only / ground_only / one_shot / world_upper。

## 二、深度渲染指标（辅助判定）

| 指标 | 定义 | 判什么 |
|---|---|---|
| `depth_absrel` | 把状态经**冻结** depth reader 渲染到**本 chunk 查询相机**，与该帧真实 LiDAR 深度比 \|pred−gt\|/gt 的均值 | 状态是否把几何组织成**可从 held-out 视角渲染**的三维结构。查询帧与测量帧严格隔离（cache v2 断言）；测的是状态内容，不是渲染器（Stage A 后冻结） |

注意：视角在轨迹上（非轨迹外新视角）；训练场景上与 depth 一致性损失同目标，held-out 场景上测泛化。

## 三、契约 / 审计指标（硬性要求）

| 指标 | 定义 | 要求 |
|---|---|---|
| `outside_latent_max` | 本步更新在测量 support **外**的 latent 最大变化量 | **必须 = 0**（逐位不变契约；任何非零即违约） |
| `measurement_support_cells` / `supervised_support_cells` | 写入区格数 / 写入区∩有效标签格数 | 写入规模与伪负防护 |
| `measurement_target_overlap` | 监督区 ∩ 真值标签 ÷ 写入区 | 测量写入与真值覆盖的重合度（本场景 ≈0.80） |
| `dgm_status` / `dgm_mad_m` / `dgm_tier2_cells` | DGM 双层锚定状态（dgm/回退）、近距层一致性、远距兜底格数 | 测量侧健康度审计 |

## 四、诊断指标（不进门控）

| 指标 | 定义 |
|---|---|
| `height_offroute_mae_diag` | off-route 区域高度误差——车不去、纯先验覆盖的区域 |
| `offroute_cells` | off-route 有效格子数（覆盖面诊断） |
| `visited_fraction` / `ahead_fraction` | 两个区域的相对大小（解释"为什么 ahead MAE 随 t 不可直接横比"：区域成员在变） |
| `measurement_source` | 测量来源（vggt_cache / +dgm / teacher fallback） |

## 五、跨实验判定方式

- **配对检验**：`compare_world_state_paired.py` 对 aligned vs xy 按 scene 配对 bootstrap（当前 n=1，CI 无效——多 scene 后生效）；
- **汇总**：formal 链的 `SUMMARY.json` 取各 control 的 t0 与终行关键指标；
- **机制诊断**：`diag_e3_e4_mechanism.py` 逐步输出 `err_before/err_after/meas_readout_err/copy_err/retention_overlap/latent_delta_in`（定位"为什么"用，不进论文主表）。
