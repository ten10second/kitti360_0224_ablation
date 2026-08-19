# ICASSP27 Round-2：Target-Centric Satellite Grounding 改造与验证计划

> 目的：交给 Codex 直接执行。  
> 仓库：`ten10second/kitti360_0224_ablation`  
> 当前主模型：`world3d/models/icassp27_predictor.py`  
> 当前结论：B1（source-only）优于 B2（sat+source），且 B2 在 inference diagnostic 中出现 `real satellite ≈ shuffled satellite > zero satellite`。  
> 本轮不追求直接让 B2 超过 B1；首要目标是证明模型开始利用 **scene-specific satellite content**，即让正确卫星图显著优于错误/错配卫星图。

---

## 0. 当前诊断与本轮假设

### 0.1 已有 pilot 结果

当前 20k-step pilot：

- B0：satellite + target raymap
- B1：source views + target raymap
- B2：satellite + source views + target raymap

总体结果：

- B1 最好
- B2 次之
- B0 最差

已有 paired satellite diagnostic：

| 条件 | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|
| B2 real | 12.135 | 0.5191 |
| B2 zero | 11.155 | 0.5694 |
| B2 shuffle | 12.273 | 0.5123 |
| B1 | **12.323** | **0.5031** |

关键现象：

1. `real > zero`：非零 satellite memory 对模型有正向分布作用。
2. `real ≈ shuffle`：模型几乎没有利用“这张卫星图属于这个 scene”这一事实。
3. B2 attention mass 中 satellite 约占 54%–62%，因此不是“完全没看 satellite”，而是 **没有建立 satellite layout → target query 的有效 scene-specific grounding**。
4. 因此当前问题优先级不是 fusion gate，而是 satellite grounding。

### 0.2 当前代码中的坐标不一致

当前实现中：

- satellite metric PE 使用：
  - `window_origin_xy + patch_offset`
  - 即 KITTI-360 global/world XY。
- target ray origin 使用：
  - `tgt_camera_center - window_origin_xyz`
  - 即 window-local translation。
- target ray direction 使用 world-frame direction。
- source translation `dt = Ts - Tt` 仍按 world axes 表达。
- `pose_vec` 中 translation 是 camera 与 IMU 的相对 offset，不提供 target 在 global map 中的完整平移位置。

因此三个几何条件没有形成一个明确的共同 reference frame。模型能看到 satellite tokens，但不容易直接学习：

`“某个 satellite patch 位于 target camera 的左前方/右后方多少米”`。

### 0.3 本轮核心假设

将所有几何条件统一到 **target-centric frame** 后：

- satellite token 位置表示为“相对 target camera 的平面位置”；
- source pose 表示为“source 相对 target 的 pose”；
- target ray 表示为 target camera 自身坐标中的 ray；
- 不再让 satellite PE 依赖 global absolute XY；

模型应更容易建立：

`correct satellite layout ↔ target-view content`

的 scene-specific 映射。

本轮首要成功标准不是 `B2 > B1`，而是：

> **B2(real satellite) > B2(shuffled/wrong satellite)**

---

# 1. 本轮明确不做的事情

为了控制变量，本轮 **不要** 同时引入以下改动：

- 不做 evidence-aware gate。
- 不做双路 cross-attention / modality-specific attention。
- 不加 modality embedding。
- 不加 auxiliary loss / contrastive loss / consistency loss。
- 不换 Emu3 tokenizer。
- 不换 SatMAE。
- 不恢复 IPM / BEV projection / RayRoPE。
- 不恢复 MM26 Stage-2。
- 不做 panorama baseline。
- 不把 target distance 扩到 25m/40m。
- 不修改 VQGAN tokenizer。
- 不引入新 backbone。
- 不把 street source 改成多 camera rig。

本轮只验证一个问题：

> **target-centric geometry 是否能让模型真正利用正确的 satellite content。**

---

# 2. 坐标系统一规范

本轮统一采用 target camera 作为几何 reference。

## 2.1 Target camera frame

KITTI camera frame 按现有 `cam0_to_world` 定义：

- camera `+x`：right
- camera `+y`：down
- camera `+z`：forward

对于地面平面上的 satellite patch，建议使用 target camera 的 **planar heading frame**，避免 pitch/roll 对 ground-plane 坐标产生不必要影响。

定义 target 的 world-frame：

- forward vector：`R_t[:, 2]`
- right vector：`R_t[:, 0]`

只取 XY 并归一化：

```python
f_xy = normalize(R_t[:2, 2])
r_xy = normalize(R_t[:2, 0])
```

若实际 KITTI 标定验证发现轴符号不同，以可视化 sanity check 为准；不要凭假设静默修改。

对于 world XY 中任一点 `p`：

```python
delta = p_xy - target_center_xy
x_right   = dot(delta, r_xy)
z_forward = dot(delta, f_xy)
```

统一把 satellite 平面位置表示为：

```text
(right_m, forward_m)
```

target camera 本身位于 `(0, 0)`。

## 2.2 Source translation

当前：

```python
dt_world = t_src - t_tgt
```

改为 target-camera-local translation：

```python
dt_tgt = R_t.T @ (t_src - t_tgt)
```

使用完整 3D：

```text
(right, down/up, forward)
```

rotation 保持：

```python
R_rel = R_t.T @ R_src
```

然后：

```python
rel_pose = concat(dt_tgt, rot6d(R_rel))
```

注意：

- 本轮 reference 固定为 `target -> source`。
- 不要同时混入 `source -> target`。
- 增加 unit test 验证同一 source/target pair 的 inverse relation。

## 2.3 Target ray

当前 ray direction 被旋转到 world frame，origin 是 window-local camera center。

本轮改成 target-camera-local：

```python
p_cam = K^-1 [u, v, 1]
d_cam = normalize(p_cam)
o_cam = [0, 0, 0]
```

因此每个 target token 的 ray：

```text
(o = 0, d_cam)
```

可进一步简化为只编码 `d_cam`，但为了最小修改，本轮可以保留 6D `(o, d)` 接口，其中 `o` 恒为 0。

关键要求：

- ray 不再依赖 `window_origin_xyz`。
- target ray 与 satellite target-centric position 属于同一个 target query reference。

## 2.4 Global pose token

本轮 **暂时保留现有 `pose_vec` 与 `VanillaPoseProjector` 不变**，不要同时重构它。

原因：

- 本轮只隔离 target-centric spatial grounding 的效果。
- 绝对 rotation / intrinsics 可能存在冗余，但暂不作为本轮变量。

后续如果 grounding 成立，再单独做 `pose token on/off` ablation。

---

# 3. Satellite positional encoding 改造

## 3.1 删除主路径中的 absolute global XY PE

当前 `_sat_patch_world_xy()` 返回：

```text
window_origin_global_xy + patch_offset
```

本轮主路径改成 target-relative planar coordinates。

建议新增：

```python
def _sat_patch_target_xy(
    self,
    window_origin_xyz,
    tgt_T_cam,
) -> torch.Tensor:
    ...
```

流程：

1. 根据 satellite crop grid + mpp 得到每个 patch 的 world XY。
2. 得到 target camera center world XY。
3. 计算 `delta_xy = patch_world_xy - target_center_xy`。
4. 根据 target camera heading 投影成：
   - `right_m`
   - `forward_m`
5. 返回 `(B, Ns, 2)`。

注意：

- Satellite crop 仍然是 window-shared crop。
- **不要按 target 重裁 satellite image**。
- 只改变 token 的几何坐标表达。
- 这样不会引入 crop-shift leakage。

## 3.2 暂停高频 Fourier PE，改为 normalized target-relative MLP PE

当前：

```python
sin(2^k x), cos(2^k x), k=0..9
```

直接作用于“米”且此前还是 absolute global XY，频率过高。

本轮默认改成简单 normalized coordinate features。

定义：

```python
R = sat_px * sat_m_per_px / 2
```

当前约：

```text
R ≈ 50.176 m
```

输入 feature：

```python
x = right_m / R
z = forward_m / R
rho = sqrt(x^2 + z^2)
theta = atan2(x, z)

feat = [
    x,
    z,
    rho,
    sin(theta),
    cos(theta),
]
```

新增 PE：

```python
class TargetRelativeSatPE(nn.Module):
    def __init__(self, d_model, hidden=256):
        ...
    def forward(self, xz):
        ...
```

MLP 建议：

```python
Linear(5, hidden)
GELU
Linear(hidden, d_model)
```

然后：

```python
sat_tokens = sat_proj(dino_sat) + target_rel_sat_pe(xz)
```

### 3.2.1 兼容旧 PE

为了后续 ablation，建议配置保留两种模式：

```yaml
sat_pe_mode: target_relative
```

可选：

```yaml
sat_pe_mode: legacy_fourier
```

但本轮主实验只训练 `target_relative`。

---

# 4. `icassp27_predictor.py` 具体修改

目标文件：

```text
world3d/models/icassp27_predictor.py
```

## 4.1 新增/替换模块

新增：

```python
class TargetRelativeSatPE(nn.Module):
    ...
```

保留旧 `MetricPE` 仅供 `legacy_fourier` ablation，不作为默认。

## 4.2 修改 satellite coordinate helper

旧：

```python
_sat_patch_world_xy(window_origin_xy)
```

新增：

```python
_sat_patch_target_xy(window_origin_xyz, tgt_T_cam)
```

输出：

```text
(B, Ns, 2) = (right_m, forward_m)
```

## 4.3 `build_memory()` 增加 target pose 参数

当前 `build_memory()` 没有拿到 `tgt_T_cam`，导致 satellite branch 无法构造 target-relative coordinates。

修改 signature：

```python
def build_memory(
    self,
    pose_vec,
    sat,
    window_origin_xyz,
    src_rgbs,
    rel_poses,
    src_mask=None,
    tgt_T_cam=None,
):
```

当 `use_sat=True` 且 `sat_pe_mode=target_relative`：

```python
assert tgt_T_cam is not None
```

调用：

```python
sat_xz = self._sat_patch_target_xy(window_origin_xyz, tgt_T_cam)
f_sat = self.sat_proj(f_sat) + self.sat_pe(sat_xz)
```

## 4.4 修改 `forward()`

当前：

```python
memory, key_padding = self.build_memory(...)
```

改为传入：

```python
tgt_T_cam=tgt_T_cam
```

然后 target ray 使用 camera-local ray，不再依赖 `window_origin_xyz`。

## 4.5 修改 `generate()`

同样：

```python
build_memory(..., tgt_T_cam=tgt_T_cam)
```

并使用 camera-local target ray。

确保 teacher-forcing 与 autoregressive generation 两条路径的 geometry 完全一致。

## 4.6 修改 `_target_rays()`

目标：

```python
d = normalize(K^-1 p)
o = zeros
```

不要：

```python
d = R_world @ d_cam
o = target_center - window_origin
```

保留 raster/token center 计算不变。

## 4.7 不修改 memory fusion

继续使用：

```text
[pose token | satellite tokens | source tokens]
```

继续使用标准 `nn.TransformerDecoderLayer` cross-attention。

**本轮不要加 gate。**

---

# 5. `kitti360_tuple_dataset.py` 具体修改

目标文件：

```text
world3d/data/kitti360_tuple_dataset.py
```

## 5.1 Source relative translation 改成 target frame

当前：

```python
dt = Ts[:3, 3] - T_tgt[:3, 3]
R_rel = T_tgt[:3, :3].T @ Ts[:3, :3]
```

修改：

```python
dt_world = Ts[:3, 3] - T_tgt[:3, 3]
dt_tgt = T_tgt[:3, :3].T @ dt_world
R_rel = T_tgt[:3, :3].T @ Ts[:3, :3]
```

然后：

```python
rel_pose = concat(dt_tgt, rot6d(R_rel))
```

## 5.2 保留 `window_origin_xyz`

仍然返回：

```python
window_origin_xyz
```

因为 model 需要从 window-shared satellite crop 恢复 patch world positions，再转到 target frame。

不要把 satellite crop 改成 target-centered crop。

## 5.3 增加 actual distance 字段

如果当前返回项里还没有可靠的 actual metric distance，则加入：

```python
actual_source_target_dist_m
```

定义优先使用：

```text
last source camera center ↔ target camera center
```

的 ground-plane Euclidean distance，另可保留 arc distance：

```python
actual_source_target_arc_m
```

后续评估分 bin 必须按 actual distance，而不是仅按 requested midpoint。

## 5.4 不在本轮重写完整 tuple sampler

当前 stochastic tuple reconstruction 虽然之后需要清理，但不是本轮 grounding 的关键变量。

本轮只保证：

- source/target pose 正确；
- actual distance 被记录；
- target-centric rel pose 正确。

---

# 6. Config 修改

目标：

```text
configs/icassp27_pilot.yaml
```

建议新增：

```yaml
model:
  geo: raymap
  sat_pe_mode: target_relative
  sat_coord_scale_m: 50.176   # 或由 sat_px * mpp / 2 自动计算
```

如果代码自动算 scale，config 可以只留：

```yaml
sat_pe_mode: target_relative
```

保留：

```yaml
legacy_fourier
```

作为后续 ablation option，但不在本轮主实验里跑。

---

# 7. 必须新增的 geometry unit tests / smoke tests

建议新增：

```text
icassp27_verify/v4_target_centric_geometry.py
```

或者扩展：

```text
icassp27_verify/smoke_test.py
```

要求覆盖以下检查。

## 7.1 Global translation invariance

对同一 synthetic geometry：

```text
all world positions + [1000, -500, 20]
```

target-relative satellite coordinates 应保持不变：

```python
max_abs_error < 1e-5
```

source target-relative translation 同样不变。

## 7.2 Global planar yaw invariance

对：

- target pose
- source pose
- satellite patch positions

整体绕 world Z 旋转同一角度。

target-centric：

```text
satellite (right, forward)
source translation
```

应保持近似不变。

允许浮点误差：

```python
max_abs_error < 1e-4
```

## 7.3 Target center sanity

目标 camera center 映射到 satellite local frame：

```text
(right, forward) = (0, 0)
```

## 7.4 Direction sanity

随机抽至少 10 个真实 tuple 做可视化：

- satellite crop
- target camera location
- target forward direction
- source camera location
- satellite patch coordinate axes

输出到：

```text
icassp27_verify/out/target_centric/
```

必须人工确认：

- forward 沿车辆实际行驶/视线方向；
- right/left 没有镜像；
- north-up satellite 到 camera-relative frame 的旋转方向正确。

这个 sanity check 很重要，不能仅靠数值 unit test。

## 7.5 Source distance sanity

检查：

```python
norm(dt_tgt) == norm(dt_world)
```

误差：

```python
< 1e-4
```

rotation 只改变坐标轴，不应改变距离。

## 7.6 Target ray sanity

中心 token ray 应接近：

```text
[0, 0, 1]
```

左侧 token：

```text
x < 0
```

右侧 token：

```text
x > 0
```

根据 KITTI camera convention 做实际验证。

---

# 8. 训练计划

因为本轮 target ray 与 source rel translation 都发生变化，**B0/B1/B2 都需要重新训练**，不能只重训 B0/B2 后直接和旧 B1 比。

## 8.1 Stage A：5k-step smoke pilots

先训练：

- B0-TC
- B1-TC
- B2-TC

TC = target-centric。

统一：

- seed
- dataset split
- tokenizer
- batch size
- sampling
- generation temperature/top-p
- checkpoint cadence

目标：

- loss 正常；
- 无 NaN；
- generation 无明显崩坏；
- B1 仍显著优于 B0；
- B0 distance curve 仍基本平；
- geometry unit tests 全过。

如果 5k 明显异常，不继续 20k。

## 8.2 Stage B：20k-step pilot

5k 通过后，将三组都跑到与旧 pilot 同样的：

```text
20,000 steps
```

新 run 名建议：

```text
runs/icassp27_tc_b0
runs/icassp27_tc_b1
runs/icassp27_tc_b2
```

不要覆盖旧结果。

---

# 9. 本轮最关键的 satellite grounding diagnostics

建议新增/整理统一脚本：

```text
scripts/eval_satellite_grounding.py
```

如果当前 real/zero/shuffle diagnostic 已有本地脚本，则将其版本化进 repo，并扩展下面条件。

要求：

- paired generation；
- 同一个 target/source tuple；
- 同一个 sampling seed；
- 只改变 satellite condition；
- 保存逐样本 metrics；
- 至少 48 个不同 target tuples；
- 建议 2 个 sampling seeds；
- 总 paired samples ≥ 96 更稳妥。

## 9.1 Real

正确 satellite crop + 正确 target-relative PE。

## 9.2 Zero

satellite memory 置零。

注意同时明确两种 zero 方式，至少实现一种并记录：

A. `encoded satellite token = 0`  
B. `sat image = 0` 后走正常 encoder

推荐主结果使用 A，因为它直接测试 memory contribution。

## 9.3 Cross-window Shuffle

替换为来自不同 window 的 satellite image。

要求：

- target/source/pose 不变；
- satellite image 换成另一个 geographic window；
- satellite token geometry PE 仍按当前 target/window coordinate pipeline 生成；
- 记录 shuffle source window id，保证不是同一地理邻域。

这是当前最重要的 negative control。

## 9.4 Spatial PE Permutation

保持正确 satellite DINO tokens 不变，但随机打乱其 target-relative coordinate PE：

```python
f_sat_j = visual_j + pe_perm(j)
```

若 spatial grounding 成立：

```text
real > PE-permuted
```

## 9.5 Satellite Token Spatial Permutation

另一项可选诊断：

- 同一张正确 satellite；
- 对 visual patch tokens 做 spatial permutation；
- PE 不 permutation。

即打破：

```text
visual content ↔ position
```

对应关系。

如果模型利用 layout：

```text
real > visual-permuted
```

## 9.6 90-degree Rotation Mismatch

把 satellite RGB 旋转 90°，但 **不要同步旋转 geometry PE**。

目的是制造：

```text
appearance layout ↔ metric coordinates
```

错配。

如果 grounding 成立：

```text
real > rot90-mismatch
```

注意：

- 这是 diagnostic，不是训练 augmentation。
- 不要把旋转后的 satellite 当作新的正确坐标。

---

# 10. Metrics 与统计

每个 paired diagnostic 至少报告：

- PSNR ↑
- LPIPS ↓

如果现有 DINO metric 工具稳定，也加：

- DINO similarity ↑

几何指标可在主 B0/B1/B2 checkpoint 上继续做：

- scale-aligned LiDAR AbsRel ↓

但 satellite grounding diagnostic 首轮不强制每个 perturbation 都跑 Metric3D，以节省成本。

## 10.1 定义 Satellite Content Sensitivity

建议显式记录：

```text
SCS-PSNR = PSNR(real) - PSNR(shuffle)
SCS-LPIPS = LPIPS(shuffle) - LPIPS(real)
```

正值代表正确 satellite 优于错配 satellite。

同时记录：

```text
ZeroGap-PSNR = PSNR(real) - PSNR(zero)
```

注意：

`real > zero` 不能证明 scene-specific grounding。

真正关键是：

```text
real > shuffle
real > spatial mismatch
```

## 10.2 统计检验

对 paired samples 做：

- paired bootstrap 95% CI，或
- Wilcoxon signed-rank test。

优先 bootstrap，因为同时可给 effect-size CI。

建议 acceptance threshold：

### Grounding Gate PASS

至少满足：

1. `mean SCS-PSNR > +0.20 dB`
2. `mean SCS-LPIPS > +0.01`
3. 两者 paired 95% CI 至少一个明确不跨 0，另一个方向一致；
4. `real > shuffle` 的逐样本 win-rate 明显 > 50%；
5. `real > PE-permutation` 或 `real > rot90-mismatch` 至少有一个成立。

阈值不是论文最终标准，只是项目继续决策门。

---

# 11. B0/B1/B2 本轮需要验证的主结果

## 11.1 基础 sanity

预期仍应成立：

```text
B1 >> B0
```

如果 target-centric 改造导致 B1 大幅退化，优先检查：

- ray axis；
- source relative translation axis；
- target/source rotation方向；
- teacher-forcing 与 generate 的 geometry 是否一致。

## 11.2 Distance behavior

按 **actual last-source → target distance** 分：

```text
[2,5)
[5,10)
[10,20]
```

报告：

- PSNR
- LPIPS
- AbsRel

继续观察：

```text
B0：随 distance 基本平
B1：distance 增大后逐渐下降
```

## 11.3 本轮 B2 的评价优先级

按以下顺序判断，不要只看最终平均 PSNR：

### Gate A：scene-specific satellite grounding

```text
B2(real) > B2(shuffle)
```

这是本轮首要目标。

### Gate B：satellite 是否有 task gain

再看：

```text
B2(real) vs B1
```

可能出现三种情况：

#### Case 1

```text
B2(real) > B2(shuffle)
B2(real) > B1
```

最好结果。

说明 grounding 与 fusion 当前都已有效。

下一步进入完整实验，不急着加 gate。

#### Case 2

```text
B2(real) > B2(shuffle)
B2(real) < B1
```

本轮也算成功。

说明：

- satellite 已有 scene-specific signal；
- 但 naive joint cross-attention 仍损伤 source evidence。

**只有这时，下一轮才正式实现 Evidence-Aware Fusion / gated fusion。**

#### Case 3

```text
B2(real) ≈ B2(shuffle)
```

target-centric geometry 仍未解决 grounding。

下一轮不要做 gate，应优先研究：

- satellite encoder 是否不适配；
- DINO satellite features 是否缺 layout-sensitive signal；
- satellite encoder finetune；
- SatMAE；
- coarse layout/road auxiliary supervision；
- satellite-target structural alignment。

---

# 12. Attention diagnostic 的使用原则

保留现有 attention mass diagnostic，但不要把它作为“使用 satellite content”的证据。

可以继续报告：

```text
attention mass to satellite
attention mass to source
```

但它只回答：

```text
decoder 是否访问该 branch
```

不回答：

```text
decoder 是否利用正确 scene-specific content
```

本轮最重要的 evidence 是 paired counterfactual：

```text
real vs wrong satellite
```

---

# 13. Logging / reproducibility 要求

每次 run 必须保存：

```text
git commit SHA
config snapshot
random seed
dataset manifest path
train/val/test split
checkpoint step
sampling temperature
top-p
sat_pe_mode
geo mode
use_sat
use_src
```

Satellite diagnostic JSONL 每条至少保存：

```json
{
  "tuple_id": "...",
  "drive": "...",
  "target_fid": 0,
  "source_fids": [],
  "window_id": 0,
  "actual_dist_m": 0.0,
  "sampling_seed": 0,
  "condition": "real|zero|shuffle|pe_perm|visual_perm|rot90",
  "shuffle_window_id": "...",
  "psnr": 0.0,
  "lpips": 0.0
}
```

并生成 aggregate summary JSON/CSV。

---

# 14. 建议的代码文件改动清单

## 必改

```text
world3d/models/icassp27_predictor.py
world3d/data/kitti360_tuple_dataset.py
configs/icassp27_pilot.yaml
```

## 建议新增

```text
icassp27_verify/v4_target_centric_geometry.py
scripts/eval_satellite_grounding.py
configs/icassp27_target_centric_pilot.yaml
```

建议新增 config，不覆盖旧 pilot config，以保留可复现对照。

## 视现有训练器接口决定是否小改

```text
world3d/train/train_icassp27.py
```

只做参数透传、logging、run naming 等必要修改。

不要在训练器中引入新 loss。

---

# 15. Codex 执行顺序

严格按以下顺序执行。

## Step 1：代码阅读确认

先读取并确认：

```text
world3d/models/icassp27_predictor.py
world3d/data/kitti360_tuple_dataset.py
world3d/train/train_icassp27.py
configs/icassp27_pilot.yaml
```

确认实际 tensor convention 后再改，不要假定轴方向。

## Step 2：实现 target-centric geometry

完成：

- satellite target-relative coordinates；
- normalized target-relative satellite PE；
- source translation target-frame 化；
- target ray camera-local 化。

## Step 3：写 unit tests

先通过：

- translation invariance；
- yaw invariance；
- ray direction；
- source distance；
- satellite local coordinate sanity。

## Step 4：生成真实 tuple 可视化

人工检查至少 10 个 window：

- forward/right；
- source/target；
- satellite orientation。

在这个步骤确认坐标轴正确之前，不开始正式训练。

## Step 5：5k B0/B1/B2 smoke pilots

确认训练稳定与 B1>B0 sanity。

## Step 6：20k B0/B1/B2 target-centric pilots

输出主 appearance + geometry 结果。

## Step 7：satellite grounding diagnostic

对 B2-TC 做：

- real
- zero
- cross-window shuffle
- PE permutation
- rot90 mismatch

至少 48 tuples × 2 seeds。

## Step 8：输出结论

最终给出一张 summary：

| Check | Result | PASS/FAIL |
|---|---|---|
| Geometry unit tests |  |  |
| B1 > B0 |  |  |
| real > zero |  |  |
| real > shuffle |  |  |
| real > PE-perm |  |  |
| real > rot90 mismatch |  |  |
| B2 vs B1 @ 3.5m |  |  |
| B2 vs B1 @ 7.5m |  |  |
| B2 vs B1 @ 15m |  |  |

并根据 Case 1/2/3 自动给出下一轮建议。

---

# 16. 本轮完成定义（Definition of Done）

本轮不是“代码能跑”就结束。

必须满足：

- [ ] 所有 satellite/source/ray geometry 的 reference frame 在代码注释中明确。
- [ ] satellite 主 PE 不再使用 absolute global XY。
- [ ] target ray 不再使用 window-local origin + world-frame direction。
- [ ] source translation 改为 target-frame。
- [ ] unit tests 通过。
- [ ] 真实 geometry visualization 人工确认无镜像/轴错。
- [ ] B0/B1/B2 target-centric 5k smoke 全部稳定。
- [ ] B0/B1/B2 target-centric 20k pilot 完成。
- [ ] real/zero/shuffle diagnostic 完成。
- [ ] 至少实现一种 spatial mismatch diagnostic。
- [ ] 输出 paired statistics，而不仅是均值。
- [ ] 明确判定属于 Case 1 / Case 2 / Case 3。

---

# 17. 本轮研究判断原则

不要为了得到 `B2 > B1` 而继续临时加模块。

本轮只回答：

> **模型是否开始利用“正确 satellite scene content”，而不是只利用 satellite branch 的 token distribution / generic prior？**

如果答案是 **是**：

下一轮才研究：

> 如何让 global satellite prior 在不覆盖 local street evidence 的情况下提供增益。

如果答案仍然是 **否**：

下一轮优先改 satellite representation/grounding，而不是 fusion gate。

这条决策原则必须保持，否则容易再次回到 MM26 式“不断堆模块但核心因果不清楚”的路径。
