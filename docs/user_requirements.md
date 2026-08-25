# User Requirements — Unified BEV Claim Probe

## Research Direction

- 核心 claim：先由 dense ground 定义并冻结 `ground-generative BEV latent space`；再由配准卫星图与 sparse ground 恢复该空间中的场景状态。
- 卫星负责全局布局与几何先验，稀疏街景负责局部真实外观；不主张卫星恢复不可见立面的真实高频纹理。
- 主证据必须来自同一个 Stage-A ground-trained、Stage-B frozen 的 RGB/depth renderer 与 geometry readout。
- aligned satellite 必须在地理隔离测试集上优于 ground-only、fixed-XY、misaligned/random satellite，尤其是在 sparse-ground 未支持区域的冻结几何指标上。
- raw latent equality、VGGT、DPT 名称、加权 splat、uncertainty 本身都不是 headline contribution。

## Data and Geometry Constraints

- KITTI-360 数据根目录：`/media/shizhm/Lenovo/KITTI-360`。
- LiDAR 根目录：`/media/shizhm/sda2/KITTI360_lidar/data_3d_raw`。
- 卫星分辨率固定使用 `0.196 m/px`。
- VGGT 权重：`/home/shizhm/Downloads/vggt.pt`；多帧联合视图推理后用车辆运动校正 metric scale。`Ns=1` 没有帧间位移，必须明确标记为 calibrated-camera-rig fallback；`Ns=2` 只有一个 motion baseline，输出中必须保留 scale source、pair count、MAD 与 pose RMSE 供审计。
- 每个互斥 source subset 必须独立执行 VGGT forward，禁止从更大的联合推理结果切片。
- BEV 使用 south-up raster、pixel-center 约定；卫星原图仍是 north-up，进入 BEV 时统一翻转。
- Metric3D 若保留，仅使用围绕光轴居中的两个前视 crop；不把外围缺失误写成全图覆盖。
- 不恢复已删除的一致性加权 splat；初版使用普通双线性均值聚合。

## Implementation Constraints

- 框架沿用现有 PyTorch 代码与 `maskgit` conda 环境。
- 保护当前未提交修改，不回退用户已有工作。
- `coordinate_only` 只使用固定公制 relative-XY/Fourier encoding，不允许可学习逐 cell 表。
- Stage B 不使用显式概率 uncertainty；只报告确定性的 observation support/provenance。
- satellite direct-height auxiliary 默认关闭或低权重，不能替代统一 latent 的冻结 geometry readout。
- `render_nadir()` 仅保留为 QA/可视化，不进入主训练损失。
- 首轮自动运行语法检查、单元测试和小规模 smoke test；20k/10k 全训练和多卡训练在代码 gate 通过后再启动。
- 当前 Git 仓库为既有仓库；本轮不自动 commit/push，除非用户另行要求。

## Document Preferences

- 语言：中文正文，代码符号与文件名保留英文。
- 文档服务于可审计实现，不扩写成论文草稿。
