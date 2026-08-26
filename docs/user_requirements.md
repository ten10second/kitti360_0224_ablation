# User Requirements — Persistent Georeferenced World State

## Research Direction

- 核心 claim：不同高度、不同时间到达的空间观测，被持续吸收到一个地理坐标固定、可更新、可查询的三维世界状态中。
- 卫星在车辆到达前初始化大范围低频静态结构；车端以固定长度 chunk 逐段到达，对同一状态做局部确认、细化和纠错。
- 状态由累计 LiDAR 定义的 world geometry 及其冻结 readers 约束，而不是由 dense-ground encoder 单独定义全部语义。
- 主证据是状态轨迹 \(Z_0\rightarrow Z_T\) 上的 visited / ahead 几何、update gain、retention，以及 held-out depth reader transfer。实验方案以 `docs/experiment_plan.md` 为准。
- chunk 只作为固定长度的车端测量包，不再做 chunk 内删帧或 spatial hole completion。
- 不主张卫星恢复精确立面、动态物体或无真值的 off-route 完整 3D。KITTI-360 v1 的 off-route 只作 coverage diagnostic。
- raw latent equality、VGGT、DINO、DPT 名称、加权 splat、校准概率 uncertainty 本身都不是 headline contribution。卫星骨干用冻结 DINOv2，只训写到 \(Z_0\) 的小 write head。

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
- Stage B / updater 不使用显式概率 uncertainty；confidence 是确定性的 normalized evidence strength。只报告 observation support / provenance / last_update。
- 主线使用 schema `world_state_v1`、独立脚本和 `runs/world_state_*` 输出目录。spatial hole-probe 已删除。
- satellite direct-height auxiliary 默认关闭或低权重，不能替代统一 latent 的冻结 geometry readout。
- `render_nadir()` 仅保留为 QA/可视化，不进入主训练损失。
- 首轮自动运行语法检查、单元测试和小规模 smoke test；20k/10k 全训练和多卡训练在代码 gate 通过后再启动。
- 当前 Git 仓库为既有仓库；本轮不自动 commit/push，除非用户另行要求。

## Document Preferences

- 语言：中文正文，代码符号与文件名保留英文。
- 文档服务于可审计实现，不扩写成论文草稿。
