# Geometry Utilities

几何变换和几何损失相关的工具函数。

## 文件说明

### 核心模块

#### `kitti_transforms.py`
KITTI 数据集的坐标变换和标定文件加载。

**主要功能**：
- `load_kitti_calib()` - 加载相机内参和外参
- `load_imu_to_velo_calib()` - 加载 IMU 到 Lidar 的变换
- `load_oxts_pose()` - 加载 GPS/IMU 姿态数据
- `compose_camera_to_satellite_transform()` - 组合相机到卫星图的完整变换链
- `invert_se3()` - SE(3) 变换矩阵求逆
- `latlon_to_utm()` - GPS 坐标转 UTM 坐标

**变换链**：
```
T_C→W' = T_W→W' · T_I→W · T_V→I · T_C→V
```
- T_C→V: Camera → Lidar (from calib files)
- T_V→I: Lidar → IMU (from calib files)
- T_I→W: IMU → World/GPS (from OXTS files)
- T_W→W': World → Satellite map (fixed)

---

#### `pose_encoding.py`
相机姿态编码，用于将 {R, T} 编码为特征向量。

**主要功能**：
- `PoseEncoder` - 姿态编码器类
- `rotation_matrix_to_6d()` - 旋转矩阵转 6D 表示
- `rotation_matrix_to_quaternion()` - 旋转矩阵转四元数

**用途**：
在 MaskGIT 模型中，将相机到卫星图的几何变换 {R, T} 编码为特征向量，
通过 cross-attention 融合到 BEV 特征中，替代原来的 yaw 角旋转。

---

#### `homography.py`
单应性矩阵计算（基于地面平面假设）。

**主要功能**：
- `compute_homography_ground_plane()` - 计算单应性矩阵 H = K · (R - T·n^T / d)
- `compute_homography_from_transform()` - 从 SE(3) 变换矩阵计算单应性矩阵
- `extract_R_T_from_transform()` - 从 SE(3) 变换矩阵提取 R 和 T

**公式**：
```
H = K · (R - T·n^T / d)
```
- K: 相机内参矩阵
- R: 旋转矩阵
- T: 平移向量
- n: 地面平面法向量
- d: 地面平面距离

**注意**：
虽然实现了单应性矩阵计算，但在实际的几何损失中，我们使用的是
`bev_to_camera_warp.py` 中的射线投影方法，因为：
1. 卫星图是正交投影（BEV）
2. 相机图是透视投影
3. 传统的单应性矩阵假设两个图像都是平面投影，不适用于这种情况

---

#### `bev_to_camera_warp.py`
将 BEV 卫星图投影到相机图像平面。

**主要功能**：
- `warp_bev_to_camera()` - 卫星图 → 相机图的 warp

**流程**：
1. 对于相机图的每个像素 (u_cam, v_cam)
2. 反投影到 3D 射线
3. 与地面平面 (Z=0) 相交 → 得到 3D 点
4. 转换为卫星图像素 (u_sat, v_sat)
5. 从卫星图采样

**返回**：
- `warped_sat`: (B, C, H_cam, W_cam) - Warp 后的卫星图
- `valid_mask`: (B, 1, H_cam, W_cam) - 有效像素掩码

**关键点**：
- 使用射线与地面平面相交，而不是单应性矩阵
- 返回有效性掩码，因为相机图的上半部分通常是天空，无法投影到地面
- 典型的有效像素比例约 50%

---

#### `pose_loss.py`
几何损失 $\mathcal{L}_{pose}$ 的实现。

**主要功能**：
- `PoseLoss` - 几何损失模块

**流程**：
1. GT Tokens → VQGAN Decode → 卫星图 x̂_sat
2. 卫星图 x̂_sat → 通过射线投影到相机平面 → x̂_sat_warped
3. L_pose = ||x̂_sat_warped - x_cam|| (只计算有效区域)

**参数**：
- `loss_type`: 'l1', 'l2', 或 'smooth_l1'
- `reduction`: 'mean' 或 'sum'

**用途**：
在训练 MaskGIT 模型时，作为辅助损失，确保生成的卫星图在几何上与
相机图一致。

---

## 使用示例

### 1. 加载 KITTI 变换

```python
from utils.geometry import (
    load_kitti_calib,
    load_imu_to_velo_calib,
    load_oxts_pose,
    compose_camera_to_satellite_transform,
    invert_se3,
)

# 加载相机内参
K, T_velo_to_cam = load_kitti_calib(calib_dir, cam='P2')

# 加载 IMU 到 Lidar 的变换
T_imu_to_velo = load_imu_to_velo_calib(calib_dir)

# 加载 GPS/IMU 姿态
T_imu_to_world, oxts_data = load_oxts_pose(oxts_file, origin=None)

# 组合完整变换链
T_cam_to_velo = invert_se3(T_velo_to_cam)
T_velo_to_imu = invert_se3(T_imu_to_velo)
T_cam_to_sat, T_world_to_sat = compose_camera_to_satellite_transform(
    T_cam_to_velo, T_velo_to_imu, T_imu_to_world,
    sat_size=512, resolution_m_per_px=0.2
)
```

### 2. 计算几何损失

```python
from utils.geometry import PoseLoss

# 创建损失模块
pose_loss = PoseLoss(loss_type='l1', reduction='mean')

# 计算损失
result = pose_loss(
    gt_tokens=sat_tokens,        # (B, L) - GT token 序列
    vqgan_tokenizer=vqgan,       # VQGAN tokenizer
    cam_image=cam_img,           # (B, 3, H_cam, W_cam) - 相机图像
    K=K,                         # (3, 3) - 相机内参
    T_cam_to_world=T_cam_to_world,  # (4, 4) - 相机到世界的变换
    resolution=0.2,              # 卫星图分辨率 (m/pixel)
    mask=None,                   # 可选的掩码
)

loss = result['loss']            # 几何损失
warped_sat = result['warped_sat']  # Warp 后的卫星图
valid_mask = result['valid_mask']  # 有效像素掩码
```

### 3. 姿态编码

```python
from utils.geometry import PoseEncoder

# 创建姿态编码器
pose_encoder = PoseEncoder(
    mode='6d',           # '6d', 'quaternion', 或 'matrix'
    include_translation=True,
    normalize_translation=True,
    translation_scale=100.0,
)

# 编码姿态
pose_embedding = pose_encoder(T_cam_to_sat)  # (B, D)
```

---

## 坐标系说明

### KITTI 坐标系

- **Camera (C)**: X=right, Y=down, Z=forward (optical axis)
- **Lidar/Velodyne (V)**: X=forward, Y=left, Z=up
- **IMU/GPS (I)**: X=forward, Y=left, Z=up
- **World/UTM (W)**: X=east (Easting), Y=north (Northing), Z=up (Altitude)

### 卫星图坐标系 (W')

- **North-up**: 北向上
- **col (u)**: 向东增加
- **row (v)**: 向南增加（row 0 = 北边缘）
- **分辨率**: 0.2 m/pixel (固定值)
- **尺寸**: 1280×1280 (覆盖范围 ±128m)

---

## 注意事项

1. **卫星图分辨率固定为 0.2 m/pixel**，不应该调整
2. **有效像素比例**：由于相机图的上半部分通常是天空，有效像素比例约 50%
3. **地面平面假设**：假设地面是平面 (Z=0)，对于起伏地形可能不准确
4. **相机高度**：KITTI 数据集中相机高度约 0.7-1.0 m

