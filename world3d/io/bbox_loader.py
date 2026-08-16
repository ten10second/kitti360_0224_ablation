#!/usr/bin/env python3
"""
Bounding Box Data Loader - 加载3D和2D检测框

需要的数据格式：

1. 3D Bounding Boxes (LiDAR坐标系):
   列: x y z l w h yaw [class_name] [score]
   
   - x, y, z: bbox中心在LiDAR坐标系下的位置（米）
   - l, w, h: bbox的长宽高（米）
   - yaw: bbox的朝向角（弧度，相对于LiDAR坐标系的x轴）
   - class_name: 类别名称（可选，如 'Car', 'Pedestrian', 'Cyclist'）
   - score: 置信度（可选，0-1之间）
   
2. 2D Bounding Boxes (图像坐标系):
   列: row_min row_max col_min col_max [class_name] [score]
   
   - row_min, row_max: 行范围（像素坐标，0-based）
   - col_min, col_max: 列范围（像素坐标，0-based）
   - class_name: 类别名称（可选）
   - score: 置信度（可选）


3. 坐标系说明：
   - LiDAR坐标系: x向前，y向左，z向上
   - 图像坐标系: row向下，col向右
"""

import os
import numpy as np
from typing import Tuple, Optional, List


def load_3d_bboxes(bbox_3d_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    加载3D检测框
    
    Args:
        bbox_3d_path: 3D bbox文件路径
    
    Returns:
        bboxes: (N, 7) - [x, y, z, l, w, h, yaw]
        classes: List[str] - 类别名称列表（长度N）
        scores: (N,) - 置信度
    """
    if not os.path.exists(bbox_3d_path):
        return np.zeros((0, 7), dtype=np.float32), [], np.zeros(0, dtype=np.float32)
    
    bboxes = []
    classes = []
    scores = []
    
    with open(bbox_3d_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 7:
                continue
            
            # 解析坐标和尺寸
            try:
                x, y, z, l, w, h, yaw = map(float, parts[:7])
            except ValueError:
                continue
            
            bboxes.append([x, y, z, l, w, h, yaw])
            
            # 解析类别（可选）
            if len(parts) > 7:
                classes.append(parts[7])
            else:
                classes.append('Unknown')
            
            # 解析置信度（可选）
            if len(parts) > 8:
                try:
                    score = float(parts[8])
                except ValueError:
                    score = 1.0
            else:
                score = 1.0
            scores.append(score)
    
    if len(bboxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), [], np.zeros(0, dtype=np.float32)
    
    return np.array(bboxes, dtype=np.float32), classes, np.array(scores, dtype=np.float32)


def load_2d_bboxes(bbox_2d_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    加载2D检测框
    
    Args:
        bbox_2d_path: 2D bbox文件路径
    
    Returns:
        bboxes: (N, 4) - [row_min, row_max, col_min, col_max]
        classes: List[str] - 类别名称列表（长度N）
        scores: (N,) - 置信度
    """
    if not os.path.exists(bbox_2d_path):
        return np.zeros((0, 4), dtype=np.float32), [], np.zeros(0, dtype=np.float32)
    
    bboxes = []
    classes = []
    scores = []
    
    with open(bbox_2d_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            # 解析坐标
            try:
                row_min, row_max, col_min, col_max = map(float, parts[:4])
            except ValueError:
                continue
            
            bboxes.append([row_min, row_max, col_min, col_max])
            
            # 解析类别（可选）
            if len(parts) > 4:
                classes.append(parts[4])
            else:
                classes.append('Unknown')
            
            # 解析置信度（可选）
            if len(parts) > 5:
                try:
                    score = float(parts[5])
                except ValueError:
                    score = 1.0
            else:
                score = 1.0
            scores.append(score)
    
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], np.zeros(0, dtype=np.float32)
    
    return np.array(bboxes, dtype=np.float32), classes, np.array(scores, dtype=np.float32)


def load_frame_bboxes(
    img_path: str,
    seq_dir: Optional[str] = None,
    bbox_dir: Optional[str] = None,
    score_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    加载一帧的3D和2D检测框
    
    搜索顺序：
    1. 如果指定了bbox_dir，在bbox_dir/bbox_3d和bbox_dir/bbox_2d中搜索
    2. 在seq_dir/bbox_3d和seq_dir/bbox_2d中搜索
    3. 在img_path的父目录的父目录中搜索（KITTI结构）
    
    Args:
        img_path: 图像路径
        seq_dir: 序列目录（可选）
        bbox_dir: bbox根目录（可选）
        score_threshold: 置信度阈值，低于此值的bbox会被过滤
    
    Returns:
        bbox_3d: (N, 7) - [x, y, z, l, w, h, yaw]
        bbox_2d: (N, 4) - [row_min, row_max, col_min, col_max]
        classes: List[str] - 类别名称列表（长度N）
        scores: (N,) - 置信度
    """
    # 获取frame_id
    base_name = os.path.basename(img_path)
    frame_id = os.path.splitext(base_name)[0]
    
    # 构建搜索路径
    search_dirs = []
    
    if bbox_dir:
        search_dirs.append(bbox_dir)
    
    if seq_dir:
        search_dirs.append(seq_dir)
    
    # KITTI结构: .../image_0X/data/xxxx.png -> .../
    img_parent = os.path.dirname(img_path)  # .../image_0X/data
    img_grandparent = os.path.dirname(img_parent)  # .../image_0X
    seq_dir_inferred = os.path.dirname(img_grandparent)  # .../
    search_dirs.append(seq_dir_inferred)
    
    # 搜索bbox文件
    bbox_3d_path = None
    bbox_2d_path = None
    
    for search_dir in search_dirs:
        # 尝试bbox_3d
        candidate_3d = os.path.join(search_dir, 'bbox_3d', f'{frame_id}.txt')
        if os.path.exists(candidate_3d):
            bbox_3d_path = candidate_3d

        # 尝试bbox_2d
        candidate_2d = os.path.join(search_dir, 'bbox_2d', f'{frame_id}.txt')
        if os.path.exists(candidate_2d):
            bbox_2d_path = candidate_2d

        if bbox_3d_path and bbox_2d_path:
            break
    
    # 加载bbox
    if bbox_3d_path:
        bbox_3d, classes_3d, scores_3d = load_3d_bboxes(bbox_3d_path)
    else:
        bbox_3d = np.zeros((0, 7), dtype=np.float32)
        classes_3d = []
        scores_3d = np.zeros(0, dtype=np.float32)
    
    if bbox_2d_path:
        bbox_2d, classes_2d, scores_2d = load_2d_bboxes(bbox_2d_path)
    else:
        bbox_2d = np.zeros((0, 4), dtype=np.float32)
        classes_2d = []
        scores_2d = np.zeros(0, dtype=np.float32)
    
    # 确保3D和2D bbox数量一致
    n_3d = len(bbox_3d)
    n_2d = len(bbox_2d)
    
    if n_3d != n_2d:
        # 取较小的数量
        n = min(n_3d, n_2d)
        bbox_3d = bbox_3d[:n]
        bbox_2d = bbox_2d[:n]
        classes = classes_3d[:n] if classes_3d else classes_2d[:n]
        scores = scores_3d[:n] if len(scores_3d) > 0 else scores_2d[:n]
    else:
        classes = classes_3d if classes_3d else classes_2d
        scores = scores_3d if len(scores_3d) > 0 else scores_2d
    
    # 过滤低置信度的bbox
    if score_threshold > 0 and len(scores) > 0:
        mask = scores >= score_threshold
        bbox_3d = bbox_3d[mask]
        bbox_2d = bbox_2d[mask]
        classes = [c for c, m in zip(classes, mask) if m]
        scores = scores[mask]

    return bbox_3d, bbox_2d, classes, scores


def project_3d_bbox_to_2d(
    bbox_3d: np.ndarray,
    K: np.ndarray,
    T_cam_lidar: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    将3D bbox投影到2D图像（如果没有2D bbox标注，可以用这个函数生成）
    
    Args:
        bbox_3d: (N, 7) - [x, y, z, l, w, h, yaw]
        K: (3, 3) - 相机内参
        T_cam_lidar: (4, 4) - LiDAR到相机的变换矩阵
        img_h: 图像高度
        img_w: 图像宽度
    
    Returns:
        bbox_2d: (N, 4) - [row_min, row_max, col_min, col_max]
    """
    N = len(bbox_3d)
    if N == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    bbox_2d = []
    
    for i in range(N):
        x, y, z, l, w, h, yaw = bbox_3d[i]
        
        # 8个角点（LiDAR坐标系）
        corners_lidar = np.array([
            [l/2, w/2, h/2],
            [l/2, w/2, -h/2],
            [l/2, -w/2, h/2],
            [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [-l/2, -w/2, -h/2],
        ])
        
        # 旋转
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        corners_lidar = corners_lidar @ R.T
        
        # 平移
        corners_lidar += np.array([x, y, z])
        
        # 转换到相机坐标系
        corners_lidar_homo = np.hstack([corners_lidar, np.ones((8, 1))])
        corners_cam = corners_lidar_homo @ T_cam_lidar.T
        corners_cam = corners_cam[:, :3]
        
        # 过滤掉相机后面的点
        valid = corners_cam[:, 2] > 0
        if not valid.any():
            # 所有点都在相机后面，跳过
            bbox_2d.append([0, 0, 0, 0])
            continue
        
        corners_cam = corners_cam[valid]
        
        # 投影到图像
        corners_img = corners_cam @ K.T
        corners_img = corners_img[:, :2] / corners_img[:, 2:3]
        
        # 计算2D bbox
        col_min = max(0, int(np.floor(corners_img[:, 0].min())))
        col_max = min(img_w, int(np.ceil(corners_img[:, 0].max())))
        row_min = max(0, int(np.floor(corners_img[:, 1].min())))
        row_max = min(img_h, int(np.ceil(corners_img[:, 1].max())))
        
        bbox_2d.append([row_min, row_max, col_min, col_max])
    
    return np.array(bbox_2d, dtype=np.float32)
