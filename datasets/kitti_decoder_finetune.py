"""
KITTI Dataset for Decoder Fine-tuning
专门用于 VQ-VAE/VQGAN decoder 微调的数据集
只需要加载密集 RGB 图像，不需要 sparse RGB 和 mask
"""

import os
import glob
from PIL import Image
from typing import Optional, Callable, Tuple, List
import torch
import torchvision.transforms as T
from torchvision.datasets import VisionDataset


class KITTIDecoderFinetune(VisionDataset):
    """
    KITTI Decoder Fine-tuning 数据集
    
    数据结构：
    root/
    ├── 2011_09_26_drive_0001_sync/
    │   └── image_02/             # 密集RGB图像
    │       └── data/             # 图像文件 (可选子目录)
    │           ├── 0000000000.png
    │           ├── 0000000001.png
    │           └── ...
    ├── 2011_09_28_drive_0002_sync/
    │   └── image_02/
    │       └── data/
    └── ...

    或者简化结构（image_02 直接包含图像）：
    root/
    ├── 2011_09_26_drive_0001_sync/
    │   └── image_02/
    │       ├── 0000000000.png
    │       ├── 0000000001.png
    │       └── ...
    └── ...

    支持所有日期的KITTI数据 (2011_09_26, 2011_09_28, 2011_09_29, 2011_09_30, 2011_10_03)
    
    返回：
    - dense_rgb: 密集RGB图像 (3, H, W) 范围[-1, 1]
    - path: 图像路径 (用于调试)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (288, 960),  # (H, W) 调整为16的倍数
        transforms: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root=root, transforms=transforms)
        
        if split not in ['train', 'val', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        
        self.split = split
        self.img_size = img_size
        self.max_samples = max_samples
        
        # 扫描所有图像文件
        self.image_paths = self._scan_data()
        
        # 根据split划分数据
        if split != 'all':
            self.image_paths = self._split_data(self.image_paths, split)
        
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"KITTI Decoder Finetune Dataset: {len(self.image_paths)} samples for split '{split}'")
    
    def _find_drive_folders(self, root_path: str) -> List[str]:
        """递归查找所有匹配 *drive_*_sync 模式的文件夹"""
        drive_folders = []

        try:
            # 遍历当前目录
            for item in os.listdir(root_path):
                item_path = os.path.join(root_path, item)

                # 跳过非目录
                if not os.path.isdir(item_path):
                    continue

                # 跳过隐藏目录和常见的系统目录
                if item.startswith('.') or item in ['lost+found', 'System Volume Information']:
                    continue

                # 检查是否匹配 drive 文件夹模式 (任意日期的 drive_*_sync)
                if 'drive_' in item and item.endswith('_sync'):
                    drive_folders.append(item_path)
                    if os.environ.get('RANK', '0') == '0':
                        print(f"  Found drive folder: {item_path}")
                else:
                    # 递归查找子目录
                    drive_folders.extend(self._find_drive_folders(item_path))

        except PermissionError:
            # 跳过没有权限的目录
            if os.environ.get('RANK', '0') == '0':
                print(f"  Warning: Permission denied for {root_path}")
        except Exception as e:
            if os.environ.get('RANK', '0') == '0':
                print(f"  Warning: Error scanning {root_path}: {e}")

        return drive_folders

    def _scan_data(self) -> List[str]:
        """扫描数据文件，返回所有图像路径的列表"""
        image_paths = []

        if os.environ.get('RANK', '0') == '0':
            print(f"Recursively scanning for drive folders in: {self.root}")

        # 递归查找所有 drive 文件夹
        drive_folders = self._find_drive_folders(self.root)
        drive_folders = sorted(drive_folders)

        if len(drive_folders) == 0:
            if os.environ.get('RANK', '0') == '0':
                print(f"Warning: No drive folders found matching pattern '*drive_*_sync'")
                print(f"Searched in: {self.root}")
        else:
            if os.environ.get('RANK', '0') == '0':
                print(f"Found {len(drive_folders)} drive folders")

        for drive_folder in drive_folders:
            # 尝试两种可能的目录结构
            # 1. image_02/data/
            image_dir = os.path.join(drive_folder, "image_02", "data")
            if not os.path.isdir(image_dir):
                # 2. image_02/ (直接包含图像)
                image_dir = os.path.join(drive_folder, "image_02")
                if not os.path.isdir(image_dir):
                    if os.environ.get('RANK', '0') == '0':
                        print(f"Warning: Skipping {drive_folder} - missing image_02 directory")
                    continue

            # 获取所有图片文件
            img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            img_files = []
            for ext in img_extensions:
                img_files.extend(glob.glob(os.path.join(image_dir, ext)))

            if len(img_files) == 0:
                if os.environ.get('RANK', '0') == '0':
                    print(f"Warning: No images found in {image_dir}")
            else:
                if os.environ.get('RANK', '0') == '0':
                    print(f"  Loaded {len(img_files)} images from {os.path.basename(drive_folder)}")

            img_files = sorted(img_files)
            image_paths.extend(img_files)

        if len(image_paths) == 0:
            if os.environ.get('RANK', '0') == '0':
                print(f"ERROR: No images found in {self.root}")
                print(f"Please check:")
                print(f"  1. ROOT path is correct")
                print(f"  2. Drive folders exist (will search recursively)")
                print(f"  3. image_02/data/ or image_02/ contains images")

        return image_paths
    
    def _split_data(self, image_paths: List[str], split: str) -> List[str]:
        """根据split划分数据"""
        total = len(image_paths)
        
        if split == 'train':
            return image_paths[:int(0.8 * total)]
        elif split == 'val':
            return image_paths[int(0.8 * total):int(0.9 * total)]
        elif split == 'test':
            return image_paths[int(0.9 * total):]
        else:
            return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            if os.environ.get('RANK', '0') == '0':
                print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像作为fallback
            img = Image.new('RGB', (self.img_size[1], self.img_size[0]), (0, 0, 0))
        
        # 调整尺寸到目标大小
        H, W = self.img_size
        img = img.resize((W, H), Image.LANCZOS)
        
        # 转换为tensor并归一化到 [0, 1]
        img = T.ToTensor()(img)
        
        # 应用额外的transforms（如果有）——在 [0,1] 空间进行
        if self.transforms is not None:
            img = self.transforms(img)
        
        # 最后把 RGB 归一化到 [-1, 1]
        img = img * 2.0 - 1.0
        
        return {
            'dense_rgb': img,
            'path': img_path
        }


def create_kitti_decoder_transforms(img_size: Tuple[int, int], split: str = 'train'):
    """创建KITTI decoder fine-tuning数据集的transforms"""
    H, W = img_size
    
    if split == 'train':
        # 训练时可以加一些数据增强
        transforms = T.Compose([
            # 颜色增强（轻微）
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    else:
        # 验证/测试时不做增强
        transforms = None
    
    return transforms


# 测试函数
if __name__ == "__main__":
    import sys
    
    # 测试数据集加载
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "/path/to/your/processed_data"
        print(f"Usage: python {sys.argv[0]} <root_path>")
        print(f"Using default path: {root_path}")
    
    print(f"\n{'='*60}")
    print(f"Testing KITTI Decoder Finetune Dataset")
    print(f"{'='*60}\n")
    
    dataset = KITTIDecoderFinetune(
        root=root_path,
        split='train',
        img_size=(288, 960),
        max_samples=10
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print(f"\nLoading first sample...")
        sample = dataset[0]
        print(f"  Dense RGB shape: {sample['dense_rgb'].shape}")
        print(f"  Dense RGB range: [{sample['dense_rgb'].min():.3f}, {sample['dense_rgb'].max():.3f}]")
        print(f"  Sample path: {sample['path']}")
        
        print(f"\nDataset structure looks good! ✓")
    else:
        print(f"\nERROR: Dataset is empty!")
        print(f"Please check your data structure.")

