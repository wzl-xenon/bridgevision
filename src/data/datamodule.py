# src/data/datamodule.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class VisionDataModule:
    """
    Simple vision datamodule / 简单视觉数据模块

    当前第一版支持 / Supported in version 1:
    - Oxford-IIIT Pets
    - Flowers102

    设计目标 / Design goals:
    1. 先把自然图像分类训练闭环跑通
       Make the natural-image classification pipeline work first
    2. 输入分辨率与当前 ResNet50 / ViT-B-16 baseline 保持一致
       Keep input resolution consistent with current ResNet50 / ViT-B-16 baseline
    3. 对没有官方 val split 的数据集，在代码里做稳定划分
       For datasets without official val split, create a stable split in code
    """

    SUPPORTED_DATASETS = {"oxfordiiitpet", "flowers102"}

    def __init__(
        self,
        dataset_name: Literal["oxfordiiitpet", "flowers102"] = "oxfordiiitpet",
        data_root: str = "./data",
        image_size: int = 224,
        batch_size: int = 4,
        num_workers: int = 0,
        download: bool = True,
        pin_memory: bool = True,
        use_imagenet_norm: bool = True,
        split_seed: int = 42,
        pet_train_ratio: float = 0.9,
    ) -> None:
        """
        Args:
            dataset_name:
                数据集名称
                Dataset name

            data_root:
                数据集根目录
                Dataset root directory

            image_size:
                模型输入尺寸，第一版建议 224
                Model input size, 224 is recommended for v1

            batch_size:
                batch 大小
                Batch size

            num_workers:
                DataLoader worker 数量
                Number of DataLoader workers

            download:
                是否自动下载
                Whether to download automatically

            pin_memory:
                是否启用 pin_memory
                Whether to enable pin_memory

            use_imagenet_norm:
                是否使用 ImageNet 的 mean/std
                Whether to use ImageNet mean/std

            split_seed:
                划分 train/val 时使用的随机种子
                Random seed used for train/val split

            pet_train_ratio:
                Oxford-IIIT Pets 的 trainval 中用于训练集的比例
                Ratio of train subset within Oxford-IIIT Pets trainval split
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name}. "
                f"Supported: {sorted(self.SUPPORTED_DATASETS)}"
            )

        if not (0.0 < pet_train_ratio < 1.0):
            raise ValueError(f"pet_train_ratio must be in (0, 1), but got {pet_train_ratio}.")

        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.pin_memory = pin_memory
        self.use_imagenet_norm = use_imagenet_norm
        self.split_seed = split_seed
        self.pet_train_ratio = pet_train_ratio

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if self.dataset_name == "oxfordiiitpet":
            self.num_classes = 37
        elif self.dataset_name == "flowers102":
            self.num_classes = 102
        else:
            raise RuntimeError(f"Unexpected dataset_name={self.dataset_name}")

    def _build_normalize(self) -> transforms.Normalize:
        """
        构建归一化 / Build normalization
        """
        if self.use_imagenet_norm:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        return transforms.Normalize(mean=mean, std=std)

    def build_train_transform(self) -> transforms.Compose:
        """
        训练集变换 / Train transform
        """
        normalize = self._build_normalize()

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform

    def build_eval_transform(self) -> transforms.Compose:
        """
        验证/测试集变换 / Eval transform
        """
        normalize = self._build_normalize()
        resize_size = int(round(self.image_size / 224 * 256))

        eval_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return eval_transform

    def _build_split_indices(self, num_samples: int, train_ratio: float) -> tuple[list[int], list[int]]:
        """
        根据固定随机种子生成 train/val 下标
        Build deterministic train/val indices with a fixed random seed
        """
        generator = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(num_samples, generator=generator).tolist()

        num_train = int(num_samples * train_ratio)
        train_indices = perm[:num_train]
        val_indices = perm[num_train:]

        return train_indices, val_indices

    def _build_oxfordiiitpet(self) -> None:
        """
        构建 Oxford-IIIT Pets 数据集
        Build Oxford-IIIT Pets datasets

        说明 / Notes:
        - 官方只有 trainval / test
          Officially only trainval / test are provided
        - 所以这里把 trainval 稳定切成 train / val
          So here we split trainval into train / val deterministically
        - official test 保持不动，作为真正测试集
          The official test split is kept unchanged as the real test set
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        # 同一个官方 trainval split，分别构建 train-transform 和 eval-transform 版本
        # Build two copies of official trainval split with different transforms
        base_trainval_for_train = datasets.OxfordIIITPet(
            root=str(self.data_root),
            split="trainval",
            target_types="category",
            transform=train_transform,
            download=self.download,
        )

        base_trainval_for_eval = datasets.OxfordIIITPet(
            root=str(self.data_root),
            split="trainval",
            target_types="category",
            transform=eval_transform,
            download=self.download,
        )

        train_indices, val_indices = self._build_split_indices(
            num_samples=len(base_trainval_for_train),
            train_ratio=self.pet_train_ratio,
        )

        self.train_dataset = Subset(base_trainval_for_train, train_indices)
        self.val_dataset = Subset(base_trainval_for_eval, val_indices)

        self.test_dataset = datasets.OxfordIIITPet(
            root=str(self.data_root),
            split="test",
            target_types="category",
            transform=eval_transform,
            download=self.download,
        )

    def _build_flowers102(self) -> None:
        """
        构建 Flowers102 数据集
        Build Flowers102 datasets

        说明 / Notes:
        - Flowers102 官方就有 train / val / test
          Flowers102 already has official train / val / test
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        self.train_dataset = datasets.Flowers102(
            root=str(self.data_root),
            split="train",
            transform=train_transform,
            download=self.download,
        )

        self.val_dataset = datasets.Flowers102(
            root=str(self.data_root),
            split="val",
            transform=eval_transform,
            download=self.download,
        )

        self.test_dataset = datasets.Flowers102(
            root=str(self.data_root),
            split="test",
            transform=eval_transform,
            download=self.download,
        )

    def setup(self) -> None:
        """
        构建数据集对象 / Build dataset objects
        """
        if self.dataset_name == "oxfordiiitpet":
            self._build_oxfordiiitpet()
        elif self.dataset_name == "flowers102":
            self._build_flowers102()
        else:
            raise RuntimeError(f"Unexpected dataset_name={self.dataset_name}")

    def train_dataloader(self) -> DataLoader:
        """
        训练集 DataLoader / Train DataLoader
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Please call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """
        验证集 DataLoader / Validation DataLoader
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Please call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """
        测试集 DataLoader / Test DataLoader
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Please call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def _demo_datamodule(dataset_name: str = "oxfordiiitpet") -> None:
    """
    简单测试 / Simple test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = VisionDataModule(
        dataset_name=dataset_name,
        data_root="./data",
        image_size=224,
        batch_size=4,
        num_workers=0,
        download=True,
        pin_memory=torch.cuda.is_available(),
        use_imagenet_norm=True,
        split_seed=42,
        pet_train_ratio=0.9,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    print(f"==== VisionDataModule Test ({dataset_name}) ====")
    print("Device                :", device)
    print("Num classes           :", dm.num_classes)
    print("Train dataset size    :", len(dm.train_dataset))
    print("Val dataset size      :", len(dm.val_dataset))
    print("Test dataset size     :", len(dm.test_dataset))
    print("Train images shape    :", train_images.shape)
    print("Train labels shape    :", train_labels.shape)
    print("Val images shape      :", val_images.shape)
    print("Val labels shape      :", val_labels.shape)
    print("Test images shape     :", test_images.shape)
    print("Test labels shape     :", test_labels.shape)
    print("Train min / max       :", train_images.min().item(), train_images.max().item())


if __name__ == "__main__":
    _demo_datamodule("oxfordiiitpet")