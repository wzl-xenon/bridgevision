from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DatasetName = Literal[
    "oxfordiiitpet",
    "flowers102",
    "dtd",
    "fgvc_aircraft",
    "country211",
    "food101",
]


class VisionDataModule:
    """
    Simple vision datamodule / 简单视觉数据模块

    当前扩展版支持 / Supported in the expanded version:
    - Oxford-IIIT Pets
    - Flowers102
    - DTD
    - FGVC Aircraft
    - Country211
    - Food101

    设计目标 / Design goals:
    1. 统一对外暴露 train / val / test 三个 split
       Always expose train / val / test splits externally
    2. 优先使用官方 split；如果没有官方 val，则在代码中稳定划分
       Prefer official splits; if official val is absent, create a deterministic split in code
    3. 让不同数据集尽量复用同一套训练入口
       Let different datasets share the same training entry as much as possible
    """

    SUPPORTED_DATASETS = {
        "oxfordiiitpet",
        "flowers102",
        "dtd",
        "fgvc_aircraft",
        "country211",
        "food101",
    }

    def __init__(
        self,
        dataset_name: DatasetName = "oxfordiiitpet",
        data_root: str = "./data",
        image_size: int = 224,
        batch_size: int = 4,
        num_workers: int = 0,
        download: bool = True,
        pin_memory: bool = True,
        use_imagenet_norm: bool = True,
        split_seed: int = 42,
        pet_train_ratio: float = 0.9,
        food101_train_ratio: float = 0.9,
        dtd_partition: int = 1,
        aircraft_annotation_level: Literal["variant", "family", "manufacturer"] = "variant",
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
                模型输入尺寸，当前建议 224
                Model input size, 224 is currently recommended

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

            food101_train_ratio:
                Food101 的官方 train split 中用于训练集的比例
                Ratio of train subset within Food101 official train split

            dtd_partition:
                DTD 的官方 partition 编号，范围 1~10
                Official DTD partition id, should be in [1, 10]

            aircraft_annotation_level:
                FGVC Aircraft 的标签粒度
                Annotation granularity for FGVC Aircraft
                - variant / family / manufacturer
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name}. "
                f"Supported: {sorted(self.SUPPORTED_DATASETS)}"
            )

        if not (0.0 < pet_train_ratio < 1.0):
            raise ValueError(f"pet_train_ratio must be in (0, 1), but got {pet_train_ratio}.")

        if not (0.0 < food101_train_ratio < 1.0):
            raise ValueError(
                f"food101_train_ratio must be in (0, 1), but got {food101_train_ratio}."
            )

        if not (1 <= dtd_partition <= 10):
            raise ValueError(f"dtd_partition must be in [1, 10], but got {dtd_partition}.")

        if aircraft_annotation_level not in {"variant", "family", "manufacturer"}:
            raise ValueError(
                "aircraft_annotation_level must be one of "
                "['variant', 'family', 'manufacturer']."
            )

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
        self.food101_train_ratio = food101_train_ratio
        self.dtd_partition = dtd_partition
        self.aircraft_annotation_level = aircraft_annotation_level

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.num_classes = self._infer_num_classes()

    def _infer_num_classes(self) -> int:
        """
        根据数据集配置推断类别数
        Infer number of classes from dataset configuration
        """
        if self.dataset_name == "oxfordiiitpet":
            return 37
        if self.dataset_name == "flowers102":
            return 102
        if self.dataset_name == "dtd":
            return 47
        if self.dataset_name == "fgvc_aircraft":
            if self.aircraft_annotation_level == "variant":
                return 100
            if self.aircraft_annotation_level == "family":
                return 70
            if self.aircraft_annotation_level == "manufacturer":
                return 30
            raise RuntimeError(
                f"Unexpected aircraft_annotation_level={self.aircraft_annotation_level}"
            )
        if self.dataset_name == "country211":
            return 211
        if self.dataset_name == "food101":
            return 101

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

    def _build_split_indices(
        self,
        num_samples: int,
        train_ratio: float,
    ) -> tuple[list[int], list[int]]:
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

    def _build_split_from_two_copies(
        self,
        train_dataset_for_train_transform,
        train_dataset_for_eval_transform,
        train_ratio: float,
    ) -> tuple[Subset, Subset]:
        """
        从同一个官方 train/trainval split 中切分 train / val
        Split train / val from the same official train or trainval split

        之所以需要两份 dataset，是为了让 train 和 val 用不同的 transform
        We need two dataset copies so that train and val can use different transforms
        """
        train_indices, val_indices = self._build_split_indices(
            num_samples=len(train_dataset_for_train_transform),
            train_ratio=train_ratio,
        )

        train_subset = Subset(train_dataset_for_train_transform, train_indices)
        val_subset = Subset(train_dataset_for_eval_transform, val_indices)

        return train_subset, val_subset

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

        self.train_dataset, self.val_dataset = self._build_split_from_two_copies(
            train_dataset_for_train_transform=base_trainval_for_train,
            train_dataset_for_eval_transform=base_trainval_for_eval,
            train_ratio=self.pet_train_ratio,
        )

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

    def _build_dtd(self) -> None:
        """
        构建 DTD 数据集
        Build DTD datasets

        说明 / Notes:
        - DTD 官方就有 train / val / test
          DTD already has official train / val / test
        - 支持 partition 1~10
          Supports partition 1~10
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        self.train_dataset = datasets.DTD(
            root=str(self.data_root),
            split="train",
            partition=self.dtd_partition,
            transform=train_transform,
            download=self.download,
        )

        self.val_dataset = datasets.DTD(
            root=str(self.data_root),
            split="val",
            partition=self.dtd_partition,
            transform=eval_transform,
            download=self.download,
        )

        self.test_dataset = datasets.DTD(
            root=str(self.data_root),
            split="test",
            partition=self.dtd_partition,
            transform=eval_transform,
            download=self.download,
        )

    def _build_fgvc_aircraft(self) -> None:
        """
        构建 FGVC Aircraft 数据集
        Build FGVC Aircraft datasets

        说明 / Notes:
        - FGVC Aircraft 官方有 train / val / test
          FGVC Aircraft already has official train / val / test
        - annotation_level 支持 variant / family / manufacturer
          annotation_level supports variant / family / manufacturer
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        self.train_dataset = datasets.FGVCAircraft(
            root=str(self.data_root),
            split="train",
            annotation_level=self.aircraft_annotation_level,
            transform=train_transform,
            download=self.download,
        )

        self.val_dataset = datasets.FGVCAircraft(
            root=str(self.data_root),
            split="val",
            annotation_level=self.aircraft_annotation_level,
            transform=eval_transform,
            download=self.download,
        )

        self.test_dataset = datasets.FGVCAircraft(
            root=str(self.data_root),
            split="test",
            annotation_level=self.aircraft_annotation_level,
            transform=eval_transform,
            download=self.download,
        )

    def _build_country211(self) -> None:
        """
        构建 Country211 数据集
        Build Country211 datasets

        说明 / Notes:
        - Country211 官方使用 train / valid / test
          Country211 uses official train / valid / test
        - 注意验证集名字是 valid，不是 val
          Note that the validation split name is 'valid', not 'val'
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        self.train_dataset = datasets.Country211(
            root=str(self.data_root),
            split="train",
            transform=train_transform,
            download=self.download,
        )

        self.val_dataset = datasets.Country211(
            root=str(self.data_root),
            split="valid",
            transform=eval_transform,
            download=self.download,
        )

        self.test_dataset = datasets.Country211(
            root=str(self.data_root),
            split="test",
            transform=eval_transform,
            download=self.download,
        )

    def _build_food101(self) -> None:
        """
        构建 Food101 数据集
        Build Food101 datasets

        说明 / Notes:
        - Food101 官方只有 train / test
          Food101 officially only provides train / test
        - 因此这里把 official train 稳定切成 train / val
          Therefore we split official train into train / val deterministically
        - official test 保持不动
          The official test split is kept unchanged
        """
        train_transform = self.build_train_transform()
        eval_transform = self.build_eval_transform()

        base_train_for_train = datasets.Food101(
            root=str(self.data_root),
            split="train",
            transform=train_transform,
            download=self.download,
        )

        base_train_for_eval = datasets.Food101(
            root=str(self.data_root),
            split="train",
            transform=eval_transform,
            download=self.download,
        )

        self.train_dataset, self.val_dataset = self._build_split_from_two_copies(
            train_dataset_for_train_transform=base_train_for_train,
            train_dataset_for_eval_transform=base_train_for_eval,
            train_ratio=self.food101_train_ratio,
        )

        self.test_dataset = datasets.Food101(
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
        elif self.dataset_name == "dtd":
            self._build_dtd()
        elif self.dataset_name == "fgvc_aircraft":
            self._build_fgvc_aircraft()
        elif self.dataset_name == "country211":
            self._build_country211()
        elif self.dataset_name == "food101":
            self._build_food101()
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

    def get_dataset_summary(self) -> dict[str, int | str]:
        """
        返回数据集摘要信息 / Return dataset summary information
        """
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise RuntimeError("Datasets are not built yet. Please call setup() first.")

        return {
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
        }


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
        food101_train_ratio=0.9,
        dtd_partition=1,
        aircraft_annotation_level="variant",
    )
    dm.setup()

    summary = dm.get_dataset_summary()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    print(f"==== VisionDataModule Test ({dataset_name}) ====")
    print("Device                :", device)
    print("Dataset name          :", summary["dataset_name"])
    print("Num classes           :", summary["num_classes"])
    print("Train dataset size    :", summary["train_size"])
    print("Val dataset size      :", summary["val_size"])
    print("Test dataset size     :", summary["test_size"])
    print("Train images shape    :", train_images.shape)
    print("Train labels shape    :", train_labels.shape)
    print("Val images shape      :", val_images.shape)
    print("Val labels shape      :", val_labels.shape)
    print("Test images shape     :", test_images.shape)
    print("Test labels shape     :", test_labels.shape)
    print("Train min / max       :", train_images.min().item(), train_images.max().item())


if __name__ == "__main__":
    _demo_datamodule("oxfordiiitpet")