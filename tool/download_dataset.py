#!/usr/bin/env python3
"""下载单个支持的数据集，或批量下载全部数据集 / Download one supported
dataset or all supported datasets without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import VisionDataModule


SUPPORTED_DATASETS = [
    "oxfordiiitpet",
    "flowers102",
    "dtd",
    "fgvc_aircraft",
    "country211",
    "food101",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download datasets for BridgeVision")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[*SUPPORTED_DATASETS, "all"],
        help="Dataset name to download, or 'all' to download every supported dataset.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory where datasets are stored.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size used to initialize the datamodule.",
    )

    # 这些参数用于保持下载脚本与 datamodule API 兼容 / These arguments keep
    # the downloader compatible with the datamodule API.
    parser.add_argument("--food101_train_ratio", type=float, default=0.9)
    parser.add_argument("--dtd_partition", type=int, default=1)
    parser.add_argument(
        "--aircraft_annotation_level",
        type=str,
        default="variant",
        choices=["variant", "family", "manufacturer"],
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pet_train_ratio", type=float, default=0.9)
    return parser


def download_single_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Downloading dataset: {dataset_name}")
    print(f"Save path          : {Path(args.data_root).resolve() / dataset_name}")
    print("=" * 60)

    datamodule = VisionDataModule(
        dataset_name=dataset_name,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=1,
        download=True,
        split_seed=args.split_seed,
        pet_train_ratio=args.pet_train_ratio,
        food101_train_ratio=args.food101_train_ratio,
        dtd_partition=args.dtd_partition,
        aircraft_annotation_level=args.aircraft_annotation_level,
    )

    datamodule.setup()

    print(f"Dataset downloaded successfully: {dataset_name}")
    print(f"Train set size: {len(datamodule.train_dataset)}")
    print(f"Val set size  : {len(datamodule.val_dataset)}")
    print(f"Test set size : {len(datamodule.test_dataset)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset_name == "all":
        print(f"Preparing to download {len(SUPPORTED_DATASETS)} datasets.")
        for dataset_name in SUPPORTED_DATASETS:
            try:
                download_single_dataset(args, dataset_name)
            except Exception as exc:
                print(f"Failed to download {dataset_name}: {exc}")
        print("\nAll datasets processed.")
    else:
        download_single_dataset(args, args.dataset_name)
        print("\nDownload completed.")


if __name__ == "__main__":
    main()
