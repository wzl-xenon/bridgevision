# run_train.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.datamodule import VisionDataModule
from src.engine.trainer import Trainer
from src.models.dual_encoder_model import DualEncoderModel


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数 / Build command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train BridgeVision baseline model")

    # ---------------------------
    # Data / 数据相关
    # ---------------------------
    parser.add_argument("--dataset_name", type=str, default="oxfordiiitpet", choices=["oxfordiiitpet", "flowers102"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pet_train_ratio", type=float, default=0.9)

    # ---------------------------
    # Model / 模型相关
    # ---------------------------
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--resnet_name", type=str, default="resnet50")
    parser.add_argument("--vit_name", type=str, default="vit_b_16")
    parser.add_argument("--pretrained_backbones", action="store_true")
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "gated"])
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--projector_hidden_dim", type=int, default=1024)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---------------------------
    # Train / 训练相关
    # ---------------------------
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    # ---------------------------
    # Save / 保存相关
    # ---------------------------
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/run_train")

    return parser


def main() -> None:
    """
    主训练入口 / Main training entry
    """
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. Build datamodule / 构建数据模块
    # ---------------------------
    datamodule = VisionDataModule(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        pin_memory=torch.cuda.is_available(),
        use_imagenet_norm=True,
        split_seed=args.split_seed,
        pet_train_ratio=args.pet_train_ratio,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    num_classes = args.num_classes if args.num_classes is not None else datamodule.num_classes

    # ---------------------------
    # 2. Build model / 构建模型
    # ---------------------------
    model = DualEncoderModel(
        num_classes=num_classes,
        resnet_name=args.resnet_name,
        vit_name=args.vit_name,
        pretrained_backbones=args.pretrained_backbones,
        freeze_backbones=args.freeze_backbones,
        projector_type=args.projector_type,
        fusion_type=args.fusion_type,
        fusion_dim=args.fusion_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
    )

    # ---------------------------
    # 3. Build optimizer / 构建优化器
    # ---------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---------------------------
    # 4. Build trainer / 构建训练器
    # ---------------------------
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
    )

    # ---------------------------
    # 5. Train / 开始训练
    # ---------------------------
    history = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    print("=" * 80)
    print("Training finished / 训练完成")
    print("History:", history)
    print("=" * 80)


if __name__ == "__main__":
    main()