# run_eval.py

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
    parser = argparse.ArgumentParser(description="Evaluate BridgeVision model")

    # ---------------------------
    # Data / 数据相关
    # ---------------------------
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="oxfordiiitpet",
        choices=["oxfordiiitpet", "flowers102"],
    )
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
    parser.add_argument(
        "--projector_type",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat",
        choices=["concat", "gated"],
    )
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--projector_hidden_dim", type=int, default=1024)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---------------------------
    # Eval / 评估相关
    # ---------------------------
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Choose which split to evaluate on.",
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--max_eval_batches", type=int, default=None)

    # ---------------------------
    # Checkpoint / 权重相关
    # ---------------------------
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./outputs/checkpoints/run_train/best_model.pt",
    )

    return parser


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """
    加载 checkpoint / Load checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def main() -> None:
    """
    主评估入口 / Main evaluation entry
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

    if args.split == "val":
        eval_loader = datamodule.val_dataloader()
        split_name = "Val"
    elif args.split == "test":
        eval_loader = datamodule.test_dataloader()
        split_name = "Test"
    else:
        raise ValueError(f"Unsupported split={args.split}")

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
    # 3. Load checkpoint / 加载 checkpoint
    # ---------------------------
    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    print("=" * 80)
    print("Checkpoint loaded / 已加载 checkpoint")
    print(f"Checkpoint path     : {args.checkpoint_path}")
    print(f"Saved epoch         : {checkpoint.get('epoch', 'N/A')}")
    print(f"Saved best_val_acc  : {checkpoint.get('best_val_acc', 'N/A')}")
    print("=" * 80)

    # ---------------------------
    # 4. Build trainer / 构建 trainer（复用 evaluate）
    # ---------------------------
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=dummy_optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        use_amp=args.use_amp,
        save_dir="./outputs/checkpoints/eval_tmp",
    )

    # ---------------------------
    # 5. Evaluate / 评估
    # ---------------------------
    metrics = trainer.evaluate(
        dataloader=eval_loader,
        split_name=split_name,
        max_batches=args.max_eval_batches,
    )

    print("=" * 80)
    print(f"{split_name} evaluation finished / {split_name} 评估完成")
    print(f"{split_name} loss : {metrics['loss']:.4f}")
    print(f"{split_name} acc  : {metrics['acc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()