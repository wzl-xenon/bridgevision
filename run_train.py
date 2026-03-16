from __future__ import annotations

import argparse
import os
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
    parser.add_argument("--dataset_name", type=str, default="oxfordiiitpet", choices=["oxfordiiitpet", "flowers102", "dtd", "fgvc_aircraft", "country211", "food101"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pet_train_ratio", type=float, default=0.9)
    parser.add_argument("--food101_train_ratio", type=float, default=0.9)
    parser.add_argument("--dtd_partition", type=int, default=1)
    parser.add_argument(
        "--aircraft_annotation_level",
        type=str,
        default="variant",
        choices=["variant", "family", "manufacturer"],
    )

    # ---------------------------
    # Model / 模型相关
    # ---------------------------
    parser.add_argument("--model_mode", type=str, default="dual", choices=["dual", "resnet_only", "vit_only"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--resnet_name", type=str, default="resnet50")
    parser.add_argument("--vit_name", type=str, default="vit_b_16")
    parser.add_argument("--pretrained_backbones", action="store_true")
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "gated", "token_bridge"])
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--projector_hidden_dim", type=int, default=1024)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_num_heads", type=int, default=8)
    parser.add_argument("--token_gate_hidden_dim", type=int, default=None)
    parser.add_argument("--token_ffn_hidden_dim", type=int, default=None)
    parser.add_argument("--disable_token_gate", action="store_false", dest="token_use_gate")
    parser.set_defaults(token_use_gate=True)
    parser.add_argument("--num_bridge_layers", type=int, default=1)
    parser.add_argument("--summary_fusion_type", type=str, default="gated", choices=["concat", "gated"])
    parser.add_argument("--disable_cnn_pos_embed", action="store_false", dest="use_cnn_pos_embed")
    parser.set_defaults(use_cnn_pos_embed=True)
    parser.add_argument("--cnn_pos_embed_base_size", type=int, default=7)

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
    # Cache / 缓存相关
    # ---------------------------
    parser.add_argument("--torch_home", type=str, default="./pretrained/torch_home")

    # ---------------------------
    # Save / 保存相关
    # ---------------------------
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/run_train")

    return parser


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    统计总参数量和可训练参数量
    Count total and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def build_model_config(args: argparse.Namespace, num_classes: int) -> dict[str, object]:
    """
    构建模型配置字典 / Build model configuration dictionary
    """
    return {
        "model_mode": args.model_mode,
        "num_classes": num_classes,
        "resnet_name": args.resnet_name,
        "vit_name": args.vit_name,
        "pretrained_backbones": args.pretrained_backbones,
        "freeze_backbones": args.freeze_backbones,
        "projector_type": args.projector_type,
        "fusion_type": args.fusion_type,
        "fusion_dim": args.fusion_dim,
        "projector_hidden_dim": args.projector_hidden_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "dropout": args.dropout,
        "token_num_heads": args.token_num_heads,
        "token_gate_hidden_dim": args.token_gate_hidden_dim,
        "token_ffn_hidden_dim": args.token_ffn_hidden_dim,
        "token_use_gate": args.token_use_gate,
        "num_bridge_layers": args.num_bridge_layers,
        "summary_fusion_type": args.summary_fusion_type,
        "use_cnn_pos_embed": args.use_cnn_pos_embed,
        "cnn_pos_embed_base_size": args.cnn_pos_embed_base_size,
    }


def main() -> None:
    """
    主训练入口 / Main training entry
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.freeze_backbones and not args.pretrained_backbones:
        raise ValueError(
            "freeze_backbones=True requires pretrained_backbones=True. "
            "Freezing randomly initialized backbones is not meaningful for stage-1 experiments."
        )

    torch_home = Path(args.torch_home).resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

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
        food101_train_ratio=args.food101_train_ratio,
        dtd_partition=args.dtd_partition,
        aircraft_annotation_level=args.aircraft_annotation_level,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    num_classes = args.num_classes if args.num_classes is not None else datamodule.num_classes

    model_config = build_model_config(args, num_classes)
    data_config = {
        "dataset_name": args.dataset_name,
        "image_size": args.image_size,
        "split_seed": args.split_seed,
        "pet_train_ratio": args.pet_train_ratio,
        "food101_train_ratio": args.food101_train_ratio,
        "dtd_partition": args.dtd_partition,
        "aircraft_annotation_level": args.aircraft_annotation_level,
    }

    # ---------------------------
    # 2. Build model / 构建模型
    # ---------------------------
    model = DualEncoderModel(**model_config)

    total_params, trainable_params = count_parameters(model)
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_param_list) == 0:
        raise ValueError("No trainable parameters found. Please check model / freeze settings.")

    print("=" * 80)
    print("Model summary / 模型摘要")
    print(f"Model mode          : {args.model_mode}")
    print(f"Dataset             : {args.dataset_name}")
    print(f"Num classes         : {num_classes}")
    print(f"Pretrained          : {args.pretrained_backbones}")
    print(f"Freeze backbones    : {args.freeze_backbones}")
    print(f"Fusion type         : {args.fusion_type}")
    print(f"Projector type      : {args.projector_type}")
    print(f"Fusion dim          : {args.fusion_dim}")
    print(f"Token heads         : {args.token_num_heads}")
    print(f"Bridge layers       : {args.num_bridge_layers}")
    print(f"Summary fusion      : {args.summary_fusion_type}")
    print(f"Use CNN pos embed   : {args.use_cnn_pos_embed}")
    print(f"Total params        : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")
    print(f"TORCH_HOME          : {torch_home}")
    print("=" * 80)

    # ---------------------------
    # 3. Build optimizer / 构建优化器
    # 只传可训练参数
    # ---------------------------
    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_meta = {
        "model_config": model_config,
        "data_config": {
            "dataset_name": args.dataset_name,
            "image_size": args.image_size,
            "split_seed": args.split_seed,
            "pet_train_ratio": args.pet_train_ratio,
            "num_classes": num_classes,
        },
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "use_amp": args.use_amp,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "save_dir": args.save_dir,
            "torch_home": str(torch_home),
        },
    }

    # ---------------------------
    # 4. Build trainer / 构建训练器
    # ---------------------------
    checkpoint_meta = {
        "model_config": model_config,
        "data_config": data_config,
        "args": vars(args),
    }

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
        checkpoint_meta=checkpoint_meta,
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