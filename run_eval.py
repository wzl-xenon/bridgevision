from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

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
    # 这些参数在 checkpoint 没存对应信息时作为 fallback
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
    # 这些参数在 checkpoint 没存对应信息时作为 fallback
    # ---------------------------
    parser.add_argument("--model_mode", type=str, default="dual", choices=["dual", "resnet_only", "vit_only"])
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
    # Cache / 缓存相关
    # ---------------------------
    parser.add_argument("--torch_home", type=str, default="./pretrained/torch_home")

    # ---------------------------
    # Checkpoint / 权重相关
    # ---------------------------
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./outputs/checkpoints/run_train/best_model.pt",
    )

    return parser


def load_checkpoint_file(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def resolve_data_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    ckpt_data_config = checkpoint.get("data_config", {})

    return {
        "dataset_name": ckpt_data_config.get("dataset_name", args.dataset_name),
        "image_size": ckpt_data_config.get("image_size", args.image_size),
        "split_seed": ckpt_data_config.get("split_seed", args.split_seed),
        "pet_train_ratio": ckpt_data_config.get("pet_train_ratio", args.pet_train_ratio),
    }


def build_model_config_from_args(args: argparse.Namespace, num_classes: int) -> dict[str, Any]:
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
    }


def resolve_model_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    num_classes: int,
) -> tuple[dict[str, Any], str]:
    ckpt_model_config = checkpoint.get("model_config")

    if ckpt_model_config is not None:
        model_config = dict(ckpt_model_config)

        ckpt_num_classes = model_config.get("num_classes")
        if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
            raise ValueError(
                f"Checkpoint num_classes={ckpt_num_classes} does not match dataset num_classes={num_classes}."
            )

        # eval 只需要结构匹配，不需要再触发预训练权重下载
        model_config["pretrained_backbones"] = False
        model_config["freeze_backbones"] = False
        model_config["num_classes"] = num_classes

        return model_config, "checkpoint"
    else:
        model_config = build_model_config_from_args(args, num_classes)
        return model_config, "cli"


def main() -> None:
    """
    主评估入口 / Main evaluation entry
    """
    parser = build_parser()
    args = parser.parse_args()

    torch_home = Path(args.torch_home).resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. Load checkpoint file first
    # ---------------------------
    checkpoint = load_checkpoint_file(
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    # ---------------------------
    # 2. Resolve data config
    # 优先使用 checkpoint 中保存的数据配置
    # ---------------------------
    resolved_data_config = resolve_data_config(args, checkpoint)

    datamodule = VisionDataModule(
        dataset_name=resolved_data_config["dataset_name"],
        data_root=args.data_root,
        image_size=resolved_data_config["image_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        pin_memory=torch.cuda.is_available(),
        use_imagenet_norm=True,
        split_seed=resolved_data_config["split_seed"],
        pet_train_ratio=resolved_data_config["pet_train_ratio"],
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
    # 3. Resolve model config
    # ---------------------------
    model_config, config_source = resolve_model_config(
        args=args,
        checkpoint=checkpoint,
        num_classes=num_classes,
    )

    model = DualEncoderModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 80)
    print("Checkpoint loaded / 已加载 checkpoint")
    print(f"Checkpoint path      : {args.checkpoint_path}")
    print(f"Config source        : {config_source}")
    print(f"Dataset              : {resolved_data_config['dataset_name']}")
    print(f"Image size           : {resolved_data_config['image_size']}")
    print(f"Split seed           : {resolved_data_config['split_seed']}")
    print(f"Pet train ratio      : {resolved_data_config['pet_train_ratio']}")
    print(f"Model mode           : {model_config['model_mode']}")
    print(f"Saved epoch          : {checkpoint.get('epoch', 'N/A')}")
    print(f"Saved best_val_acc   : {checkpoint.get('best_val_acc', 'N/A')}")
    print("=" * 80)

    # ---------------------------
    # 4. Build trainer / 复用 evaluate
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