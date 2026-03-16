from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from src.data.datamodule import VisionDataModule
from src.engine.trainer import Trainer
from src.models.dual_encoder_model import DualEncoderModel
from src.utils.logger import ExperimentLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BridgeVision baseline model")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="oxfordiiitpet",
        choices=["oxfordiiitpet", "flowers102", "dtd", "fgvc_aircraft", "country211", "food101"],
    )
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

    parser.add_argument("--model_mode", type=str, default="dual", choices=["dual", "resnet_only", "vit_only"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--resnet_name", type=str, default="resnet50")
    parser.add_argument("--vit_name", type=str, default="vit_b_16")
    parser.add_argument("--pretrained_backbones", action="store_true")
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat",
        choices=["concat", "gated", "token_bridge", "matched_token_gated"],
    )
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
    parser.add_argument("--matched_token_count", type=int, default=16)
    parser.add_argument("--summary_fusion_type", type=str, default="gated", choices=["concat", "gated"])
    parser.add_argument("--disable_cnn_pos_embed", action="store_false", dest="use_cnn_pos_embed")
    parser.set_defaults(use_cnn_pos_embed=True)
    parser.add_argument("--cnn_pos_embed_base_size", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

    parser.add_argument(
        "--debug_interval",
        type=int,
        default=0,
        help="Log debug scalars every N train steps. 0 disables it.",
    )
    parser.add_argument(
        "--log_debug_gates",
        action="store_true",
        help="Whether to log gate/debug scalars.",
    )

    parser.add_argument("--torch_home", type=str, default="./pretrained/torch_home")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/run_train")

    return parser


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def build_model_config(args: argparse.Namespace, num_classes: int) -> dict[str, object]:
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
        "matched_token_count": args.matched_token_count,
        "summary_fusion_type": args.summary_fusion_type,
        "use_cnn_pos_embed": args.use_cnn_pos_embed,
        "cnn_pos_embed_base_size": args.cnn_pos_embed_base_size,
    }


def build_data_config(args: argparse.Namespace, num_classes: int) -> dict[str, object]:
    return {
        "dataset_name": args.dataset_name,
        "data_root": args.data_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "download": args.download,
        "split_seed": args.split_seed,
        "pet_train_ratio": args.pet_train_ratio,
        "food101_train_ratio": args.food101_train_ratio,
        "dtd_partition": args.dtd_partition,
        "aircraft_annotation_level": args.aircraft_annotation_level,
        "num_classes": num_classes,
    }


def main() -> None:
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
    data_config = build_data_config(args, num_classes)

    experiment_logger = ExperimentLogger(
        save_dir=args.save_dir,
        run_name="train",
        enable_console=True,
    )

    experiment_logger.log_config(
        {
            "args": vars(args),
            "model_config": model_config,
            "data_config": data_config,
        }
    )

    model = DualEncoderModel(**model_config)

    total_params, trainable_params = count_parameters(model)
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_param_list) == 0:
        raise ValueError("No trainable parameters found. Please check model / freeze settings.")

    experiment_logger.info("=" * 80)
    experiment_logger.info("Model summary")
    experiment_logger.info(f"Model mode          : {args.model_mode}")
    experiment_logger.info(f"Dataset             : {args.dataset_name}")
    experiment_logger.info(f"Num classes         : {num_classes}")
    experiment_logger.info(f"Pretrained          : {args.pretrained_backbones}")
    experiment_logger.info(f"Freeze backbones    : {args.freeze_backbones}")
    experiment_logger.info(f"Fusion type         : {args.fusion_type}")
    experiment_logger.info(f"Projector type      : {args.projector_type}")
    experiment_logger.info(f"Fusion dim          : {args.fusion_dim}")
    if args.fusion_type in {"token_bridge", "matched_token_gated"}:
        experiment_logger.info(f"Token heads         : {args.token_num_heads}")
        experiment_logger.info(f"Bridge layers       : {args.num_bridge_layers}")
    if args.fusion_type == "matched_token_gated":
        experiment_logger.info(f"Matched token count : {args.matched_token_count}")
    if args.fusion_type == "token_bridge":
        experiment_logger.info(f"Summary fusion      : {args.summary_fusion_type}")
    experiment_logger.info(f"Use CNN pos embed   : {args.use_cnn_pos_embed}")
    experiment_logger.info(f"Total params        : {total_params:,}")
    experiment_logger.info(f"Trainable params    : {trainable_params:,}")
    experiment_logger.info(f"TORCH_HOME          : {torch_home}")
    experiment_logger.info("=" * 80)

    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_meta = {
        "model_config": model_config,
        "data_config": data_config,
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
            "debug_interval": args.debug_interval,
            "log_debug_gates": args.log_debug_gates,
        },
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
        experiment_logger=experiment_logger,
        debug_interval=args.debug_interval,
        log_debug_gates=args.log_debug_gates,
    )

    try:
        try:
            history = trainer.fit(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=args.epochs,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
            )
        except KeyboardInterrupt:
            experiment_logger.info("=" * 80)
            experiment_logger.info("Training interrupted by user")
            experiment_logger.info("=" * 80)
            return

        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1

        summary = {
            "dataset_name": args.dataset_name,
            "model_mode": args.model_mode,
            "resnet_name": args.resnet_name,
            "vit_name": args.vit_name,
            "pretrained_backbones": args.pretrained_backbones,
            "freeze_backbones": args.freeze_backbones,
            "fusion_type": args.fusion_type,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_val_acc": history["val_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

        experiment_logger.save_history(history)
        experiment_logger.save_summary(summary)

        experiment_logger.info("=" * 80)
        experiment_logger.info("Training finished")
        experiment_logger.info(f"History: {history}")
        experiment_logger.info("=" * 80)

    finally:
        experiment_logger.close()


if __name__ == "__main__":
    main()
