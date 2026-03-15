# src/engine/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datamodule import VisionDataModule
from src.models.dual_encoder_model import DualEncoderModel


class Trainer:
    """
    Simple trainer for image classification / 简单图像分类训练器

    当前版本功能 / Current features:
    1. 支持 train / val 循环
       Support train / val loops
    2. 支持 CrossEntropyLoss
       Support CrossEntropyLoss
    3. 支持 CUDA + AMP 混合精度
       Support CUDA + AMP mixed precision
    4. 支持保存 best checkpoint
       Support saving best checkpoint
    5. 支持 max_train_batches / max_val_batches 做快速冒烟测试
       Support max_train_batches / max_val_batches for quick smoke tests
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        scheduler: Any | None = None,
        use_amp: bool = True,
        save_dir: str = "./outputs/checkpoints",
    ) -> None:
        """
        Args:
            model:
                要训练的模型
                Model to train

            device:
                训练设备
                Training device

            optimizer:
                优化器
                Optimizer

            criterion:
                损失函数，默认 CrossEntropyLoss
                Loss function, default is CrossEntropyLoss

            scheduler:
                学习率调度器，可选
                Optional learning rate scheduler

            use_amp:
                是否在 CUDA 上启用混合精度
                Whether to enable mixed precision on CUDA

            save_dir:
                checkpoint 保存目录
                Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.use_amp = use_amp and device.type == "cuda"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # AMP 设备类型
        # AMP device type
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"

        # 新版 AMP 写法
        # New AMP API
        self.scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=self.use_amp,
        )

        # 记录最佳验证精度
        # Track best validation accuracy
        self.best_val_acc = -1.0

    def move_batch_to_device(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将 batch 移动到目标设备 / Move a batch to target device
        """
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算 top-1 accuracy / Compute top-1 accuracy
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        return acc

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        epoch_index: int,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """
        训练一个 epoch / Train one epoch

        Args:
            dataloader:
                训练集 DataLoader
                Train DataLoader

            epoch_index:
                当前 epoch 编号
                Current epoch index

            max_batches:
                最多训练多少个 batch，用于快速测试
                Maximum number of batches for quick testing

        Returns:
            {
                "loss": 平均训练损失 / average training loss,
                "acc": 平均训练准确率 / average training accuracy
            }
        """
        self.model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"[Train] Epoch {epoch_index}",
            leave=False,
        )

        for batch_index, batch in enumerate(progress_bar):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, labels = batch
            images, labels = self.move_batch_to_device(images, labels)

            self.optimizer.zero_grad(set_to_none=True)

            # CUDA 下可选混合精度
            # Optional mixed precision on CUDA
            with torch.amp.autocast(
                self.amp_device_type,
                enabled=self.use_amp,
            ):
                outputs = self.model(images)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            batch_size = labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()

            running_loss += loss.item() * batch_size
            running_correct += correct
            running_total += batch_size

            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total

            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}",
                }
            )

        if running_total == 0:
            return {"loss": 0.0, "acc": 0.0}

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "Val",
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """
        在验证集或测试集上评估 / Evaluate on validation or test set

        Args:
            dataloader:
                验证集或测试集 DataLoader
                Validation or test DataLoader

            split_name:
                当前评估 split 名称
                Name of current evaluation split

            max_batches:
                最多评估多少个 batch，用于快速测试
                Maximum number of batches for quick testing

        Returns:
            {
                "loss": 平均损失 / average loss,
                "acc": 平均准确率 / average accuracy
            }
        """
        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"[{split_name}]",
            leave=False,
        )

        for batch_index, batch in enumerate(progress_bar):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, labels = batch
            images, labels = self.move_batch_to_device(images, labels)

            with torch.amp.autocast(
                self.amp_device_type,
                enabled=self.use_amp,
            ):
                outputs = self.model(images)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()

            running_loss += loss.item() * batch_size
            running_correct += correct
            running_total += batch_size

            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total

            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}",
                }
            )

        if running_total == 0:
            return {"loss": 0.0, "acc": 0.0}

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
        }

    def save_checkpoint(
        self,
        epoch_index: int,
        val_metrics: dict[str, float],
        filename: str = "best_model.pt",
    ) -> None:
        """
        保存 checkpoint / Save checkpoint
        """
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            "epoch": epoch_index,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "val_metrics": val_metrics,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, list[float]]:
        """
        训练整个过程 / Run the full training process

        Args:
            train_dataloader:
                训练集 DataLoader
                Train DataLoader

            val_dataloader:
                验证集 DataLoader
                Validation DataLoader

            num_epochs:
                总训练轮数
                Total number of epochs

            max_train_batches:
                每个 epoch 最多训练多少个 batch
                Maximum training batches per epoch

            max_val_batches:
                每个 epoch 最多验证多少个 batch
                Maximum validation batches per epoch

        Returns:
            history:
                记录训练历史
                Training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        print("=" * 80)
        print("Start training / 开始训练")
        print(f"Device             : {self.device}")
        print(f"Use AMP            : {self.use_amp}")
        print(f"Num epochs         : {num_epochs}")
        print(f"Max train batches  : {max_train_batches}")
        print(f"Max val batches    : {max_val_batches}")
        print(f"Save dir           : {self.save_dir}")
        print("=" * 80)

        for epoch_index in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(
                dataloader=train_dataloader,
                epoch_index=epoch_index,
                max_batches=max_train_batches,
            )

            val_metrics = self.evaluate(
                dataloader=val_dataloader,
                split_name="Val",
                max_batches=max_val_batches,
            )

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_metrics["loss"])
                except TypeError:
                    self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])

            print(
                f"Epoch [{epoch_index}/{num_epochs}] | "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['acc']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}"
            )

            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.save_checkpoint(
                    epoch_index=epoch_index,
                    val_metrics=val_metrics,
                    filename="best_model.pt",
                )
                print(
                    f"New best model saved / 保存新的最佳模型: "
                    f"val_acc={self.best_val_acc:.4f}"
                )

        return history


def _demo_train_smoke_test() -> None:
    """
    一个非常小的训练冒烟测试 / A very small training smoke test

    说明 / Notes:
    - 这里只是检查 trainer + datamodule + model 能不能连起来
      This only checks whether trainer + datamodule + model can work together
    - 不是为了得到有意义的性能
      It is NOT intended to produce meaningful performance
    - 为了避免本机 8GB 显存压力太大，这里只跑很少几个 batch
      To avoid stressing an 8GB local GPU, only a few batches are run
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. Build datamodule / 构建数据模块
    # ---------------------------
    datamodule = VisionDataModule(
        dataset_name="oxfordiiitpet",
        data_root="./data",
        image_size=224,
        batch_size=2,
        num_workers=0,
        download=False,
        pin_memory=torch.cuda.is_available(),
        use_imagenet_norm=True,
        split_seed=42,
        pet_train_ratio=0.9,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # ---------------------------
    # 2. Build model / 构建模型
    # ---------------------------
    model = DualEncoderModel(
        num_classes=datamodule.num_classes,
        resnet_name="resnet50",
        vit_name="vit_b_16",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="concat",
        fusion_dim=512,
        projector_hidden_dim=1024,
        fusion_hidden_dim=512,
        dropout=0.1,
    )

    # ---------------------------
    # 3. Build optimizer / 构建优化器
    # ---------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
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
        use_amp=torch.cuda.is_available(),
        save_dir="./outputs/checkpoints/debug_trainer",
    )

    # ---------------------------
    # 5. Run a tiny smoke test / 跑一个很小的冒烟测试
    # ---------------------------
    history = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=1,
        max_train_batches=2,
        max_val_batches=1,
    )

    print("=" * 80)
    print("Smoke test finished / 冒烟测试完成")
    print("History:", history)
    print("=" * 80)


if __name__ == "__main__":
    _demo_train_smoke_test()