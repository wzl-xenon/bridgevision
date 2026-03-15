from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Simple trainer for image classification / 简单图像分类训练器
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
        checkpoint_meta: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.use_amp = use_amp and device.type == "cuda"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_meta = checkpoint_meta or {}

        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"

        self.scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=self.use_amp,
        )

        self.best_val_acc = -1.0

    def move_batch_to_device(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct / total

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        epoch_index: int,
        max_batches: int | None = None,
    ) -> dict[str, float]:
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

        checkpoint.update(self.checkpoint_meta)

        torch.save(checkpoint, checkpoint_path)

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, list[float]]:
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