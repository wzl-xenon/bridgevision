from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """
    轻量级实验日志器，同时写入控制台和文件 / Lightweight experiment logger
    that writes to both console and file.
    """

    def __init__(
        self,
        save_dir: str | Path,
        run_name: str = "train",
        enable_console: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name
        self.enable_console = enable_console

        self.log_file_path = self.log_dir / f"{run_name}.log"
        self.config_file_path = self.log_dir / "config.json"
        self.metrics_csv_path = self.log_dir / "metrics.csv"
        self.debug_jsonl_path = self.log_dir / "debug.jsonl"
        self.history_json_path = self.log_dir / "history.json"
        self.summary_json_path = self.log_dir / "summary.json"

        logger_name = f"BridgeVisionLogger::{self.log_dir.resolve()}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            if self.enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter("%(message)s")
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)

        self._metrics_header_written = self.metrics_csv_path.exists()

    def info(self, message: str) -> None:
        self.logger.info(message)

    def log_config(self, config: dict[str, Any]) -> None:
        with open(self.config_file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.info("=" * 80)
        self.info("Config saved")
        self.info(f"Config path: {self.config_file_path}")
        self.info("=" * 80)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        write_header = not self._metrics_header_written

        with open(self.metrics_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                self._metrics_header_written = True

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    def log_debug(
        self,
        epoch: int,
        step: int,
        split: str,
        debug_stats: dict[str, float],
    ) -> None:
        record = {
            "epoch": epoch,
            "step": step,
            "split": split,
            **debug_stats,
        }

        with open(self.debug_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_history(self, history: dict[str, list[float]]) -> None:
        with open(self.history_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        self.info(f"History saved: {self.history_json_path}")

    def save_summary(self, summary: dict[str, Any]) -> None:
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.info(f"Summary saved: {self.summary_json_path}")

    def close(self) -> None:
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
