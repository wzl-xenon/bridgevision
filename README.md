# BridgeVision

BridgeVision 是一个基于 PyTorch 的图像分类实验框架，核心目标是研究 CNN 与 ViT 的双分支融合（dual-encoder fusion）。

这个项目目前更偏向研究与实验（research / experimentation），而不是产品化部署。它提供了统一的数据加载、训练、评估和 checkpoint 管理流程，适合快速验证不同融合策略的效果。

当前重点支持三类融合思路：
- 全局特征融合（global-feature fusion）：`concat`、`gated`
- token 级双向交互（token-level bridge）：`token_bridge`
- 固定长度 token 对齐后再做逐维门控融合：`matched_token_gated`

## 1. 项目概览

当前仓库的主要入口如下：

- 训练入口（training entry）：[run_train.py](/D:/workspace/bridgevision/run_train.py)
- 评估入口（evaluation entry）：[run_eval.py](/D:/workspace/bridgevision/run_eval.py)
- 主模型（main model）：[src/models/dual_encoder_model.py](/D:/workspace/bridgevision/src/models/dual_encoder_model.py)
- 数据模块（data module）：[src/data/datamodule.py](/D:/workspace/bridgevision/src/data/datamodule.py)

需要注意：
- 当前项目主要通过命令行参数（CLI arguments）进行配置。
- `configs/` 目录目前保留着，但还不是主要配置入口。

## 2. 支持的模式

### 2.1 模型模式（Model Modes）

- `dual`
  同时启用 ResNet 和 ViT 两个分支。
- `resnet_only`
  只运行 CNN 基线。
- `vit_only`
  只运行 ViT 基线。

### 2.2 融合方式（Fusion Types）

- `concat`
  将两个全局特征拼接后送入 MLP 做融合。
- `gated`
  使用门控（gate）对两个全局特征做加权融合。
- `token_bridge`
  让 CNN tokens 与 ViT tokens 做双向 cross-attention，再从摘要 token 读出分类特征。
- `matched_token_gated`
  先进行 token bridge，再把两路 token 重采样到固定长度 `N_m`，最后在 `[B, N_m, D]` 上做逐 token、逐维门控融合。

## 3. 支持的数据集

当前 `VisionDataModule` 支持以下数据集：

- `oxfordiiitpet`
- `flowers102`
- `dtd`
- `fgvc_aircraft`
- `country211`
- `food101`

数据划分策略做了统一封装：
- 如果数据集官方提供 `train / val / test`，就直接使用官方划分。
- 如果官方只提供 `train / test` 或 `trainval / test`，项目会按固定随机种子稳定切分出 `train / val`。

## 4. 环境与依赖（Environment）

本仓库最近一次验证环境如下：

- Python `3.12.12`
- PyTorch `2.10.0`
- TorchVision `0.25.0`
- tqdm `4.67.3`
- Pillow `12.0.0`
- pytest `9.0.2`

安装依赖：

```powershell
pip install -r requirements.txt
```

如果你需要指定 CUDA 版本，建议先按 PyTorch 官方方式安装匹配的 `torch` / `torchvision`，再执行上面的命令补齐其余依赖。

## 5. 快速开始（Quick Start）

### 5.1 预下载预训练权重

```powershell
python tool/predownload_weights.py
```

### 5.2 下载数据集

下载单个数据集：

```powershell
python tool/download_dataset.py --dataset_name oxfordiiitpet --data_root ./data
```

下载全部支持的数据集：

```powershell
python tool/download_dataset.py --dataset_name all --data_root ./data
```

### 5.3 训练一个全局融合基线

```powershell
python run_train.py --dataset_name oxfordiiitpet --data_root ./data --download --model_mode dual --fusion_type concat --projector_type mlp --pretrained_backbones --epochs 10 --batch_size 8 --lr 1e-4 --save_dir ./outputs/checkpoints/concat_exp
```

### 5.4 训练 token bridge 模型

```powershell
python run_train.py --dataset_name oxfordiiitpet --data_root ./data --download --image_size 224 --batch_size 8 --num_workers 0 --model_mode dual --resnet_name resnet18 --vit_name vit_b_32 --pretrained_backbones --projector_type mlp --fusion_type token_bridge --fusion_dim 256 --projector_hidden_dim 512 --fusion_hidden_dim 256 --token_num_heads 8 --token_gate_hidden_dim 256 --token_ffn_hidden_dim 512 --num_bridge_layers 1 --summary_fusion_type gated --epochs 30 --lr 1e-4 --weight_decay 1e-4 --use_amp --save_dir ./outputs/checkpoints/token_bridge_ep30
```

### 5.5 训练固定长度 token 门控融合模型

这是当前项目里较新的 readout 方案：

```powershell
python run_train.py --dataset_name oxfordiiitpet --data_root ./data --download --image_size 224 --batch_size 8 --num_workers 0 --model_mode dual --resnet_name resnet18 --vit_name vit_b_32 --pretrained_backbones --projector_type mlp --fusion_type matched_token_gated --fusion_dim 256 --projector_hidden_dim 512 --fusion_hidden_dim 256 --token_num_heads 8 --token_gate_hidden_dim 256 --token_ffn_hidden_dim 512 --num_bridge_layers 1 --matched_token_count 12 --epochs 30 --lr 1e-4 --weight_decay 1e-4 --use_amp --save_dir ./outputs/checkpoints/matched_token_gated_ep30
```

### 5.6 评估模型

```powershell
python run_eval.py --checkpoint_path ./outputs/checkpoints/matched_token_gated_ep30/best_model.pt --split test
```

### 5.7 运行前向 shape 测试

```powershell
pytest tests/test_model_forward.py
```

## 6. 模型结构说明

### 6.1 全局融合路径

`concat` / `gated` 的数据流可以概括为：

`image -> ResNet pooled feature + ViT cls feature -> projector -> fusion -> classifier`

这条路径简单、稳定，适合作为 baseline。

### 6.2 token bridge 路径

`token_bridge` 的核心流程：

`image -> ResNet feature map + pooled feature`

`image -> ViT full token sequence`

`CNN feature map -> spatial tokens + CNN global token`

`CNN tokens / ViT tokens -> shared hidden space`

`-> one or more TokenBridgeFusion blocks`

`-> read summary tokens`

`-> summary fusion`

`-> classifier`

这里的 `TokenBridgeFusion` 会做双向 cross-attention：
- `CNN <- ViT`
- `ViT <- CNN`

然后配合 gate 和 FFN refinement 完成 token 级交互。

### 6.3 matched token gated 路径

`matched_token_gated` 在 token bridge 之后再多做两步：

1. 将两路 token 重采样到固定长度 `N_m`
2. 在 `[B, N_m, D]` 上做逐 token、逐维度的门控融合

高层流程如下：

`image -> dual backbones -> token bridge -> fixed token alignment -> token-dim gated fusion -> mean pooling -> classifier`

这条路径的目的，是让最终分类不仅依赖第一个 token，而是让全部对齐后的融合 token 共同参与决策。

## 7. 数据流（Data Flow）

训练时的总体流向：

`dataset -> datamodule -> dataloader -> model -> logits -> cross entropy -> backward -> optimizer step`

checkpoint 中会保存：

- 模型权重（model weights）
- 模型配置（model config）
- 数据配置（data config）
- 训练配置（train config）
- 最佳验证精度（best validation accuracy）

日志默认保存在：

```text
outputs/checkpoints/<run_name>/logs/
```

常见输出文件包括：

- `config.json`
- `metrics.csv`
- `history.json`
- `summary.json`
- `train.log`
- `debug.jsonl`（开启 debug logging 时）

## 8. 常用参数说明

### 数据相关

- `--dataset_name`
  选择数据集。
- `--data_root`
  数据集根目录。
- `--image_size`
  输入图像尺寸。
- `--batch_size`
  batch size。
- `--download`
  若支持则自动下载数据集。
- `--split_seed`
  控制稳定划分 train / val 的随机种子。

### 模型相关

- `--model_mode`
  `dual`、`resnet_only` 或 `vit_only`。
- `--resnet_name`
  `resnet18`、`resnet34`、`resnet50`、`resnet101`。
- `--vit_name`
  `vit_b_16`、`vit_b_32`、`vit_l_16`、`vit_l_32`。
- `--pretrained_backbones`
  是否加载 torchvision 预训练权重。
- `--freeze_backbones`
  是否冻结 backbone 参数。
- `--fusion_type`
  `concat`、`gated`、`token_bridge`、`matched_token_gated`。
- `--fusion_dim`
  共享隐藏维度（shared hidden dimension）。

### token 融合相关

- `--token_num_heads`
  多头注意力 head 数。
- `--token_gate_hidden_dim`
  token gate MLP 的隐藏维度。
- `--token_ffn_hidden_dim`
  token FFN 的隐藏维度。
- `--num_bridge_layers`
  bridge block 层数。
- `--matched_token_count`
  `matched_token_gated` 中固定 token 数。
- `--summary_fusion_type`
  `token_bridge` 模式下摘要特征的融合方式。
- `--disable_token_gate`
  禁用 bridge 内部 gate。
- `--disable_cnn_pos_embed`
  禁用 CNN 位置编码。

### 训练相关

- `--epochs`
  训练轮数。
- `--lr`
  学习率。
- `--weight_decay`
  权重衰减。
- `--use_amp`
  启用混合精度（AMP）。
- `--max_train_batches`
  仅跑部分 train batches，适合调试。
- `--max_val_batches`
  仅跑部分 val batches，适合调试。
- `--save_dir`
  checkpoint 与日志输出目录。

## 9. 目录结构

```text
bridgevision/
|-- README.md
|-- requirements.txt
|-- run_train.py
|-- run_eval.py
|-- src/
|   |-- data/
|   |   `-- datamodule.py
|   |-- engine/
|   |   `-- trainer.py
|   |-- losses/
|   |   `-- classification_loss.py
|   |-- models/
|   |   |-- backbones/
|   |   |-- embeddings/
|   |   |-- fusions/
|   |   |-- projectors/
|   |   |-- tokenizers/
|   |   |-- dual_encoder_model.py
|   |   `-- token_bridge_model.py
|   `-- utils/
|       `-- logger.py
|-- tests/
|   `-- test_model_forward.py
`-- tool/
    |-- download_dataset.py
    `-- predownload_weights.py
```

## 10. 说明与建议

- 这是一个实验框架（experiment framework），更适合做结构验证和对比实验。
- 当前最主要的模型入口是 `DualEncoderModel`。
- `matched_token_gated` 是当前较新的分支，适合研究“固定长度 token 对齐 + 逐维门控融合”的效果。
- 项目目前已有基础前向 shape 测试，但还不是完整 benchmark 套件。

如果你接下来准备继续做 token 级融合实验，建议优先从：
- `resnet18 + vit_b_32`
- `fusion_dim=256`
- `num_bridge_layers=1`
- `matched_token_count=12`

这组配置开始，先验证训练稳定性，再逐步加大模型复杂度。
