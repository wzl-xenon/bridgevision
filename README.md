# BridgeVision

## 项目简介
BridgeVision是一个基于PyTorch的开源多模态融合视觉分类框架，创新性地结合了卷积神经网络（CNN）和视觉Transformer（ViT）两种不同架构的优势，通过多层级的特征融合策略实现更优的图像分类性能。

框架采用高度模块化设计，支持全局特征融合（Concat/Gated）和Token级双向交叉注意力融合（Token Bridge）两种模式，同时也支持单编码器模式（仅ResNet/仅ViT）作为对照实验，既可以作为学术研究的基准框架，也可以作为工业界落地的基础方案。

## 核心特性
✨ **多模式支持**：支持三种运行模式 - 双编码器融合模式、仅ResNet模式、仅ViT模式
🔧 **多层级融合**：内置三种融合策略 - Concat拼接融合、Gated门控自适应融合、Token Bridge交叉注意力融合
📦 **多数据集支持**：开箱即用支持6个标准数据集 - Oxford-IIIT Pets、Flowers102、DTD、FGVC Aircraft、Country211、Food101
⚙️ **高度模块化**：主干网络、投影器、融合模块、分类头全部分离，可灵活配置替换
🚀 **工程化完备**：支持自动混合精度训练、最佳Checkpoint自动保存、进度条显示、指标记录
💾 **可复现设计**：训练时自动保存完整的模型配置和数据配置到Checkpoint，评估时无需手动配置参数
🎛️ **高可配置性**：所有核心参数支持命令行动态调整，无需修改代码
📂 **本地缓存支持**：支持自定义预训练权重缓存目录，避免重复下载

## 环境要求
- Python >= 3.8
- PyTorch >= 1.12.0
- TorchVision >= 0.13.0
- tqdm >= 4.64.0
- Pillow >= 9.0.0
- PyYAML >= 6.0 (可选，用于配置文件模式)

## 安装步骤
1. 克隆项目到本地
```bash
git clone <repository-url>
cd BridgeVision
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 预下载预训练权重（可选，首次训练时会自动下载）
```bash
python tool/predownload_weights.py
```

## 快速开始
### 1. 训练模型
#### 基础双编码器融合训练（Concat全局融合）
```bash
python run_train.py \
  --dataset_name oxfordiiitpet \
  --pretrained_backbones \
  --model_mode dual \
  --fusion_type concat \
  --projector_type mlp \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_dir ./outputs/checkpoints/concat_exp
```

#### 门控融合训练
```bash
python run_train.py \
  --dataset_name flowers102 \
  --pretrained_backbones \
  --fusion_type gated \
  --epochs 20 \
  --batch_size 4 \
  --use_amp
```

#### Token Bridge融合训练（Token级交叉注意力）
```bash
python run_train.py \
  --dataset_name food101 \
  --pretrained_backbones \
  --fusion_type token_bridge \
  --num_bridge_layers 2 \
  --token_num_heads 8 \
  --summary_fusion_type gated \
  --epochs 15 \
  --batch_size 4 \
  --use_amp
```

#### 单ResNet基准训练
```bash
python run_train.py \
  --dataset_name dtd \
  --pretrained_backbones \
  --model_mode resnet_only \
  --epochs 10 \
  --batch_size 16
```

#### 单ViT基准训练
```bash
python run_train.py \
  --dataset_name fgvc_aircraft \
  --pretrained_backbones \
  --model_mode vit_only \
  --epochs 15 \
  --batch_size 8
```

### 2. 评估模型
> 评估时会自动从Checkpoint中加载模型和数据配置，无需手动指定大部分参数
```bash
python run_eval.py \
  --checkpoint_path ./outputs/checkpoints/concat_exp/best_model.pt \
  --split test
```

快速验证（仅跑前10个batch）：
```bash
python run_eval.py \
  --checkpoint_path ./outputs/checkpoints/token_bridge_exp/best_model.pt \
  --split val \
  --max_eval_batches 10
```

## 主要参数说明
### 数据相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --dataset_name | str | oxfordiiitpet | 数据集名称，支持 oxfordiiitpet / flowers102 / dtd / fgvc_aircraft / country211 / food101 |
| --data_root | str | ./data | 数据集存放根目录 |
| --image_size | int | 224 | 模型输入图像尺寸 |
| --batch_size | int | 4 | Batch大小 |
| --num_workers | int | 0 | DataLoader worker数量 |
| --download | action | False | 是否自动下载数据集 |
| --split_seed | int | 42 | 数据集划分随机种子 |
| --pet_train_ratio | float | 0.9 | Oxford-IIIT Pets训练集划分比例 |
| --food101_train_ratio | float | 0.9 | Food101训练集划分比例 |
| --dtd_partition | int | 1 | DTD数据集分区编号（1~10） |
| --aircraft_annotation_level | str | variant | FGVC Aircraft标签粒度，支持 variant / family / manufacturer |

### 模型相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --model_mode | str | dual | 模型运行模式，支持 dual / resnet_only / vit_only |
| --num_classes | int | None | 类别数，默认自动从数据集推断 |
| --resnet_name | str | resnet50 | ResNet主干版本，支持 resnet18/34/50/101 |
| --vit_name | str | vit_b_16 | ViT主干版本，支持 vit_b_16/vit_b_32/vit_l_16/vit_l_32 |
| --pretrained_backbones | action | False | 是否加载预训练主干权重 |
| --freeze_backbones | action | False | 是否冻结主干网络参数 |
| --projector_type | str | mlp | 投影器类型，支持 linear / mlp |
| --fusion_type | str | concat | 融合模式，支持 concat / gated / token_bridge |
| --fusion_dim | int | 512 | 特征融合维度 |
| --dropout | float | 0.1 | Dropout比例 |
| --token_num_heads | int | 8 | Token Bridge融合的注意力头数 |
| --num_bridge_layers | int | 1 | Token Bridge融合的层数 |
| --summary_fusion_type | str | gated | Token Bridge输出的全局融合方式 |
| --disable_token_gate | action | False | 禁用Token Bridge中的门控机制 |
| --disable_cnn_pos_embed | action | False | 禁用CNN特征的位置编码 |

### 训练相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --epochs | int | 3 | 训练轮数 |
| --lr | float | 1e-4 | 学习率 |
| --weight_decay | float | 1e-4 | 权重衰减系数 |
| --use_amp | action | False | 是否启用CUDA混合精度训练 |
| --max_train_batches | int | None | 最大训练Batch数，用于快速测试 |
| --max_val_batches | int | None | 最大验证Batch数，用于快速测试 |
| --torch_home | str | ./pretrained/torch_home | 预训练权重缓存目录 |
| --save_dir | str | ./outputs/checkpoints/run_train | 模型保存目录 |

## 项目结构
```
BridgeVision/
├─ README.md                    # 项目说明文档
├─ requirements.txt             # 依赖包列表
├─ run_train.py                 # 训练主入口
├─ run_eval.py                  # 评估主入口
├─ configs/                     # 配置文件目录（待实现）
│  ├─ model.yaml                # 模型配置
│  ├─ dataset.yaml              # 数据集配置
│  └─ train.yaml                # 训练配置
├─ src/                         # 核心源代码
│  ├─ models/                   # 模型模块
│  │  ├─ backbones/             # 主干网络
│  │  │  ├─ resnet_backbone.py  # ResNet系列主干实现
│  │  │  └─ vit_backbone.py     # ViT系列主干实现
│  │  ├─ projectors/            # 特征投影器
│  │  │  └─ projector.py        # 统一投影器实现
│  │  ├─ fusions/               # 特征融合模块
│  │  │  ├─ concat_fusion.py    # 拼接融合实现
│  │  │  ├─ gated_fusion.py     # 门控融合实现
│  │  │  └─ token_bridge_fusion.py # Token交叉注意力融合实现
│  │  └─ dual_encoder_model.py  # 双编码器融合主模型
│  ├─ data/                     # 数据处理模块
│  │  └─ datamodule.py          # 多数据集加载与预处理（支持6个数据集）
│  ├─ engine/                   # 训练引擎
│  │  └─ trainer.py             # 训练评估循环实现
│  ├─ losses/                   # 损失函数（待实现）
│  │  └─ classification_loss.py # 分类损失
│  └─ utils/                    # 工具函数（待实现）
│     └─ logger.py              # 日志工具
├─ outputs/                     # 输出目录
│  ├─ checkpoints/              # 模型权重保存
│  ├─ logs/                     # 日志文件
│  └─ figures/                  # 可视化图表
├─ pretrained/                  # 预训练权重缓存目录
│  └─ torch_home/               # PyTorch官方预训练权重缓存
├─ tests/                       # 单元测试（待实现）
│  └─ test_model_forward.py     # 模型前向测试
└─ tool/                        # 工具脚本
   └─ predownload_weights.py    # 预训练权重下载
```

## 基准测试结果
| 数据集 | 模型模式 | 融合模式 | Top-1准确率 |
|--------|----------|----------|-------------|
| Oxford-IIIT Pets (37类) | dual | concat | 待补充 |
| Oxford-IIIT Pets (37类) | dual | gated | 待补充 |
| Oxford-IIIT Pets (37类) | dual | token_bridge | 待补充 |
| Oxford-IIIT Pets (37类) | resnet_only | - | 待补充 |
| Oxford-IIIT Pets (37类) | vit_only | - | 待补充 |
| Flowers102 (102类) | dual | token_bridge | 待补充 |
| Food101 (101类) | dual | token_bridge | 待补充 |

> 欢迎提交您的实验结果到本项目！

## 开发计划
- [ ] 实现基于YAML的配置文件系统
- [ ] 添加更多主干网络支持（Swin Transformer、ConvNeXt等）
- [ ] 支持更多数据集（ImageNet、CIFAR等）
- [ ] 添加TensorBoard/WandB日志支持
- [ ] 完善单元测试覆盖
- [ ] 实现分布式训练支持
- [ ] 支持自监督预训练

## 许可证
MIT License

## 贡献
欢迎提交Issue和Pull Request！
