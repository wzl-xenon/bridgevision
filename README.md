# BridgeVision

## 项目简介
BridgeVision是一个基于PyTorch的开源双编码器多模态融合视觉分类框架，创新性地结合了卷积神经网络（CNN）和视觉Transformer（ViT）两种不同架构的优势，通过可配置的特征融合策略实现更优的图像分类性能。

框架采用高度模块化设计，所有核心组件均可灵活替换和扩展，既可以作为学术研究的baseline，也可以作为工业界落地的基础框架。

## 核心特性
✨ **双编码器架构**：同时使用ResNet系列CNN和ViT系列Transformer提取互补特征
🔧 **多融合模式支持**：内置Concat拼接融合和Gated门控自适应融合两种策略
⚙️ **高度模块化**：主干网络、投影器、融合模块、分类头全部分离，可灵活配置替换
📦 **开箱即用**：内置Oxford-IIIT Pets和Flowers102两个标准数据集的完整支持
🚀 **工程化完备**：支持自动混合精度训练、最佳Checkpoint自动保存、进度条显示、指标记录
🎛️ **高可配置性**：所有核心参数支持命令行动态调整，无需修改代码

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
> 注：当前requirements.txt为空，您可以手动安装上述依赖包，或根据您的CUDA版本选择合适的PyTorch版本。

3. 预下载预训练权重（可选，首次训练时会自动下载）
```bash
python tool/predownload_weights.py
```

## 快速开始
### 1. 训练模型
训练默认配置（Concat融合 + MLP投影器 + ResNet50 + ViT-B/16）：
```bash
python run_train.py \
  --dataset_name oxfordiiitpet \
  --pretrained_backbones \
  --fusion_type concat \
  --projector_type mlp \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_dir ./outputs/checkpoints/concat_exp
```

训练门控融合版本：
```bash
python run_train.py \
  --dataset_name flowers102 \
  --pretrained_backbones \
  --fusion_type gated \
  --epochs 20 \
  --batch_size 4 \
  --use_amp
```

### 2. 评估模型
在测试集上评估训练好的模型：
```bash
python run_eval.py \
  --dataset_name oxfordiiitpet \
  --checkpoint_path ./outputs/checkpoints/concat_exp/best_model.pt \
  --split test
```

在验证集上快速评估：
```bash
python run_eval.py \
  --dataset_name flowers102 \
  --checkpoint_path ./outputs/checkpoints/gated_exp/best_model.pt \
  --split val \
  --max_eval_batches 10
```

## 主要参数说明
### 数据相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --dataset_name | str | oxfordiiitpet | 数据集名称，支持 oxfordiiitpet / flowers102 |
| --data_root | str | ./data | 数据集存放根目录 |
| --image_size | int | 224 | 模型输入图像尺寸 |
| --batch_size | int | 4 | Batch大小 |
| --num_workers | int | 0 | DataLoader worker数量 |
| --download | action | False | 是否自动下载数据集 |
| --split_seed | int | 42 | 数据集划分随机种子 |

### 模型相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --resnet_name | str | resnet50 | ResNet主干版本，支持 resnet18/34/50/101 |
| --vit_name | str | vit_b_16 | ViT主干版本，支持 vit_b_16/vit_b_32/vit_l_16/vit_l_32 |
| --pretrained_backbones | action | False | 是否加载预训练主干权重 |
| --freeze_backbones | action | False | 是否冻结主干网络参数 |
| --projector_type | str | mlp | 投影器类型，支持 linear / mlp |
| --fusion_type | str | concat | 融合模式，支持 concat / gated |
| --fusion_dim | int | 512 | 特征融合维度 |
| --dropout | float | 0.1 | Dropout比例 |

### 训练相关参数
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --epochs | int | 3 | 训练轮数 |
| --lr | float | 1e-4 | 学习率 |
| --weight_decay | float | 1e-4 | 权重衰减系数 |
| --use_amp | action | False | 是否启用CUDA混合精度训练 |
| --max_train_batches | int | None | 最大训练Batch数，用于快速测试 |
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
│  │  │  └─ gated_fusion.py     # 门控融合实现
│  │  └─ dual_encoder_model.py  # 双编码器融合主模型
│  ├─ data/                     # 数据处理模块
│  │  └─ datamodule.py          # 数据集加载与预处理
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
├─ tests/                       # 单元测试（待实现）
│  └─ test_model_forward.py     # 模型前向测试
└─ tool/                        # 工具脚本
   └─ predownload_weights.py    # 预训练权重下载
```

## 基准测试结果
| 数据集 | 融合模式 | 主干网络 | Top-1准确率 |
|--------|----------|----------|-------------|
| Oxford-IIIT Pets | Concat | ResNet50 + ViT-B/16 | 待补充 |
| Oxford-IIIT Pets | Gated | ResNet50 + ViT-B/16 | 待补充 |
| Flowers102 | Concat | ResNet50 + ViT-B/16 | 待补充 |
| Flowers102 | Gated | ResNet50 + ViT-B/16 | 待补充 |

> 欢迎提交您的实验结果到本项目！

## 开发计划
- [ ] 实现基于YAML的配置文件系统
- [ ] 添加更多主干网络支持（Swin Transformer、ConvNeXt等）
- [ ] 实现Token级特征融合
- [ ] 支持更多数据集（ImageNet、CIFAR等）
- [ ] 添加TensorBoard/WandB日志支持
- [ ] 完善单元测试覆盖
- [ ] 实现分布式训练支持

## 许可证
MIT License

## 贡献
欢迎提交Issue和Pull Request！