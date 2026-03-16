# src/models/backbones/resnet_backbone.py

from __future__ import annotations

from typing import Dict

import os

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import torch
import torch.nn as nn

# 为部分缺少 torchvision::nms 注册的环境提供兼容占位符
# Provide a compatibility stub for environments missing torchvision::nms registration
try:
    lib = torch.library.Library("torchvision", "DEF")
    lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from torchvision import models


class ResNetBackbone(nn.Module):
    """
    ResNet backbone wrapper / ResNet 主干网络封装

    功能 / Features:
    1. 支持 torchvision 的 resnet18/resnet34/resnet50/resnet101
       Support resnet18/resnet34/resnet50/resnet101 from torchvision
    2. 支持是否加载预训练权重
       Support pretrained weights
    3. 支持是否冻结 backbone 参数
       Support freezing backbone parameters
    4. 返回最后一层空间特征图和全局池化特征
       Return both spatial feature map and pooled feature
    """

    _MODEL_FACTORY = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }

    _WEIGHT_FACTORY = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
        "resnet101": models.ResNet101_Weights,
    }

    _OUT_CHANNELS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        if model_name not in self._MODEL_FACTORY:
            raise ValueError(
                f"Unsupported model_name={model_name}. "
                f"Supported models: {list(self._MODEL_FACTORY.keys())}"
            )

        model_fn = self._MODEL_FACTORY[model_name]
        weight_enum = self._WEIGHT_FACTORY[model_name]

        # 根据是否使用预训练权重初始化模型
        # Initialize model with or without pretrained weights
        if pretrained:
            backbone = model_fn(weights=weight_enum.DEFAULT)
        else:
            backbone = model_fn(weights=None)

        # 保留到 layer4，去掉分类头 fc
        # Keep layers up to layer4, remove final classification head
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.model_name = model_name
        self.out_channels = self._OUT_CHANNELS[model_name]

        if freeze:
            self.freeze_parameters()

    def freeze_parameters(self) -> None:
        """
        冻结所有参数 / Freeze all parameters
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """
        解冻所有参数 / Unfreeze all parameters
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取最后的空间特征图 / Extract final spatial feature map

        Args:
            x: [B, 3, H, W]

        Returns:
            feature_map: [B, C, H_out, W_out]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取全局池化后的特征 / Extract pooled global feature

        Args:
            x: [B, 3, H, W]

        Returns:
            pooled_feature: [B, C]
        """
        feature_map = self.forward_features(x)
        pooled = self.avgpool(feature_map)           # [B, C, 1, 1]
        pooled = torch.flatten(pooled, start_dim=1)  # [B, C]
        return pooled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Returns:
            {
                "feature_map": [B, C, H_out, W_out],
                "pooled_feature": [B, C],
            }
        """
        feature_map = self.forward_features(x)
        pooled_feature = self.avgpool(feature_map)
        pooled_feature = torch.flatten(pooled_feature, start_dim=1)

        return {
            "feature_map": feature_map,
            "pooled_feature": pooled_feature,
        }


def _demo_forward() -> None:
    """
    简单前向传播测试 / Simple forward test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造一个假的输入 batch
    # Create a fake input batch
    x = torch.randn(2, 3, 224, 224).to(device)

    model = ResNetBackbone(
        model_name="resnet50",
        pretrained=False,
        freeze=False,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== ResNet Backbone Forward Test ====")
    print("Input shape        :", x.shape)
    print("Feature map shape  :", outputs["feature_map"].shape)
    print("Pooled feature shape:", outputs["pooled_feature"].shape)


if __name__ == "__main__":
    _demo_forward()