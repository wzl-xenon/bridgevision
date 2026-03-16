from __future__ import annotations

from typing import Dict

import os

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import torch
import torch.nn as nn

# 为缺少 torchvision::nms 的环境提供兼容占位符 / Provide a compatibility
# stub for environments where torchvision::nms is unavailable during import.
try:
    lib = torch.library.Library("torchvision", "DEF")
    lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from torchvision import models


class ResNetBackbone(nn.Module):
    """封装 torchvision ResNet，并暴露空间特征和池化特征 / Wrap a torchvision
    ResNet and expose spatial and pooled features."""

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
        backbone = model_fn(weights=weight_enum.DEFAULT if pretrained else None)

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
        """冻结全部主干参数 / Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """解冻全部主干参数 / Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回最后一层空间特征图 / Return the final spatial feature map with
        shape [B, C, H, W]."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """返回全局池化特征 / Return the pooled global feature with shape [B, C]."""
        feature_map = self.forward_features(x)
        pooled = self.avgpool(feature_map)
        return torch.flatten(pooled, start_dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """同时返回空间特征图和池化特征 / Return both the spatial feature map
        and pooled feature."""
        feature_map = self.forward_features(x)
        pooled_feature = self.avgpool(feature_map)
        pooled_feature = torch.flatten(pooled_feature, start_dim=1)

        return {
            "feature_map": feature_map,
            "pooled_feature": pooled_feature,
        }


def _demo_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Input shape         :", x.shape)
    print("Feature map shape   :", outputs["feature_map"].shape)
    print("Pooled feature shape:", outputs["pooled_feature"].shape)


if __name__ == "__main__":
    _demo_forward()
