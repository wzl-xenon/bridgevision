# src/models/backbones/vit_backbone.py

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


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone wrapper / ViT 主干网络封装

    功能 / Features:
    1. 支持 torchvision 的 vit_b_16 / vit_b_32 / vit_l_16 / vit_l_32
       Support standard torchvision ViT backbones
    2. 支持是否加载预训练权重
       Support pretrained weights
    3. 支持是否冻结 backbone 参数
       Support freezing backbone parameters
    4. 返回 cls token、patch tokens 和全部 tokens
       Return cls token, patch tokens, and all tokens
    """

    _MODEL_FACTORY = {
        "vit_b_16": models.vit_b_16,
        "vit_b_32": models.vit_b_32,
        "vit_l_16": models.vit_l_16,
        "vit_l_32": models.vit_l_32,
    }

    _WEIGHT_FACTORY = {
        "vit_b_16": models.ViT_B_16_Weights,
        "vit_b_32": models.ViT_B_32_Weights,
        "vit_l_16": models.ViT_L_16_Weights,
        "vit_l_32": models.ViT_L_32_Weights,
    }

    _HIDDEN_DIM = {
        "vit_b_16": 768,
        "vit_b_32": 768,
        "vit_l_16": 1024,
        "vit_l_32": 1024,
    }

    def __init__(
        self,
        model_name: str = "vit_b_16",
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

        # 根据是否使用预训练权重初始化 ViT
        # Initialize ViT with or without pretrained weights
        if pretrained:
            self.backbone = model_fn(weights=weight_enum.DEFAULT)
        else:
            self.backbone = model_fn(weights=None)

        self.model_name = model_name
        self.hidden_dim = self._HIDDEN_DIM[model_name]

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

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取所有 token / Extract all tokens

        Args:
            x: [B, 3, H, W]

        Returns:
            all_tokens: [B, 1 + N, D]
                - 第 0 个 token 是 cls token
                  The first token is the cls token
                - 后面是 patch tokens
                  The rest are patch tokens
        """
        # 1. 把输入图像转成 patch embeddings
        # Convert image to patch embeddings
        x = self.backbone._process_input(x)  # [B, N, D]
        batch_size = x.shape[0]

        # 2. 拼接 cls token
        # Concatenate cls token
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)                               # [B, 1+N, D]

        # 3. 送入 Transformer encoder
        # Feed into Transformer encoder
        x = self.backbone.encoder(x)                                       # [B, 1+N, D]

        return x

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅返回 cls token 特征 / Return cls token feature only

        Args:
            x: [B, 3, H, W]

        Returns:
            cls_feature: [B, D]
        """
        all_tokens = self.forward_tokens(x)
        cls_feature = all_tokens[:, 0]
        return cls_feature

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Returns:
            {
                "cls_feature": [B, D],
                "patch_tokens": [B, N, D],
                "all_tokens": [B, 1+N, D],
            }
        """
        all_tokens = self.forward_tokens(x)
        cls_feature = all_tokens[:, 0]
        patch_tokens = all_tokens[:, 1:]

        return {
            "cls_feature": cls_feature,
            "patch_tokens": patch_tokens,
            "all_tokens": all_tokens,
        }


def _demo_forward() -> None:
    """
    简单前向传播测试 / Simple forward test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造一个假的输入 batch
    # Create a fake input batch
    x = torch.randn(2, 3, 224, 224).to(device)

    model = ViTBackbone(
        model_name="vit_b_16",
        pretrained=False,
        freeze=False,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== ViT Backbone Forward Test ====")
    print("Input shape       :", x.shape)
    print("CLS feature shape :", outputs["cls_feature"].shape)
    print("Patch tokens shape:", outputs["patch_tokens"].shape)
    print("All tokens shape  :", outputs["all_tokens"].shape)


if __name__ == "__main__":
    _demo_forward()