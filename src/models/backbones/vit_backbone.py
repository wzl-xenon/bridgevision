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


class ViTBackbone(nn.Module):
    """封装 torchvision ViT，并暴露 cls 与 patch token / Wrap a torchvision
    ViT and expose cls and patch tokens."""

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
        self.backbone = model_fn(weights=weight_enum.DEFAULT if pretrained else None)

        self.model_name = model_name
        self.hidden_dim = self._HIDDEN_DIM[model_name]

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

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """返回完整 token 序列 / Return the full token sequence with shape
        [B, 1 + N, D]."""
        x = self.backbone._process_input(x)
        batch_size = x.shape[0]

        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.backbone.encoder(x)
        return x

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """只返回 cls token / Return only the cls token with shape [B, D]."""
        all_tokens = self.forward_tokens(x)
        return all_tokens[:, 0]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回 cls 特征、patch token 和完整 token 序列 / Return cls features,
        patch tokens, and the full token sequence."""
        all_tokens = self.forward_tokens(x)
        cls_feature = all_tokens[:, 0]
        patch_tokens = all_tokens[:, 1:]

        return {
            "cls_feature": cls_feature,
            "patch_tokens": patch_tokens,
            "all_tokens": all_tokens,
        }


def _demo_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Input shape        :", x.shape)
    print("CLS feature shape  :", outputs["cls_feature"].shape)
    print("Patch tokens shape :", outputs["patch_tokens"].shape)
    print("All tokens shape   :", outputs["all_tokens"].shape)


if __name__ == "__main__":
    _demo_forward()
