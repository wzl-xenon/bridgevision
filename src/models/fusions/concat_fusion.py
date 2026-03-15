# src/models/fusions/concat_fusion.py

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Concat-based fusion / 基于拼接的融合模块

    功能 / Features:
    1. 输入两个全局特征向量
       Take two global feature vectors as input
    2. 先拼接，再通过一个 FFN 融合
       Concatenate first, then fuse with an FFN
    3. 输出一个融合后的特征向量
       Output one fused feature vector

    输入 / Input:
        feature_a: [B, D]
        feature_b: [B, D]

    输出 / Output:
        fused_feature: [B, D]

    设计思路 / Design idea:
    - 这是最朴素、最稳定的 baseline
      This is the simplest and most stable baseline
    - 它不显式建模两分支交互，只是把学习交给后面的 MLP
      It does not explicitly model branch interaction; the MLP learns the fusion
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        layers = [
            nn.Linear(feature_dim * 2, self.hidden_dim),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        layers.extend(
            [
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, feature_dim),
            ]
        )
        if use_layernorm:
            layers.append(nn.LayerNorm(feature_dim))
        layers.extend(
            [
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )

        self.fusion = nn.Sequential(*layers)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass

        Args:
            feature_a: [B, D]
            feature_b: [B, D]

        Returns:
            fused_feature: [B, D]
        """
        if feature_a.dim() != 2 or feature_b.dim() != 2:
            raise ValueError("ConcatFusion only supports 2D tensors of shape [B, D].")

        if feature_a.shape != feature_b.shape:
            raise ValueError(
                f"Shape mismatch: feature_a.shape={feature_a.shape}, "
                f"feature_b.shape={feature_b.shape}"
            )

        if feature_a.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, "
                f"but got {feature_a.shape[-1]}"
            )

        fused_input = torch.cat([feature_a, feature_b], dim=-1)  # [B, 2D]
        fused_feature = self.fusion(fused_input)                  # [B, D]
        return fused_feature


def _demo_forward() -> None:
    """
    简单前向传播测试 / Simple forward test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_a = torch.randn(2, 512).to(device)
    feature_b = torch.randn(2, 512).to(device)

    model = ConcatFusion(
        feature_dim=512,
        hidden_dim=512,
        dropout=0.1,
        use_layernorm=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        fused_feature = model(feature_a, feature_b)

    print("==== Concat Fusion Forward Test ====")
    print("Feature A shape    :", feature_a.shape)
    print("Feature B shape    :", feature_b.shape)
    print("Fused feature shape:", fused_feature.shape)


if __name__ == "__main__":
    _demo_forward()