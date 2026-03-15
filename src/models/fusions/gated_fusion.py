# src/models/fusions/gated_fusion.py

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion / 门控融合模块

    功能 / Features:
    1. 输入两个全局特征向量
       Take two global feature vectors as input
    2. 根据二者拼接后的信息生成门控系数
       Generate gate values from concatenated features
    3. 用门控系数对两路特征进行加权融合
       Fuse two branches with gate-weighted combination

    输入 / Input:
        feature_a: [B, D]
        feature_b: [B, D]

    输出 / Output:
        fused_feature: [B, D]

    数学形式 / Formula:
        gate = sigmoid(MLP([a; b]))
        fused = gate * a + (1 - gate) * b

    设计思路 / Design idea:
    - concat fusion 更像“先拼起来再学”
      Concat fusion is more like "concatenate first, then learn"
    - gated fusion 更像“让模型决定当前样本更信哪一路”
      Gated fusion lets the model decide which branch to trust more for each sample
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        refine_output: bool = True,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.refine_output = refine_output

        gate_layers = [
            nn.Linear(feature_dim * 2, self.hidden_dim),
        ]
        if use_layernorm:
            gate_layers.append(nn.LayerNorm(self.hidden_dim))
        gate_layers.extend(
            [
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, feature_dim),
                nn.Sigmoid(),
            ]
        )
        self.gate_network = nn.Sequential(*gate_layers)

        if refine_output:
            refine_layers = [
                nn.Linear(feature_dim, feature_dim),
            ]
            if use_layernorm:
                refine_layers.append(nn.LayerNorm(feature_dim))
            refine_layers.extend(
                [
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            self.refine = nn.Sequential(*refine_layers)
        else:
            self.refine = nn.Identity()

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            feature_a: [B, D]
            feature_b: [B, D]
            return_gate:
                是否返回门控值
                Whether to return gate values

        Returns:
            fused_feature: [B, D]
            or
            (fused_feature, gate): ([B, D], [B, D])
        """
        if feature_a.dim() != 2 or feature_b.dim() != 2:
            raise ValueError("GatedFusion only supports 2D tensors of shape [B, D].")

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

        gate_input = torch.cat([feature_a, feature_b], dim=-1)  # [B, 2D]
        gate = self.gate_network(gate_input)                     # [B, D], in [0, 1]

        fused_feature = gate * feature_a + (1.0 - gate) * feature_b
        fused_feature = self.refine(fused_feature)

        if return_gate:
            return fused_feature, gate
        return fused_feature


def _demo_forward() -> None:
    """
    简单前向传播测试 / Simple forward test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_a = torch.randn(2, 512).to(device)
    feature_b = torch.randn(2, 512).to(device)

    model = GatedFusion(
        feature_dim=512,
        hidden_dim=512,
        dropout=0.1,
        use_layernorm=True,
        refine_output=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        fused_feature, gate = model(feature_a, feature_b, return_gate=True)

    print("==== Gated Fusion Forward Test ====")
    print("Feature A shape    :", feature_a.shape)
    print("Feature B shape    :", feature_b.shape)
    print("Gate shape         :", gate.shape)
    print("Fused feature shape:", fused_feature.shape)
    print("Gate min / max     :", gate.min().item(), gate.max().item())


if __name__ == "__main__":
    _demo_forward()