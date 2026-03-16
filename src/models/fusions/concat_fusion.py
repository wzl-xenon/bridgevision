from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """通过拼接后接 MLP 融合两个全局特征 / Fuse two global features by
    concatenation followed by an MLP."""

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

        layers = [nn.Linear(feature_dim * 2, self.hidden_dim)]
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
        layers.extend([nn.GELU(), nn.Dropout(dropout)])

        self.fusion = nn.Sequential(*layers)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        """将两个 [B, D] 张量融合为一个 [B, D] 张量 / Fuse two tensors shaped
        [B, D] into one tensor shaped [B, D]."""
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

        fused_input = torch.cat([feature_a, feature_b], dim=-1)
        return self.fusion(fused_input)


def _demo_forward() -> None:
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
