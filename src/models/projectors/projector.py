from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class Projector(nn.Module):
    """将最后一维投影到共享隐藏空间 / Project the last feature dimension into
    a shared hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projector_type: Literal["linear", "mlp"] = "mlp",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projector_type = projector_type
        self.hidden_dim = hidden_dim if hidden_dim is not None else output_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        if projector_type == "linear":
            layers = [nn.Linear(input_dim, output_dim)]
            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.projector = nn.Sequential(*layers)
        elif projector_type == "mlp":
            layers = [nn.Linear(input_dim, self.hidden_dim)]
            if use_layernorm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.extend(
                [
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, output_dim),
                ]
            )
            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.projector = nn.Sequential(*layers)
        else:
            raise ValueError(
                f"Unsupported projector_type={projector_type}. "
                "Supported types: ['linear', 'mlp']"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """沿最后一维投影 [B, C] 或 [B, N, C] 张量 / Project [B, C] or [B, N,
        C] tensors along the last dimension."""
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Projector only supports 2D or 3D tensors, but got x.dim()={x.dim()}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input last dimension mismatch: expected {self.input_dim}, "
                f"but got {x.shape[-1]}."
            )

        return self.projector(x)


def _demo_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_global = torch.randn(2, 2048).to(device)
    projector_global = Projector(
        input_dim=2048,
        output_dim=512,
        projector_type="mlp",
        hidden_dim=1024,
        dropout=0.1,
        use_layernorm=True,
    ).to(device)

    x_tokens = torch.randn(2, 49, 2048).to(device)
    projector_tokens = Projector(
        input_dim=2048,
        output_dim=512,
        projector_type="linear",
        dropout=0.1,
        use_layernorm=True,
    ).to(device)

    projector_global.eval()
    projector_tokens.eval()

    with torch.no_grad():
        y_global = projector_global(x_global)
        y_tokens = projector_tokens(x_tokens)

    print("==== Projector Forward Test ====")
    print("Global input shape :", x_global.shape)
    print("Global output shape:", y_global.shape)
    print("Token input shape  :", x_tokens.shape)
    print("Token output shape :", y_tokens.shape)


if __name__ == "__main__":
    _demo_forward()
