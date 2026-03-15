# src/models/projectors/projector.py

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Unified projector / 统一投影模块

    功能 / Features:
    1. 支持两种模式：
       Support two modes:
       - "linear": 单层线性投影
       - "mlp": 两层 MLP 投影
    2. 支持输入形状：
       Support input shapes:
       - [B, C]
       - [B, N, C]
    3. 输出最后一个维度映射到 output_dim
       Project the last dimension to output_dim

    设计思路 / Design idea:
    - 这里把 projector 看成“轻量适配器”
      We treat projector as a lightweight adapter
    - 不只是改维度，也顺便做一点分布对齐
      It not only changes feature dimension, but also mildly aligns feature distributions
    """

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
            layers = [
                nn.Linear(input_dim, output_dim),
            ]
            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            layers.extend(
                [
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            self.projector = nn.Sequential(*layers)

        elif projector_type == "mlp":
            layers = [
                nn.Linear(input_dim, self.hidden_dim),
            ]
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
            layers.extend(
                [
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            self.projector = nn.Sequential(*layers)

        else:
            raise ValueError(
                f"Unsupported projector_type={projector_type}. "
                f"Supported types: ['linear', 'mlp']"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass

        Args:
            x:
                - [B, C]
                - [B, N, C]

        Returns:
            projected_x:
                - [B, output_dim]
                - [B, N, output_dim]

        说明 / Notes:
        - Linear 和 LayerNorm 都作用在最后一个维度上
          Linear and LayerNorm operate on the last dimension
        - 所以无论输入是 2D 还是 3D，都可以直接处理
          Therefore both 2D and 3D inputs are supported directly
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Projector only supports 2D or 3D tensor, but got x.dim()={x.dim()}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input last dimension mismatch: expected {self.input_dim}, "
                f"but got {x.shape[-1]}."
            )

        return self.projector(x)


def _demo_forward() -> None:
    """
    简单前向传播测试 / Simple forward test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 全局特征 / Global feature
    x_global = torch.randn(2, 2048).to(device)

    projector_global = Projector(
        input_dim=2048,
        output_dim=512,
        projector_type="mlp",
        hidden_dim=1024,
        dropout=0.1,
        use_layernorm=True,
    ).to(device)

    # 2) token 特征 / Token features
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
    print("Global input shape  :", x_global.shape)
    print("Global output shape :", y_global.shape)
    print("Token input shape   :", x_tokens.shape)
    print("Token output shape  :", y_tokens.shape)


if __name__ == "__main__":
    _demo_forward()