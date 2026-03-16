from __future__ import annotations

import torch
import torch.nn as nn


class FixedTokenResampler(nn.Module):
    """
    将变长 token 序列 [B, N, D] 重采样为固定长度 / Resample a variable-length
    token sequence [B, N, D] to a fixed number of tokens [B, target_tokens, D]
    with adaptive average pooling.
    """

    def __init__(self, target_tokens: int) -> None:
        super().__init__()

        if target_tokens < 1:
            raise ValueError(f"target_tokens must be >= 1, but got {target_tokens}.")

        self.target_tokens = target_tokens
        self.pool = nn.AdaptiveAvgPool1d(target_tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, D], but got {tokens.shape}.")

        pooled = self.pool(tokens.transpose(1, 2))
        return pooled.transpose(1, 2).contiguous()
