from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_1d_sincos_pos_embed(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    构建 1D 正弦余弦位置编码 / Build 1D sine-cosine positional embedding

    Args:
        length:
            token 序列长度
            Token sequence length

        dim:
            token 维度
            Token dimension

    Returns:
        pos_embed: [1, length, dim]
    """
    if dim % 2 != 0:
        raise ValueError(f"1D positional encoding requires even dim, but got dim={dim}.")

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )  # [dim/2]

    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0).to(dtype=dtype)  # [1, L, D]


def build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    构建 2D 正弦余弦位置编码 / Build 2D sine-cosine positional embedding

    Args:
        height:
            特征图高度
            Feature-map height

        width:
            特征图宽度
            Feature-map width

        dim:
            token 维度
            Token dimension

    Returns:
        pos_embed: [1, H*W, dim]
    """
    if dim % 4 != 0:
        raise ValueError(f"2D positional encoding requires dim % 4 == 0, but got dim={dim}.")

    dim_half = dim // 2
    dim_quarter = dim // 4

    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    yy = yy.reshape(-1, 1)  # [N, 1]
    xx = xx.reshape(-1, 1)  # [N, 1]

    div_term = torch.exp(
        torch.arange(0, dim_quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim_quarter)
    )  # [dim/4]

    pe_y = torch.zeros(height * width, dim_half, device=device, dtype=torch.float32)
    pe_x = torch.zeros(height * width, dim_half, device=device, dtype=torch.float32)

    pe_y[:, 0::2] = torch.sin(yy * div_term)
    pe_y[:, 1::2] = torch.cos(yy * div_term)
    pe_x[:, 0::2] = torch.sin(xx * div_term)
    pe_x[:, 1::2] = torch.cos(xx * div_term)

    pe = torch.cat([pe_y, pe_x], dim=1)  # [N, D]
    return pe.unsqueeze(0).to(dtype=dtype)  # [1, N, D]


class SinCos1DPositionalEncoding(nn.Module):
    """
    1D 位置编码模块 / 1D positional encoding module

    适用场景 / Recommended use:
    - ViT patch token 序列的 bridge-space 外置位置编码
      External bridge-space positional encoding for ViT patch tokens

    注意 / Notes:
    - 对于 ViT 来说，这个外置编码应该是“可选”的
      For ViT tokens, this external positional encoding should be optional
    - 因为 backbone 内部通常已经使用过位置编码
      because ViT backbone usually already uses positional encoding internally
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]

        Returns:
            tokens_with_pos: [B, N, D]
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, D], but got {tokens.shape}.")

        _, n, d = tokens.shape
        pos_embed = build_1d_sincos_pos_embed(
            length=n,
            dim=d,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self.dropout(tokens + pos_embed)


class SinCos2DPositionalEncoding(nn.Module):
    """
    2D 位置编码模块 / 2D positional encoding module

    适用场景 / Recommended use:
    - CNN feature map flatten 后得到的 spatial tokens
      Spatial tokens obtained by flattening CNN feature maps

    输入仍然是 token 序列 / Input is still token sequence:
    - tokens: [B, N, D]
    - hw: (H, W)
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        hw: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]
            hw: (H, W)

        Returns:
            tokens_with_pos: [B, N, D]
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, D], but got {tokens.shape}.")

        h, w = hw
        _, n, d = tokens.shape

        if n != h * w:
            raise ValueError(
                f"Token count mismatch: n={n}, but h*w={h*w} for hw={hw}."
            )

        pos_embed = build_2d_sincos_pos_embed(
            height=h,
            width=w,
            dim=d,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self.dropout(tokens + pos_embed)


def _demo_positional_encoding() -> None:
    """
    简单测试 / Simple test
    """
    cnn_tokens = torch.randn(2, 49, 256)
    vit_tokens = torch.randn(2, 196, 256)

    cnn_pos = SinCos2DPositionalEncoding(dropout=0.0)
    vit_pos = SinCos1DPositionalEncoding(dropout=0.0)

    cnn_out = cnn_pos(cnn_tokens, hw=(7, 7))
    vit_out = vit_pos(vit_tokens)

    print("==== Positional Encoding Test ====")
    print("CNN tokens in   :", cnn_tokens.shape)
    print("CNN tokens out  :", cnn_out.shape)
    print("ViT tokens in   :", vit_tokens.shape)
    print("ViT tokens out  :", vit_out.shape)


if __name__ == "__main__":
    _demo_positional_encoding()