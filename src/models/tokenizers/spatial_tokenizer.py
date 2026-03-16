from __future__ import annotations

import torch
import torch.nn as nn


class SpatialTokenizer(nn.Module):
    """
    将 CNN 特征图 [B, C, H, W] 转成空间 token [B, N, C] / Convert a CNN
    feature map [B, C, H, W] into spatial tokens [B, N, C], where N = H * W.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        feature_map: torch.Tensor,
        return_hw: bool = False,
    ):
        """
        参数 / Args:
            feature_map: [B, C, H, W]
            return_hw: 是否同时返回 (H, W) / Whether to also return (H, W)

        返回 / Returns:
            tokens: [B, H*W, C]
            hw: (H, W)，可选 / Optional
        """
        if feature_map.ndim != 4:
            raise ValueError(
                f"Expected feature_map with shape [B, C, H, W], but got {feature_map.shape}."
            )

        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]

        if return_hw:
            return tokens, (h, w)
        return tokens


def _demo_spatial_tokenizer() -> None:
    x = torch.randn(2, 2048, 7, 7)
    tokenizer = SpatialTokenizer()
    tokens, hw = tokenizer(x, return_hw=True)

    print("==== SpatialTokenizer Test ====")
    print("Input feature map :", x.shape)
    print("Output tokens     :", tokens.shape)
    print("HW                :", hw)


if __name__ == "__main__":
    _demo_spatial_tokenizer()
