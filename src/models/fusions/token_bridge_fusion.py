from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Transformer 风格前馈网络 / Transformer-style feed-forward network

    作用 / Purpose:
    1. 在 cross-attention 之后进一步做 token 级非线性变换
       Apply token-wise nonlinear refinement after cross-attention
    2. 保持输入输出维度一致
       Keep input and output dimensions unchanged
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or feature_dim * 4

        self.pre_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.post_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass

        Args:
            tokens: [B, N, D]

        Returns:
            refined_tokens: [B, N, D]
        """
        residual = tokens
        tokens = self.pre_norm(tokens)
        tokens = self.ffn(tokens)
        tokens = residual + tokens
        tokens = self.post_norm(tokens)
        return tokens


class TokenBridgeFusion(nn.Module):
    """
    双向 token-level bridge fusion block / Bidirectional token-level bridge fusion block

    固定版结构 / Fixed structure:
    1. CNN <- ViT 双向交叉注意力中的一个方向
       CNN tokens query ViT tokens as key/value
    2. ViT <- CNN 双向交叉注意力中的另一个方向
       ViT tokens query CNN tokens as key/value
    3. 两边都做 gated residual update
       Apply gated residual update on both branches
    4. 两边都接一个 FFN refinement
       Apply one FFN refinement block on both branches

    输入 / Inputs:
    - cnn_tokens: [B, N_c, D]
    - vit_tokens: [B, N_v, D]

    输出 / Outputs:
    - fused_cnn_tokens: [B, N_c, D]
    - fused_vit_tokens: [B, N_v, D]
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_hidden_dim: int | None = None,
        ffn_hidden_dim: int | None = None,
        use_gate: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_layernorm = use_layernorm

        # ---------------------------
        # Norm layers / 归一化层
        # ---------------------------
        self.cnn_query_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.vit_query_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.cnn_kv_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.vit_kv_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()

        # ---------------------------
        # Bidirectional cross-attention / 双向交叉注意力
        # ---------------------------
        self.cnn_queries_vit_kv_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.vit_queries_cnn_kv_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ---------------------------
        # Token-wise gates / token 级门控
        # ---------------------------
        if use_gate:
            gate_hidden_dim = gate_hidden_dim or feature_dim

            self.cnn_gate_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, feature_dim),
            )

            self.vit_gate_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, feature_dim),
            )
        else:
            self.cnn_gate_mlp = None
            self.vit_gate_mlp = None

        # ---------------------------
        # Post-attention norms / 注意力后的归一化
        # ---------------------------
        self.cnn_output_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.vit_output_norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()

        # ---------------------------
        # FFN refinement / 前馈网络细化
        # ---------------------------
        self.cnn_ffn = FeedForwardBlock(
            feature_dim=feature_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.vit_ffn = FeedForwardBlock(
            feature_dim=feature_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        cnn_tokens: torch.Tensor,
        vit_tokens: torch.Tensor,
        return_gate: bool = False,
        return_attn_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            cnn_tokens: [B, N_c, D]
            vit_tokens: [B, N_v, D]
            return_gate:
                是否返回门控值
                Whether to return gate tensors
            return_attn_weights:
                是否返回注意力权重
                Whether to return attention weights

        Returns:
            {
                "fused_cnn_tokens": [B, N_c, D],
                "fused_vit_tokens": [B, N_v, D],
                "cnn_gate": [B, N_c, D], optional
                "vit_gate": [B, N_v, D], optional
                "cnn_attn_weights": ..., optional
                "vit_attn_weights": ..., optional
            }
        """
        if cnn_tokens.ndim != 3 or vit_tokens.ndim != 3:
            raise ValueError(
                f"Expected cnn_tokens and vit_tokens to be [B, N, D], "
                f"but got {cnn_tokens.shape} and {vit_tokens.shape}."
            )

        if cnn_tokens.shape[-1] != self.feature_dim or vit_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Last dimension must match feature_dim={self.feature_dim}, "
                f"but got {cnn_tokens.shape[-1]} and {vit_tokens.shape[-1]}."
            )

        # ---------------------------
        # 1. CNN <- ViT
        # CNN tokens as query, ViT tokens as key/value
        # ---------------------------
        cnn_q = self.cnn_query_norm(cnn_tokens)
        vit_kv_for_cnn = self.vit_kv_norm(vit_tokens)

        cnn_cross, cnn_attn_weights = self.cnn_queries_vit_kv_attn(
            query=cnn_q,
            key=vit_kv_for_cnn,
            value=vit_kv_for_cnn,
            need_weights=return_attn_weights,
        )

        # ---------------------------
        # 2. ViT <- CNN
        # ViT tokens as query, CNN tokens as key/value
        # ---------------------------
        vit_q = self.vit_query_norm(vit_tokens)
        cnn_kv_for_vit = self.cnn_kv_norm(cnn_tokens)

        vit_cross, vit_attn_weights = self.vit_queries_cnn_kv_attn(
            query=vit_q,
            key=cnn_kv_for_vit,
            value=cnn_kv_for_vit,
            need_weights=return_attn_weights,
        )

        outputs: dict[str, torch.Tensor] = {}

        # ---------------------------
        # 3. Gated residual update / 门控残差更新
        # ---------------------------
        if self.use_gate:
            assert self.cnn_gate_mlp is not None
            assert self.vit_gate_mlp is not None

            cnn_gate = torch.sigmoid(
                self.cnn_gate_mlp(torch.cat([cnn_tokens, cnn_cross], dim=-1))
            )
            vit_gate = torch.sigmoid(
                self.vit_gate_mlp(torch.cat([vit_tokens, vit_cross], dim=-1))
            )

            fused_cnn_tokens = cnn_tokens + cnn_gate * cnn_cross
            fused_vit_tokens = vit_tokens + vit_gate * vit_cross

            if return_gate:
                outputs["cnn_gate"] = cnn_gate
                outputs["vit_gate"] = vit_gate
        else:
            fused_cnn_tokens = cnn_tokens + cnn_cross
            fused_vit_tokens = vit_tokens + vit_cross

        fused_cnn_tokens = self.cnn_output_norm(fused_cnn_tokens)
        fused_vit_tokens = self.vit_output_norm(fused_vit_tokens)

        # ---------------------------
        # 4. FFN refinement / FFN 细化
        # ---------------------------
        fused_cnn_tokens = self.cnn_ffn(fused_cnn_tokens)
        fused_vit_tokens = self.vit_ffn(fused_vit_tokens)

        outputs["fused_cnn_tokens"] = fused_cnn_tokens
        outputs["fused_vit_tokens"] = fused_vit_tokens

        if return_attn_weights:
            outputs["cnn_attn_weights"] = cnn_attn_weights
            outputs["vit_attn_weights"] = vit_attn_weights

        return outputs


def _demo_token_bridge_fusion() -> None:
    """
    简单测试 / Simple test
    """
    cnn_tokens = torch.randn(2, 50, 256)
    vit_tokens = torch.randn(2, 197, 256)

    fusion = TokenBridgeFusion(
        feature_dim=256,
        num_heads=8,
        dropout=0.1,
        gate_hidden_dim=256,
        ffn_hidden_dim=512,
        use_gate=True,
        use_layernorm=True,
    )

    outputs = fusion(
        cnn_tokens=cnn_tokens,
        vit_tokens=vit_tokens,
        return_gate=True,
        return_attn_weights=False,
    )

    print("==== TokenBridgeFusion Test ====")
    print("CNN tokens in       :", cnn_tokens.shape)
    print("ViT tokens in       :", vit_tokens.shape)
    print("Fused CNN tokens    :", outputs["fused_cnn_tokens"].shape)
    print("Fused ViT tokens    :", outputs["fused_vit_tokens"].shape)
    print("CNN gate            :", outputs["cnn_gate"].shape)
    print("ViT gate            :", outputs["vit_gate"].shape)


if __name__ == "__main__":
    _demo_token_bridge_fusion()