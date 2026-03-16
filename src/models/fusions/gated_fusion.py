from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """用逐维门控融合两个全局特征 / Fuse two global features with a learned
    per-dimension gate."""

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

        gate_layers = [nn.Linear(feature_dim * 2, self.hidden_dim)]
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
            refine_layers = [nn.Linear(feature_dim, feature_dim)]
            if use_layernorm:
                refine_layers.append(nn.LayerNorm(feature_dim))
            refine_layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.refine = nn.Sequential(*refine_layers)
        else:
            self.refine = nn.Identity()

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """融合两个 [B, D] 张量，并可选返回门控 / Fuse two tensors shaped
        [B, D] and optionally return the gate."""
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

        gate_input = torch.cat([feature_a, feature_b], dim=-1)
        gate = self.gate_network(gate_input)

        fused_feature = gate * feature_a + (1.0 - gate) * feature_b
        fused_feature = self.refine(fused_feature)

        if return_gate:
            return fused_feature, gate
        return fused_feature


class TokenDimGatedFusion(nn.Module):
    """对齐后的 token 序列逐 token、逐维门控融合 / Fuse aligned token
    sequences shaped [B, N, D] with a [B, N, D] gate."""

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

        gate_layers = [nn.Linear(feature_dim * 2, self.hidden_dim)]
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
            refine_layers = [nn.Linear(feature_dim, feature_dim)]
            if use_layernorm:
                refine_layers.append(nn.LayerNorm(feature_dim))
            refine_layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.refine = nn.Sequential(*refine_layers)
        else:
            self.refine = nn.Identity()

    def forward(
        self,
        token_a: torch.Tensor,
        token_b: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """融合两个 [B, N, D] 张量，并可选返回门控 / Fuse two tensors shaped
        [B, N, D] and optionally return the gate."""
        if token_a.dim() != 3 or token_b.dim() != 3:
            raise ValueError("TokenDimGatedFusion only supports 3D tensors of shape [B, N, D].")

        if token_a.shape != token_b.shape:
            raise ValueError(
                f"Shape mismatch: token_a.shape={token_a.shape}, token_b.shape={token_b.shape}"
            )

        if token_a.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, "
                f"but got {token_a.shape[-1]}"
            )

        gate_input = torch.cat([token_a, token_b], dim=-1)
        gate = self.gate_network(gate_input)

        fused_tokens = gate * token_a + (1.0 - gate) * token_b
        fused_tokens = self.refine(fused_tokens)

        if return_gate:
            return fused_tokens, gate
        return fused_tokens


def _demo_forward() -> None:
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
