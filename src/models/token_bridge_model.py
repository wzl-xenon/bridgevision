from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn

from src.models.backbones.resnet_backbone import ResNetBackbone
from src.models.backbones.vit_backbone import ViTBackbone
from src.models.embeddings.positional_encoding import (
    SinCos1DPositionalEncoding,
    SinCos2DPositionalEncoding,
)
from src.models.fusions.token_bridge_fusion import TokenBridgeFusion
from src.models.projectors.projector import Projector
from src.models.tokenizers.spatial_tokenizer import SpatialTokenizer


class ClassificationHead(nn.Module):
    """将融合特征映射为分类 logits / Map a fused feature vector to
    classification logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BranchGateFusion(nn.Module):
    """用可学习门控融合 CNN 和 ViT 的池化特征 / Fuse pooled CNN and ViT
    features with a learnable gate."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or feature_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        cnn_feature: torch.Tensor,
        vit_feature: torch.Tensor,
        return_gate: bool = False,
    ) -> dict[str, torch.Tensor]:
        gate = torch.sigmoid(self.gate_mlp(torch.cat([cnn_feature, vit_feature], dim=-1)))
        fused_feature = gate * cnn_feature + (1.0 - gate) * vit_feature
        fused_feature = self.output_norm(fused_feature)

        outputs = {"fused_feature": fused_feature}
        if return_gate:
            outputs["branch_gate"] = gate
        return outputs


class TokenBridgeModel(nn.Module):
    """带 CNN 和 ViT 双分支的 token-bridge 原型模型 / Prototype
    token-bridge model with CNN and ViT branches."""

    SUPPORTED_POOL_TYPES = {"mean"}

    def __init__(
        self,
        num_classes: int,
        resnet_name: str = "resnet50",
        vit_name: str = "vit_b_16",
        pretrained_backbones: bool = False,
        freeze_backbones: bool = False,
        projector_type: Literal["linear", "mlp"] = "linear",
        bridge_dim: int = 256,
        projector_hidden_dim: int | None = None,
        bridge_num_heads: int = 8,
        token_gate_hidden_dim: int | None = None,
        branch_gate_hidden_dim: int | None = None,
        use_token_gate: bool = True,
        use_cnn_pos_embed: bool = True,
        use_vit_pos_embed: bool = False,
        pool_type: Literal["mean"] = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if pool_type not in self.SUPPORTED_POOL_TYPES:
            raise ValueError(
                f"Unsupported pool_type={pool_type}. "
                f"Supported: {sorted(self.SUPPORTED_POOL_TYPES)}"
            )

        self.num_classes = num_classes
        self.resnet_name = resnet_name
        self.vit_name = vit_name
        self.pretrained_backbones = pretrained_backbones
        self.freeze_backbones = freeze_backbones
        self.projector_type = projector_type
        self.bridge_dim = bridge_dim
        self.use_token_gate = use_token_gate
        self.use_cnn_pos_embed = use_cnn_pos_embed
        self.use_vit_pos_embed = use_vit_pos_embed
        self.pool_type = pool_type

        self.resnet_backbone = ResNetBackbone(
            model_name=resnet_name,
            pretrained=pretrained_backbones,
            freeze=freeze_backbones,
        )
        self.vit_backbone = ViTBackbone(
            model_name=vit_name,
            pretrained=pretrained_backbones,
            freeze=freeze_backbones,
        )

        resnet_out_dim = self.resnet_backbone.out_channels
        vit_out_dim = self.vit_backbone.hidden_dim

        self.spatial_tokenizer = SpatialTokenizer()
        self.cnn_projector = Projector(
            input_dim=resnet_out_dim,
            output_dim=bridge_dim,
            projector_type=projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )
        self.vit_projector = Projector(
            input_dim=vit_out_dim,
            output_dim=bridge_dim,
            projector_type=projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        self.cnn_pos_embed = (
            SinCos2DPositionalEncoding(dropout=dropout) if use_cnn_pos_embed else None
        )
        self.vit_pos_embed = (
            SinCos1DPositionalEncoding(dropout=dropout) if use_vit_pos_embed else None
        )

        self.bridge = TokenBridgeFusion(
            feature_dim=bridge_dim,
            num_heads=bridge_num_heads,
            dropout=dropout,
            gate_hidden_dim=token_gate_hidden_dim,
            use_gate=use_token_gate,
            use_layernorm=True,
        )
        self.branch_fusion = BranchGateFusion(
            feature_dim=bridge_dim,
            hidden_dim=branch_gate_hidden_dim,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(
            input_dim=bridge_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def extract_cnn_tokens(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor | tuple[int, int]]:
        """提取投影后的 CNN 空间 token 与原始特征图 / Extract projected CNN
        spatial tokens and the source feature map."""
        resnet_out = self.resnet_backbone(x)
        feature_map = resnet_out["feature_map"]

        cnn_tokens, hw = self.spatial_tokenizer(
            feature_map,
            return_hw=True,
        )
        cnn_tokens = self.cnn_projector(cnn_tokens)

        if self.cnn_pos_embed is not None:
            cnn_tokens = self.cnn_pos_embed(cnn_tokens, hw=hw)

        return {
            "cnn_feature_map": feature_map,
            "cnn_tokens": cnn_tokens,
            "cnn_hw": hw,
        }

    def extract_vit_tokens(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """提取投影后的 ViT patch token / Extract projected ViT patch tokens."""
        vit_out = self.vit_backbone(x)
        patch_tokens = vit_out["patch_tokens"]
        vit_tokens = self.vit_projector(patch_tokens)

        if self.vit_pos_embed is not None:
            vit_tokens = self.vit_pos_embed(vit_tokens)

        return {
            "vit_patch_tokens": patch_tokens,
            "vit_tokens": vit_tokens,
        }

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """将 token 序列池化为一个全局特征 / Pool a token sequence into one
        global feature."""
        if self.pool_type == "mean":
            return tokens.mean(dim=1)

        raise RuntimeError(f"Unexpected pool_type={self.pool_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = True,
        return_attn_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """执行完整的 token-bridge 前向传播 / Run the full token-bridge
        forward pass."""
        cnn_dict = self.extract_cnn_tokens(x)
        vit_dict = self.extract_vit_tokens(x)

        cnn_tokens = cnn_dict["cnn_tokens"]
        vit_tokens = vit_dict["vit_tokens"]

        bridge_dict = self.bridge(
            cnn_tokens=cnn_tokens,
            vit_tokens=vit_tokens,
            return_gate=return_gate,
            return_attn_weights=return_attn_weights,
        )

        fused_cnn_tokens = bridge_dict["fused_cnn_tokens"]
        fused_vit_tokens = bridge_dict["fused_vit_tokens"]

        pooled_cnn_feature = self.pool_tokens(fused_cnn_tokens)
        pooled_vit_feature = self.pool_tokens(fused_vit_tokens)

        fusion_dict = self.branch_fusion(
            cnn_feature=pooled_cnn_feature,
            vit_feature=pooled_vit_feature,
            return_gate=return_gate,
        )
        fused_feature = fusion_dict["fused_feature"]
        logits = self.classifier(fused_feature)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "cnn_feature_map": cnn_dict["cnn_feature_map"],
            "cnn_tokens": cnn_tokens,
            "vit_patch_tokens": vit_dict["vit_patch_tokens"],
            "vit_tokens": vit_tokens,
            "fused_cnn_tokens": fused_cnn_tokens,
            "fused_vit_tokens": fused_vit_tokens,
            "pooled_cnn_feature": pooled_cnn_feature,
            "pooled_vit_feature": pooled_vit_feature,
            "fused_feature": fused_feature,
        }

        if "cnn_gate" in bridge_dict:
            outputs["cnn_token_gate"] = bridge_dict["cnn_gate"]
        if "vit_gate" in bridge_dict:
            outputs["vit_token_gate"] = bridge_dict["vit_gate"]
        if "branch_gate" in fusion_dict:
            outputs["branch_gate"] = fusion_dict["branch_gate"]
        if "cnn_attn_weights" in bridge_dict:
            outputs["cnn_attn_weights"] = bridge_dict["cnn_attn_weights"]
        if "vit_attn_weights" in bridge_dict:
            outputs["vit_attn_weights"] = bridge_dict["vit_attn_weights"]

        return outputs


def _demo_token_bridge_model() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    model = TokenBridgeModel(
        num_classes=37,
        resnet_name="resnet50",
        vit_name="vit_b_16",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="linear",
        bridge_dim=256,
        projector_hidden_dim=None,
        bridge_num_heads=8,
        token_gate_hidden_dim=256,
        branch_gate_hidden_dim=256,
        use_token_gate=True,
        use_cnn_pos_embed=True,
        use_vit_pos_embed=False,
        pool_type="mean",
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            x,
            return_gate=True,
            return_attn_weights=False,
        )

    print("==== TokenBridgeModel Forward Test ====")
    print("Input shape         :", x.shape)
    print("CNN feature map     :", outputs["cnn_feature_map"].shape)
    print("CNN tokens          :", outputs["cnn_tokens"].shape)
    print("ViT patch tokens    :", outputs["vit_patch_tokens"].shape)
    print("ViT tokens          :", outputs["vit_tokens"].shape)
    print("Fused CNN tokens    :", outputs["fused_cnn_tokens"].shape)
    print("Fused ViT tokens    :", outputs["fused_vit_tokens"].shape)
    print("Pooled CNN feature  :", outputs["pooled_cnn_feature"].shape)
    print("Pooled ViT feature  :", outputs["pooled_vit_feature"].shape)
    print("Final fused feature :", outputs["fused_feature"].shape)
    print("Logits              :", outputs["logits"].shape)

    if "cnn_token_gate" in outputs:
        print("CNN token gate      :", outputs["cnn_token_gate"].shape)
    if "vit_token_gate" in outputs:
        print("ViT token gate      :", outputs["vit_token_gate"].shape)
    if "branch_gate" in outputs:
        print("Branch gate         :", outputs["branch_gate"].shape)


if __name__ == "__main__":
    _demo_token_bridge_model()
