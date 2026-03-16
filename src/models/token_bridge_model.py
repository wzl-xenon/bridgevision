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
    """
    简单分类头 / Simple classification head

    功能 / Features:
    1. 接收融合后的全局特征
       Take the fused global feature as input
    2. 输出分类 logits
       Output classification logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim:
                输入特征维度
                Input feature dimension

            num_classes:
                类别数
                Number of classes

            dropout:
                dropout 概率
                Dropout probability
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                输入特征，形状为 [B, D]
                Input feature of shape [B, D]

        Returns:
            logits:
                分类输出，形状为 [B, num_classes]
                Classification logits of shape [B, num_classes]
        """
        return self.head(x)


class BranchGateFusion(nn.Module):
    """
    分支级门控融合 / Branch-level gated fusion

    设计目的 / Design purpose:
    - 在 token-level 双向 bridge 之后，
      再对 CNN 分支全局特征和 ViT 分支全局特征做一次样本级/维度级门控融合
      After bidirectional token-level bridge,
      perform one more sample-/dimension-level gated fusion
      on the pooled CNN feature and pooled ViT feature

    输入 / Inputs:
    - cnn_feature: [B, D]
    - vit_feature: [B, D]

    输出 / Outputs:
    - fused_feature: [B, D]
    - branch_gate: [B, D] (optional)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            feature_dim:
                输入与输出特征维度
                Input/output feature dimension

            hidden_dim:
                门控 MLP 的隐藏层维度
                Hidden dimension of the gate MLP

            dropout:
                dropout 概率
                Dropout probability
        """
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
        """
        Args:
            cnn_feature:
                CNN 分支的全局特征，形状为 [B, D]
                Global CNN feature of shape [B, D]

            vit_feature:
                ViT 分支的全局特征，形状为 [B, D]
                Global ViT feature of shape [B, D]

            return_gate:
                是否返回分支级门控值
                Whether to return the branch-level gate

        Returns:
            outputs:
                {
                    "fused_feature": [B, D],
                    "branch_gate": [B, D]  # optional
                }
        """
        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([cnn_feature, vit_feature], dim=-1))
        )  # [B, D]

        # 用 gate 控制更偏向 CNN 还是更偏向 ViT
        # Use gate to control whether the fusion leans more toward CNN or ViT
        fused_feature = gate * cnn_feature + (1.0 - gate) * vit_feature
        fused_feature = self.output_norm(fused_feature)

        outputs = {"fused_feature": fused_feature}
        if return_gate:
            outputs["branch_gate"] = gate
        return outputs


class TokenBridgeModel(nn.Module):
    """
    Token Bridge V1 / Token Bridge 第一版模型

    研究目标 / Research goal:
    - 不再只做全局 pooled feature 级融合
      No longer fuse only global pooled features
    - 而是把 CNN 的最后一层 feature map 转成 spatial tokens，
      与 ViT 的 patch tokens 在 token 粒度上做双向交互
      Instead, convert the last CNN feature map into spatial tokens
      and let them interact bidirectionally with ViT patch tokens

    整体流程 / Overall pipeline:

    CNN branch / CNN 分支:
        image
        -> CNN backbone
        -> feature_map [B, C, H, W]
        -> spatial tokens [B, N_c, C]
        -> projector [B, N_c, d]
        -> optional 2D positional encoding

    ViT branch / ViT 分支:
        image
        -> ViT backbone
        -> patch_tokens [B, N_v, D]
        -> projector [B, N_v, d]
        -> optional 1D positional encoding

    Bridge / 桥接模块:
        bidirectional cross-attention + token-wise gates
        双向交叉注意力 + token 级门控

    Aggregation / 聚合:
        pool fused CNN tokens and fused ViT tokens separately
        对融合后的两路 token 分别做池化

    Final fusion / 最终融合:
        branch-level gated fusion
        分支级门控融合

    Head / 任务头:
        classifier
        分类头
    """

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
        """
        Args:
            num_classes:
                类别数
                Number of classes

            resnet_name:
                CNN backbone 名称
                CNN backbone name

            vit_name:
                ViT backbone 名称
                ViT backbone name

            pretrained_backbones:
                是否使用预训练 backbone
                Whether to use pretrained backbones

            freeze_backbones:
                是否冻结 backbone 参数
                Whether to freeze backbone parameters

            projector_type:
                projector 类型，可选 linear / mlp
                Projector type, either linear or mlp

            bridge_dim:
                token bridge 使用的公共特征维度
                Shared feature dimension used in the token bridge

            projector_hidden_dim:
                projector 的隐藏层维度
                Hidden dimension of the projector

            bridge_num_heads:
                cross-attention 的 head 数
                Number of attention heads in cross-attention

            token_gate_hidden_dim:
                token 级门控 MLP 的隐藏层维度
                Hidden dimension of the token-level gate MLP

            branch_gate_hidden_dim:
                分支级门控 MLP 的隐藏层维度
                Hidden dimension of the branch-level gate MLP

            use_token_gate:
                是否启用 token 级门控
                Whether to enable token-level gates

            use_cnn_pos_embed:
                是否给 CNN spatial tokens 加外置 2D 位置编码
                Whether to add external 2D positional encoding to CNN spatial tokens

            use_vit_pos_embed:
                是否给 ViT patch tokens 加外置 1D 位置编码
                Whether to add external 1D positional encoding to ViT patch tokens

            pool_type:
                token 池化方式，目前只支持 mean
                Token pooling type, currently only mean is supported

            dropout:
                dropout 概率
                Dropout probability
        """
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

        # ---------------------------
        # 1. Backbones / 主干网络
        # ---------------------------
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

        # ---------------------------
        # 2. Tokenizer + Projectors / token 化与投影
        # ---------------------------
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

        # ---------------------------
        # 3. Positional encodings / 位置编码
        # CNN token 默认建议加显式 2D 位置编码
        # CNN tokens are recommended to use explicit 2D positional encoding
        #
        # ViT token 的外置位置编码默认可关，因为 backbone 内部一般已有位置编码
        # External ViT positional encoding is optional because
        # the backbone usually already contains positional encoding
        # ---------------------------
        self.cnn_pos_embed = (
            SinCos2DPositionalEncoding(dropout=dropout)
            if use_cnn_pos_embed else None
        )

        self.vit_pos_embed = (
            SinCos1DPositionalEncoding(dropout=dropout)
            if use_vit_pos_embed else None
        )

        # ---------------------------
        # 4. Bidirectional token bridge / 双向 token bridge
        # ---------------------------
        self.bridge = TokenBridgeFusion(
            feature_dim=bridge_dim,
            num_heads=bridge_num_heads,
            dropout=dropout,
            gate_hidden_dim=token_gate_hidden_dim,
            use_gate=use_token_gate,
            use_layernorm=True,
        )

        # ---------------------------
        # 5. Final branch-level gate / 最终分支级门控
        # ---------------------------
        self.branch_fusion = BranchGateFusion(
            feature_dim=bridge_dim,
            hidden_dim=branch_gate_hidden_dim,
            dropout=dropout,
        )

        # ---------------------------
        # 6. Classification head / 分类头
        # ---------------------------
        self.classifier = ClassificationHead(
            input_dim=bridge_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def extract_cnn_tokens(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor | tuple[int, int]]:
        """
        提取 CNN 分支 tokens / Extract CNN branch tokens

        Args:
            x:
                输入图像，形状为 [B, 3, H, W]
                Input image of shape [B, 3, H, W]

        Returns:
            outputs:
                {
                    "cnn_feature_map": [B, C, Hf, Wf],
                    "cnn_tokens": [B, N_c, d],
                    "cnn_hw": (Hf, Wf),
                }
        """
        resnet_out = self.resnet_backbone(x)
        feature_map = resnet_out["feature_map"]  # [B, C, H, W]

        # 把 CNN feature map 变成 spatial token 序列
        # Convert CNN feature map into a spatial token sequence
        cnn_tokens, hw = self.spatial_tokenizer(
            feature_map,
            return_hw=True,
        )  # [B, N_c, C], (H, W)

        # 把 CNN token 维度投影到 bridge 公共空间
        # Project CNN token dimension to the shared bridge space
        cnn_tokens = self.cnn_projector(cnn_tokens)  # [B, N_c, d]

        # 可选的 CNN 外置 2D 位置编码
        # Optional external 2D positional encoding for CNN tokens
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
        """
        提取 ViT 分支 tokens / Extract ViT branch tokens

        Args:
            x:
                输入图像，形状为 [B, 3, H, W]
                Input image of shape [B, 3, H, W]

        Returns:
            outputs:
                {
                    "vit_patch_tokens": [B, N_v, D],
                    "vit_tokens": [B, N_v, d],
                }
        """
        vit_out = self.vit_backbone(x)
        patch_tokens = vit_out["patch_tokens"]  # [B, N_v, D]

        # 把 ViT patch token 维度投影到 bridge 公共空间
        # Project ViT patch token dimension to the shared bridge space
        vit_tokens = self.vit_projector(patch_tokens)  # [B, N_v, d]

        # ViT 外置位置编码默认可关
        # External ViT positional encoding is optional and can be disabled
        if self.vit_pos_embed is not None:
            vit_tokens = self.vit_pos_embed(vit_tokens)

        return {
            "vit_patch_tokens": patch_tokens,
            "vit_tokens": vit_tokens,
        }

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        对 token 序列做池化 / Pool a token sequence into a global feature

        Args:
            tokens:
                token 序列，形状为 [B, N, D]
                Token sequence of shape [B, N, D]

        Returns:
            pooled_feature:
                池化后的全局特征，形状为 [B, D]
                Pooled global feature of shape [B, D]
        """
        if self.pool_type == "mean":
            return tokens.mean(dim=1)

        raise RuntimeError(f"Unexpected pool_type={self.pool_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = True,
        return_attn_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            x:
                输入图像，形状为 [B, 3, H, W]
                Input image of shape [B, 3, H, W]

            return_gate:
                是否返回 token 级门控和分支级门控
                Whether to return token-level and branch-level gates

            return_attn_weights:
                是否返回双向 cross-attention 的注意力权重
                Whether to return attention weights from bidirectional cross-attention

        Returns:
            outputs:
                一个包含 logits 和中间特征的字典
                A dictionary containing logits and intermediate features
        """
        # ---------------------------
        # 1. Extract branch tokens / 提取两路 token
        # ---------------------------
        cnn_dict = self.extract_cnn_tokens(x)
        vit_dict = self.extract_vit_tokens(x)

        cnn_tokens = cnn_dict["cnn_tokens"]
        vit_tokens = vit_dict["vit_tokens"]

        # ---------------------------
        # 2. Bidirectional bridge / 双向 bridge
        # 在 token 粒度上做双向交叉注意力交互
        # Perform bidirectional cross-attention interaction at token level
        # ---------------------------
        bridge_dict = self.bridge(
            cnn_tokens=cnn_tokens,
            vit_tokens=vit_tokens,
            return_gate=return_gate,
            return_attn_weights=return_attn_weights,
        )

        fused_cnn_tokens = bridge_dict["fused_cnn_tokens"]
        fused_vit_tokens = bridge_dict["fused_vit_tokens"]

        # ---------------------------
        # 3. Pool two branches separately / 两路分别池化
        # ---------------------------
        pooled_cnn_feature = self.pool_tokens(fused_cnn_tokens)
        pooled_vit_feature = self.pool_tokens(fused_vit_tokens)

        # ---------------------------
        # 4. Branch-level gated fusion / 分支级门控融合
        # 在全局特征层面再做一次可学习融合
        # Perform another learnable fusion at the global-feature level
        # ---------------------------
        fusion_dict = self.branch_fusion(
            cnn_feature=pooled_cnn_feature,
            vit_feature=pooled_vit_feature,
            return_gate=return_gate,
        )

        fused_feature = fusion_dict["fused_feature"]

        # ---------------------------
        # 5. Classification / 分类输出
        # ---------------------------
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

        # token 级 gate
        # token-level gates
        if "cnn_gate" in bridge_dict:
            outputs["cnn_token_gate"] = bridge_dict["cnn_gate"]

        if "vit_gate" in bridge_dict:
            outputs["vit_token_gate"] = bridge_dict["vit_gate"]

        # 分支级 gate
        # branch-level gate
        if "branch_gate" in fusion_dict:
            outputs["branch_gate"] = fusion_dict["branch_gate"]

        # cross-attention 权重
        # cross-attention weights
        if "cnn_attn_weights" in bridge_dict:
            outputs["cnn_attn_weights"] = bridge_dict["cnn_attn_weights"]

        if "vit_attn_weights" in bridge_dict:
            outputs["vit_attn_weights"] = bridge_dict["vit_attn_weights"]

        return outputs


def _demo_token_bridge_model() -> None:
    """
    简单前向测试 / Simple forward test
    """
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
    print("Input shape          :", x.shape)
    print("CNN feature map      :", outputs["cnn_feature_map"].shape)
    print("CNN tokens           :", outputs["cnn_tokens"].shape)
    print("ViT patch tokens     :", outputs["vit_patch_tokens"].shape)
    print("ViT tokens           :", outputs["vit_tokens"].shape)
    print("Fused CNN tokens     :", outputs["fused_cnn_tokens"].shape)
    print("Fused ViT tokens     :", outputs["fused_vit_tokens"].shape)
    print("Pooled CNN feature   :", outputs["pooled_cnn_feature"].shape)
    print("Pooled ViT feature   :", outputs["pooled_vit_feature"].shape)
    print("Final fused feature  :", outputs["fused_feature"].shape)
    print("Logits               :", outputs["logits"].shape)

    if "cnn_token_gate" in outputs:
        print("CNN token gate       :", outputs["cnn_token_gate"].shape)

    if "vit_token_gate" in outputs:
        print("ViT token gate       :", outputs["vit_token_gate"].shape)

    if "branch_gate" in outputs:
        print("Branch gate          :", outputs["branch_gate"].shape)


if __name__ == "__main__":
    _demo_token_bridge_model()