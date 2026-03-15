# src/models/dual_encoder_model.py

from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn

from src.models.backbones.resnet_backbone import ResNetBackbone
from src.models.backbones.vit_backbone import ViTBackbone
from src.models.projectors.projector import Projector
from src.models.fusions.concat_fusion import ConcatFusion
from src.models.fusions.gated_fusion import GatedFusion


class ClassificationHead(nn.Module):
    """
    Simple classification head / 简单分类头

    功能 / Features:
    1. 接收融合后的特征
       Take fused feature as input
    2. 输出分类 logits
       Output classification logits
    """

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
        """
        Args:
            x: [B, D]

        Returns:
            logits: [B, num_classes]
        """
        return self.head(x)


class DualEncoderModel(nn.Module):
    """
    Dual-encoder fusion model / 双编码器融合模型

    第一版设计 / Version-1 design:
    1. ResNet 提供全局 pooled feature
       ResNet provides global pooled feature
    2. ViT 提供 cls token feature
       ViT provides cls token feature
    3. 两路特征分别经过 projector 对齐到相同维度
       Two features are aligned to the same dimension by projectors
    4. 用 concat 或 gated fusion 做融合
       Use concat or gated fusion for feature fusion
    5. 最后接分类头
       Finally apply a classification head

    注意 / Notes:
    - 这是全局特征级 baseline
      This is a global-feature-level baseline
    - 还不是 token-level bridge 版本
      This is not the token-level bridge version yet
    """

    def __init__(
        self,
        num_classes: int,
        resnet_name: str = "resnet50",
        vit_name: str = "vit_b_16",
        pretrained_backbones: bool = False,
        freeze_backbones: bool = False,
        projector_type: Literal["linear", "mlp"] = "mlp",
        fusion_type: Literal["concat", "gated"] = "concat",
        fusion_dim: int = 512,
        projector_hidden_dim: int | None = None,
        fusion_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.resnet_name = resnet_name
        self.vit_name = vit_name
        self.pretrained_backbones = pretrained_backbones
        self.freeze_backbones = freeze_backbones
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim

        # ---------------------------
        # 1. Build backbones / 构建两个 backbone
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

        # ResNet 和 ViT 输出维度
        # Output dimensions of ResNet and ViT
        resnet_out_dim = self.resnet_backbone.out_channels
        vit_out_dim = self.vit_backbone.hidden_dim

        # ---------------------------
        # 2. Build projectors / 构建投影器
        # ---------------------------
        self.resnet_projector = Projector(
            input_dim=resnet_out_dim,
            output_dim=fusion_dim,
            projector_type=projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        self.vit_projector = Projector(
            input_dim=vit_out_dim,
            output_dim=fusion_dim,
            projector_type=projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        # ---------------------------
        # 3. Build fusion module / 构建融合模块
        # ---------------------------
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                feature_dim=fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                feature_dim=fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
                refine_output=True,
            )
        else:
            raise ValueError(
                f"Unsupported fusion_type={fusion_type}. "
                f"Supported: ['concat', 'gated']"
            )

        # ---------------------------
        # 4. Build classifier / 构建分类头
        # ---------------------------
        self.classifier = ClassificationHead(
            input_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def extract_branch_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取两个分支的全局特征 / Extract global features from two branches

        Args:
            x: [B, 3, H, W]

        Returns:
            {
                "resnet_feature": [B, C_r],
                "vit_feature": [B, C_v],
            }
        """
        resnet_out = self.resnet_backbone(x)
        vit_out = self.vit_backbone(x)

        # 第一版只用全局特征
        # In version 1, only global features are used
        resnet_feature = resnet_out["pooled_feature"]  # [B, 2048] for resnet50
        vit_feature = vit_out["cls_feature"]           # [B, 768] for vit_b_16

        return {
            "resnet_feature": resnet_feature,
            "vit_feature": vit_feature,
        }

    def align_features(
        self,
        resnet_feature: torch.Tensor,
        vit_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        用 projector 对齐两个分支特征 / Align two branch features by projectors

        Args:
            resnet_feature: [B, C_r]
            vit_feature: [B, C_v]

        Returns:
            {
                "resnet_projected": [B, D],
                "vit_projected": [B, D],
            }
        """
        resnet_projected = self.resnet_projector(resnet_feature)
        vit_projected = self.vit_projector(vit_feature)

        return {
            "resnet_projected": resnet_projected,
            "vit_projected": vit_projected,
        }

    def fuse_features(
        self,
        resnet_projected: torch.Tensor,
        vit_projected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        融合两个对齐后的特征 / Fuse two aligned features

        Args:
            resnet_projected: [B, D]
            vit_projected: [B, D]

        Returns:
            {
                "fused_feature": [B, D],
                "gate": [B, D] (only when fusion_type='gated')
            }
        """
        if self.fusion_type == "concat":
            fused_feature = self.fusion(resnet_projected, vit_projected)
            return {
                "fused_feature": fused_feature,
            }

        elif self.fusion_type == "gated":
            fused_feature, gate = self.fusion(
                resnet_projected,
                vit_projected,
                return_gate=True,
            )
            return {
                "fused_feature": fused_feature,
                "gate": gate,
            }

        else:
            raise RuntimeError(f"Unexpected fusion_type={self.fusion_type}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            x: [B, 3, H, W]

        Returns:
            A dictionary containing:
            {
                "logits": [B, num_classes],
                "resnet_feature": [B, C_r],
                "vit_feature": [B, C_v],
                "resnet_projected": [B, D],
                "vit_projected": [B, D],
                "fused_feature": [B, D],
                "gate": [B, D]  # only for gated fusion
            }

        说明 / Notes:
        - 返回中间结果，方便后面做调试和消融
          Intermediate outputs are returned for debugging and ablation
        """
        # 1. Extract features / 提取两个分支的特征
        feature_dict = self.extract_branch_features(x)
        resnet_feature = feature_dict["resnet_feature"]
        vit_feature = feature_dict["vit_feature"]

        # 2. Align features / 对齐特征维度
        aligned_dict = self.align_features(resnet_feature, vit_feature)
        resnet_projected = aligned_dict["resnet_projected"]
        vit_projected = aligned_dict["vit_projected"]

        # 3. Fuse features / 融合特征
        fused_dict = self.fuse_features(resnet_projected, vit_projected)
        fused_feature = fused_dict["fused_feature"]

        # 4. Classification / 分类输出
        logits = self.classifier(fused_feature)

        outputs = {
            "logits": logits,
            "resnet_feature": resnet_feature,
            "vit_feature": vit_feature,
            "resnet_projected": resnet_projected,
            "vit_projected": vit_projected,
            "fused_feature": fused_feature,
        }

        if "gate" in fused_dict:
            outputs["gate"] = fused_dict["gate"]

        return outputs


def _demo_forward_concat() -> None:
    """
    concat fusion 的简单前向测试
    Simple forward test for concat fusion
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        resnet_name="resnet50",
        vit_name="vit_b_16",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="concat",
        fusion_dim=512,
        projector_hidden_dim=1024,
        fusion_hidden_dim=512,
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== DualEncoderModel Forward Test (Concat) ====")
    print("Input shape            :", x.shape)
    print("ResNet feature shape   :", outputs["resnet_feature"].shape)
    print("ViT feature shape      :", outputs["vit_feature"].shape)
    print("ResNet projected shape :", outputs["resnet_projected"].shape)
    print("ViT projected shape    :", outputs["vit_projected"].shape)
    print("Fused feature shape    :", outputs["fused_feature"].shape)
    print("Logits shape           :", outputs["logits"].shape)


def _demo_forward_gated() -> None:
    """
    gated fusion 的简单前向测试
    Simple forward test for gated fusion
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        resnet_name="resnet50",
        vit_name="vit_b_16",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="gated",
        fusion_dim=512,
        projector_hidden_dim=1024,
        fusion_hidden_dim=512,
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== DualEncoderModel Forward Test (Gated) ====")
    print("Input shape            :", x.shape)
    print("ResNet feature shape   :", outputs["resnet_feature"].shape)
    print("ViT feature shape      :", outputs["vit_feature"].shape)
    print("ResNet projected shape :", outputs["resnet_projected"].shape)
    print("ViT projected shape    :", outputs["vit_projected"].shape)
    print("Fused feature shape    :", outputs["fused_feature"].shape)
    print("Gate shape             :", outputs["gate"].shape)
    print("Logits shape           :", outputs["logits"].shape)
    print("Gate min / max         :", outputs["gate"].min().item(), outputs["gate"].max().item())


if __name__ == "__main__":
    _demo_forward_concat()
    print()
    _demo_forward_gated()