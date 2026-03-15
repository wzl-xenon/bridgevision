from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn

from src.models.backbones.resnet_backbone import ResNetBackbone
from src.models.backbones.vit_backbone import ViTBackbone
from src.models.fusions.concat_fusion import ConcatFusion
from src.models.fusions.gated_fusion import GatedFusion
from src.models.projectors.projector import Projector


class ClassificationHead(nn.Module):
    """
    Simple classification head / 简单分类头
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
        return self.head(x)


class DualEncoderModel(nn.Module):
    """
    BridgeVision 第一阶段模型

    支持三种模式：
    1. dual
       ResNet pooled_feature -> projector
       ViT cls_feature -> projector
       fusion -> classifier

    2. resnet_only
       ResNet pooled_feature -> classifier

    3. vit_only
       ViT cls_feature -> classifier
    """

    SUPPORTED_MODEL_MODES = {"dual", "resnet_only", "vit_only"}

    def __init__(
        self,
        num_classes: int,
        model_mode: Literal["dual", "resnet_only", "vit_only"] = "dual",
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

        if model_mode not in self.SUPPORTED_MODEL_MODES:
            raise ValueError(
                f"Unsupported model_mode={model_mode}. "
                f"Supported: {sorted(self.SUPPORTED_MODEL_MODES)}"
            )

        self.num_classes = num_classes
        self.model_mode = model_mode
        self.resnet_name = resnet_name
        self.vit_name = vit_name
        self.pretrained_backbones = pretrained_backbones
        self.freeze_backbones = freeze_backbones
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim

        self.resnet_backbone: ResNetBackbone | None = None
        self.vit_backbone: ViTBackbone | None = None
        self.resnet_projector: Projector | None = None
        self.vit_projector: Projector | None = None
        self.fusion: nn.Module | None = None

        # ---------------------------
        # 1. Build backbones
        # ---------------------------
        if self.model_mode in {"dual", "resnet_only"}:
            self.resnet_backbone = ResNetBackbone(
                model_name=resnet_name,
                pretrained=pretrained_backbones,
                freeze=freeze_backbones,
            )

        if self.model_mode in {"dual", "vit_only"}:
            self.vit_backbone = ViTBackbone(
                model_name=vit_name,
                pretrained=pretrained_backbones,
                freeze=freeze_backbones,
            )

        # ---------------------------
        # 2. Build heads / projectors / fusion by mode
        # ---------------------------
        if self.model_mode == "dual":
            assert self.resnet_backbone is not None
            assert self.vit_backbone is not None

            resnet_out_dim = self.resnet_backbone.out_channels
            vit_out_dim = self.vit_backbone.hidden_dim

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

            self.classifier = ClassificationHead(
                input_dim=fusion_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        elif self.model_mode == "resnet_only":
            assert self.resnet_backbone is not None

            resnet_out_dim = self.resnet_backbone.out_channels
            self.classifier = ClassificationHead(
                input_dim=resnet_out_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        elif self.model_mode == "vit_only":
            assert self.vit_backbone is not None

            vit_out_dim = self.vit_backbone.hidden_dim
            self.classifier = ClassificationHead(
                input_dim=vit_out_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        else:
            raise RuntimeError(f"Unexpected model_mode={self.model_mode}")

    def extract_branch_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取分支特征
        """
        outputs: Dict[str, torch.Tensor] = {}

        if self.resnet_backbone is not None:
            resnet_out = self.resnet_backbone(x)
            outputs["resnet_feature"] = resnet_out["pooled_feature"]

        if self.vit_backbone is not None:
            vit_out = self.vit_backbone(x)
            outputs["vit_feature"] = vit_out["cls_feature"]

        return outputs

    def align_features(
        self,
        resnet_feature: torch.Tensor,
        vit_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        仅 dual 模式需要 projector 对齐
        """
        if self.model_mode != "dual":
            raise RuntimeError("align_features() is only available when model_mode='dual'.")

        assert self.resnet_projector is not None
        assert self.vit_projector is not None

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
        仅 dual 模式需要融合
        """
        if self.model_mode != "dual":
            raise RuntimeError("fuse_features() is only available when model_mode='dual'.")

        assert self.fusion is not None

        if self.fusion_type == "concat":
            fused_feature = self.fusion(resnet_projected, vit_projected)
            return {
                "fused_feature": fused_feature,
            }

        if self.fusion_type == "gated":
            fused_feature, gate = self.fusion(
                resnet_projected,
                vit_projected,
                return_gate=True,
            )
            return {
                "fused_feature": fused_feature,
                "gate": gate,
            }

        raise RuntimeError(f"Unexpected fusion_type={self.fusion_type}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        feature_dict = self.extract_branch_features(x)

        if self.model_mode == "resnet_only":
            resnet_feature = feature_dict["resnet_feature"]
            logits = self.classifier(resnet_feature)
            return {
                "logits": logits,
                "resnet_feature": resnet_feature,
                "fused_feature": resnet_feature,
            }

        if self.model_mode == "vit_only":
            vit_feature = feature_dict["vit_feature"]
            logits = self.classifier(vit_feature)
            return {
                "logits": logits,
                "vit_feature": vit_feature,
                "fused_feature": vit_feature,
            }

        if self.model_mode == "dual":
            resnet_feature = feature_dict["resnet_feature"]
            vit_feature = feature_dict["vit_feature"]

            aligned_dict = self.align_features(resnet_feature, vit_feature)
            resnet_projected = aligned_dict["resnet_projected"]
            vit_projected = aligned_dict["vit_projected"]

            fused_dict = self.fuse_features(resnet_projected, vit_projected)
            fused_feature = fused_dict["fused_feature"]

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

        raise RuntimeError(f"Unexpected model_mode={self.model_mode}")


def _demo_forward_dual_concat() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        model_mode="dual",
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

    print("==== DualEncoderModel Forward Test (Dual + Concat) ====")
    print("Input shape            :", x.shape)
    print("ResNet feature shape   :", outputs["resnet_feature"].shape)
    print("ViT feature shape      :", outputs["vit_feature"].shape)
    print("ResNet projected shape :", outputs["resnet_projected"].shape)
    print("ViT projected shape    :", outputs["vit_projected"].shape)
    print("Fused feature shape    :", outputs["fused_feature"].shape)
    print("Logits shape           :", outputs["logits"].shape)


def _demo_forward_resnet_only() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        model_mode="resnet_only",
        resnet_name="resnet50",
        pretrained_backbones=False,
        freeze_backbones=False,
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== DualEncoderModel Forward Test (ResNet Only) ====")
    print("Input shape            :", x.shape)
    print("ResNet feature shape   :", outputs["resnet_feature"].shape)
    print("Logits shape           :", outputs["logits"].shape)


def _demo_forward_vit_only() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        model_mode="vit_only",
        vit_name="vit_b_16",
        pretrained_backbones=False,
        freeze_backbones=False,
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== DualEncoderModel Forward Test (ViT Only) ====")
    print("Input shape            :", x.shape)
    print("ViT feature shape      :", outputs["vit_feature"].shape)
    print("Logits shape           :", outputs["logits"].shape)


if __name__ == "__main__":
    _demo_forward_dual_concat()
    print()
    _demo_forward_resnet_only()
    print()
    _demo_forward_vit_only()