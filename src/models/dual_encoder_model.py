from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones.resnet_backbone import ResNetBackbone
from src.models.backbones.vit_backbone import ViTBackbone
from src.models.fusions.concat_fusion import ConcatFusion
from src.models.fusions.gated_fusion import GatedFusion, TokenDimGatedFusion
from src.models.fusions.token_bridge_fusion import TokenBridgeFusion
from src.models.projectors.projector import Projector
from src.models.tokenizers.fixed_token_resampler import FixedTokenResampler


class ClassificationHead(nn.Module):
    """将特征向量映射为分类 logits / Map a feature vector to classification
    logits."""

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
    """BridgeVision 第一阶段双编码器模型 / BridgeVision stage-1 dual-encoder
    model."""

    SUPPORTED_MODEL_MODES = {"dual", "resnet_only", "vit_only"}
    SUPPORTED_FUSION_TYPES = {"concat", "gated", "token_bridge", "matched_token_gated"}
    SUPPORTED_SUMMARY_FUSION_TYPES = {"concat", "gated"}

    def __init__(
        self,
        num_classes: int,
        model_mode: Literal["dual", "resnet_only", "vit_only"] = "dual",
        resnet_name: str = "resnet50",
        vit_name: str = "vit_b_16",
        pretrained_backbones: bool = False,
        freeze_backbones: bool = False,
        projector_type: Literal["linear", "mlp"] = "mlp",
        fusion_type: Literal["concat", "gated", "token_bridge", "matched_token_gated"] = "concat",
        fusion_dim: int = 512,
        projector_hidden_dim: int | None = None,
        fusion_hidden_dim: int | None = None,
        dropout: float = 0.1,
        token_num_heads: int = 8,
        token_gate_hidden_dim: int | None = None,
        token_ffn_hidden_dim: int | None = None,
        token_use_gate: bool = True,
        num_bridge_layers: int = 1,
        matched_token_count: int = 16,
        summary_fusion_type: Literal["concat", "gated"] = "gated",
        use_cnn_pos_embed: bool = True,
        cnn_pos_embed_base_size: int = 7,
    ) -> None:
        super().__init__()

        if model_mode not in self.SUPPORTED_MODEL_MODES:
            raise ValueError(
                f"Unsupported model_mode={model_mode}. "
                f"Supported: {sorted(self.SUPPORTED_MODEL_MODES)}"
            )

        if fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type={fusion_type}. "
                f"Supported: {sorted(self.SUPPORTED_FUSION_TYPES)}"
            )

        if summary_fusion_type not in self.SUPPORTED_SUMMARY_FUSION_TYPES:
            raise ValueError(
                f"Unsupported summary_fusion_type={summary_fusion_type}. "
                f"Supported: {sorted(self.SUPPORTED_SUMMARY_FUSION_TYPES)}"
            )

        if num_bridge_layers < 1:
            raise ValueError("num_bridge_layers must be >= 1.")

        if matched_token_count < 1:
            raise ValueError("matched_token_count must be >= 1.")

        if cnn_pos_embed_base_size < 1:
            raise ValueError("cnn_pos_embed_base_size must be >= 1.")

        self.num_classes = num_classes
        self.model_mode = model_mode
        self.resnet_name = resnet_name
        self.vit_name = vit_name
        self.pretrained_backbones = pretrained_backbones
        self.freeze_backbones = freeze_backbones
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        self.token_num_heads = token_num_heads
        self.token_gate_hidden_dim = token_gate_hidden_dim
        self.token_ffn_hidden_dim = token_ffn_hidden_dim
        self.token_use_gate = token_use_gate
        self.num_bridge_layers = num_bridge_layers
        self.matched_token_count = matched_token_count
        self.summary_fusion_type = summary_fusion_type
        self.use_cnn_pos_embed = use_cnn_pos_embed
        self.cnn_pos_embed_base_size = cnn_pos_embed_base_size

        self.resnet_backbone: ResNetBackbone | None = None
        self.vit_backbone: ViTBackbone | None = None
        self.resnet_projector: Projector | None = None
        self.vit_projector: Projector | None = None
        self.fusion: nn.Module | None = None
        self.resnet_token_projector: Projector | None = None
        self.vit_token_projector: Projector | None = None
        self.token_fusion_blocks: nn.ModuleList | None = None
        self.summary_fusion: nn.Module | None = None
        self.token_resampler: FixedTokenResampler | None = None
        self.token_dim_fusion: TokenDimGatedFusion | None = None

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

        if self.model_mode == "dual":
            assert self.resnet_backbone is not None
            assert self.vit_backbone is not None

            resnet_out_dim = self.resnet_backbone.out_channels
            vit_out_dim = self.vit_backbone.hidden_dim

            if fusion_type in {"concat", "gated"}:
                self._build_global_fusion_modules(
                    resnet_out_dim=resnet_out_dim,
                    vit_out_dim=vit_out_dim,
                    projector_hidden_dim=projector_hidden_dim,
                    fusion_hidden_dim=fusion_hidden_dim,
                    dropout=dropout,
                )
            elif fusion_type == "token_bridge":
                self._build_token_bridge_modules(
                    resnet_out_dim=resnet_out_dim,
                    vit_out_dim=vit_out_dim,
                    projector_hidden_dim=projector_hidden_dim,
                    fusion_hidden_dim=fusion_hidden_dim,
                    dropout=dropout,
                )
            else:
                self._build_matched_token_modules(
                    resnet_out_dim=resnet_out_dim,
                    vit_out_dim=vit_out_dim,
                    projector_hidden_dim=projector_hidden_dim,
                    fusion_hidden_dim=fusion_hidden_dim,
                    dropout=dropout,
                )

        elif self.model_mode == "resnet_only":
            assert self.resnet_backbone is not None
            self.classifier = ClassificationHead(
                input_dim=self.resnet_backbone.out_channels,
                num_classes=num_classes,
                dropout=dropout,
            )

        elif self.model_mode == "vit_only":
            assert self.vit_backbone is not None
            self.classifier = ClassificationHead(
                input_dim=self.vit_backbone.hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        else:
            raise RuntimeError(f"Unexpected model_mode={self.model_mode}")

    def _build_global_fusion_modules(
        self,
        resnet_out_dim: int,
        vit_out_dim: int,
        projector_hidden_dim: int | None,
        fusion_hidden_dim: int | None,
        dropout: float,
    ) -> None:
        """构建旧版全局特征融合模块 / Build the modules used by legacy
        global-feature fusion."""
        self.resnet_projector = Projector(
            input_dim=resnet_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )
        self.vit_projector = Projector(
            input_dim=vit_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        if self.fusion_type == "concat":
            self.fusion = ConcatFusion(
                feature_dim=self.fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
            )
        elif self.fusion_type == "gated":
            self.fusion = GatedFusion(
                feature_dim=self.fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
                refine_output=True,
            )
        else:
            raise RuntimeError(f"Unexpected global fusion_type={self.fusion_type}")

        self.classifier = ClassificationHead(
            input_dim=self.fusion_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def _build_token_bridge_modules(
        self,
        resnet_out_dim: int,
        vit_out_dim: int,
        projector_hidden_dim: int | None,
        fusion_hidden_dim: int | None,
        dropout: float,
    ) -> None:
        """构建 token-bridge 融合模块 / Build the modules used by token-bridge
        fusion."""
        self.resnet_token_projector = Projector(
            input_dim=resnet_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )
        self.vit_token_projector = Projector(
            input_dim=vit_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        self.token_fusion_blocks = nn.ModuleList(
            [
                TokenBridgeFusion(
                    feature_dim=self.fusion_dim,
                    num_heads=self.token_num_heads,
                    dropout=dropout,
                    gate_hidden_dim=self.token_gate_hidden_dim,
                    ffn_hidden_dim=self.token_ffn_hidden_dim,
                    use_gate=self.token_use_gate,
                    use_layernorm=True,
                )
                for _ in range(self.num_bridge_layers)
            ]
        )

        if self.summary_fusion_type == "concat":
            self.summary_fusion = ConcatFusion(
                feature_dim=self.fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
            )
        else:
            self.summary_fusion = GatedFusion(
                feature_dim=self.fusion_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
                refine_output=True,
            )

        self.cnn_global_pos_embed = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        self.cnn_spatial_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.fusion_dim,
                self.cnn_pos_embed_base_size,
                self.cnn_pos_embed_base_size,
            )
        )

        nn.init.trunc_normal_(self.cnn_global_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cnn_spatial_pos_embed, std=0.02)

        self.classifier = ClassificationHead(
            input_dim=self.fusion_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def _build_matched_token_modules(
        self,
        resnet_out_dim: int,
        vit_out_dim: int,
        projector_hidden_dim: int | None,
        fusion_hidden_dim: int | None,
        dropout: float,
    ) -> None:
        """构建 token-bridge 与固定长度 token 融合模块 / Build token-bridge
        modules followed by fixed-length token fusion."""
        self.resnet_token_projector = Projector(
            input_dim=resnet_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )
        self.vit_token_projector = Projector(
            input_dim=vit_out_dim,
            output_dim=self.fusion_dim,
            projector_type=self.projector_type,
            hidden_dim=projector_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
        )

        self.token_fusion_blocks = nn.ModuleList(
            [
                TokenBridgeFusion(
                    feature_dim=self.fusion_dim,
                    num_heads=self.token_num_heads,
                    dropout=dropout,
                    gate_hidden_dim=self.token_gate_hidden_dim,
                    ffn_hidden_dim=self.token_ffn_hidden_dim,
                    use_gate=self.token_use_gate,
                    use_layernorm=True,
                )
                for _ in range(self.num_bridge_layers)
            ]
        )

        self.token_resampler = FixedTokenResampler(target_tokens=self.matched_token_count)
        self.token_dim_fusion = TokenDimGatedFusion(
            feature_dim=self.fusion_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            use_layernorm=True,
            refine_output=True,
        )

        self.cnn_global_pos_embed = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
        self.cnn_spatial_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.fusion_dim,
                self.cnn_pos_embed_base_size,
                self.cnn_pos_embed_base_size,
            )
        )

        nn.init.trunc_normal_(self.cnn_global_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cnn_spatial_pos_embed, std=0.02)

        self.classifier = ClassificationHead(
            input_dim=self.fusion_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def extract_branch_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取各前向路径共享的 backbone 特征 / Extract the backbone features
        that are shared across forward paths."""
        outputs: Dict[str, torch.Tensor] = {}

        if self.resnet_backbone is not None:
            resnet_out = self.resnet_backbone(x)
            outputs["resnet_feature"] = resnet_out["pooled_feature"]
            outputs["resnet_feature_map"] = resnet_out["feature_map"]

        if self.vit_backbone is not None:
            vit_out = self.vit_backbone(x)
            outputs["vit_feature"] = vit_out["cls_feature"]
            outputs["vit_patch_tokens"] = vit_out["patch_tokens"]
            outputs["vit_tokens"] = vit_out["all_tokens"]

        return outputs

    def align_features(
        self,
        resnet_feature: torch.Tensor,
        vit_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """将全局分支特征投影到共享融合空间 / Project global branch features
        into the shared fusion space."""
        if self.model_mode != "dual" or self.fusion_type not in {"concat", "gated"}:
            raise RuntimeError(
                "align_features() is only available when model_mode='dual' "
                "and fusion_type is 'concat' or 'gated'."
            )

        assert self.resnet_projector is not None
        assert self.vit_projector is not None

        return {
            "resnet_projected": self.resnet_projector(resnet_feature),
            "vit_projected": self.vit_projector(vit_feature),
        }

    def fuse_features(
        self,
        resnet_projected: torch.Tensor,
        vit_projected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """对齐并融合旧版双分支的全局特征 / Fuse aligned global features for
        the legacy dual-branch path."""
        if self.model_mode != "dual" or self.fusion_type not in {"concat", "gated"}:
            raise RuntimeError(
                "fuse_features() is only available when model_mode='dual' "
                "and fusion_type is 'concat' or 'gated'."
            )

        assert self.fusion is not None

        if self.fusion_type == "concat":
            fused_feature = self.fusion(resnet_projected, vit_projected)
            return {"fused_feature": fused_feature}

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

    def _get_cnn_spatial_pos_embed(self, height: int, width: int) -> torch.Tensor:
        """将学习到的 CNN 空间位置网格缩放到当前特征图尺寸 / Resize the
        learned CNN spatial position grid to the current feature map size."""
        assert hasattr(self, "cnn_spatial_pos_embed")

        pos_embed_2d = F.interpolate(
            self.cnn_spatial_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return pos_embed_2d.flatten(2).transpose(1, 2)

    def _build_cnn_tokens(
        self,
        feature_map: torch.Tensor,
        pooled_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建一个 CNN 全局 token 和展平后的空间 token / Build one global
        CNN token plus flattened spatial tokens."""
        assert self.resnet_token_projector is not None

        _, _, height, width = feature_map.shape

        cnn_spatial_tokens = feature_map.flatten(2).transpose(1, 2)
        cnn_global_token = pooled_feature.unsqueeze(1)

        cnn_spatial_tokens = self.resnet_token_projector(cnn_spatial_tokens)
        cnn_global_token = self.resnet_token_projector(cnn_global_token)

        if self.use_cnn_pos_embed:
            cnn_global_token = cnn_global_token + self.cnn_global_pos_embed
            cnn_spatial_tokens = cnn_spatial_tokens + self._get_cnn_spatial_pos_embed(height, width)

        cnn_tokens = torch.cat([cnn_global_token, cnn_spatial_tokens], dim=1)
        return cnn_tokens, cnn_global_token, cnn_spatial_tokens

    def _run_global_dual_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行旧版全局特征双分支前向 / Run the legacy dual-branch forward
        pass with global feature fusion."""
        feature_dict = self.extract_branch_features(x)
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
            outputs["fusion_gate"] = fused_dict["gate"]

        return outputs

    def _run_token_bridge_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行 token-bridge 前向，并读取融合后的全局 token / Run the
        token-bridge forward path and read out fused global tokens."""
        assert self.resnet_backbone is not None
        assert self.vit_backbone is not None
        assert self.resnet_token_projector is not None
        assert self.vit_token_projector is not None
        assert self.token_fusion_blocks is not None
        assert self.summary_fusion is not None

        resnet_out = self.resnet_backbone(x)
        vit_out = self.vit_backbone(x)

        resnet_feature_map = resnet_out["feature_map"]
        resnet_pooled_feature = resnet_out["pooled_feature"]
        vit_tokens = vit_out["all_tokens"]

        cnn_tokens, cnn_global_token, cnn_spatial_tokens = self._build_cnn_tokens(
            feature_map=resnet_feature_map,
            pooled_feature=resnet_pooled_feature,
        )
        vit_tokens_projected = self.vit_token_projector(vit_tokens)

        fused_cnn_tokens = cnn_tokens
        fused_vit_tokens = vit_tokens_projected

        last_bridge_outputs: dict[str, torch.Tensor] = {}
        for bridge_block in self.token_fusion_blocks:
            last_bridge_outputs = bridge_block(
                cnn_tokens=fused_cnn_tokens,
                vit_tokens=fused_vit_tokens,
                return_gate=True,
                return_attn_weights=False,
            )
            fused_cnn_tokens = last_bridge_outputs["fused_cnn_tokens"]
            fused_vit_tokens = last_bridge_outputs["fused_vit_tokens"]

        cnn_readout = fused_cnn_tokens[:, 0]
        vit_readout = fused_vit_tokens[:, 0]

        if self.summary_fusion_type == "gated":
            fused_feature, summary_gate = self.summary_fusion(
                cnn_readout,
                vit_readout,
                return_gate=True,
            )
        else:
            fused_feature = self.summary_fusion(cnn_readout, vit_readout)
            summary_gate = None

        logits = self.classifier(fused_feature)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "resnet_feature_map": resnet_feature_map,
            "resnet_pooled_feature": resnet_pooled_feature,
            "vit_tokens": vit_tokens,
            "cnn_global_token": cnn_global_token,
            "cnn_spatial_tokens": cnn_spatial_tokens,
            "cnn_tokens": cnn_tokens,
            "vit_tokens_projected": vit_tokens_projected,
            "fused_cnn_tokens": fused_cnn_tokens,
            "fused_vit_tokens": fused_vit_tokens,
            "cnn_readout": cnn_readout,
            "vit_readout": vit_readout,
            "fused_feature": fused_feature,
        }

        if summary_gate is not None:
            outputs["summary_gate"] = summary_gate
            outputs["fusion_gate"] = summary_gate

        for key, value in last_bridge_outputs.items():
            if key not in {"fused_cnn_tokens", "fused_vit_tokens"}:
                outputs[key] = value

        return outputs

    def _run_matched_token_gated_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行 token bridge、重采样双分支，再融合对齐 token / Run token
        bridge, resample both branches, then fuse aligned tokens."""
        assert self.resnet_backbone is not None
        assert self.vit_backbone is not None
        assert self.resnet_token_projector is not None
        assert self.vit_token_projector is not None
        assert self.token_fusion_blocks is not None
        assert self.token_resampler is not None
        assert self.token_dim_fusion is not None

        resnet_out = self.resnet_backbone(x)
        vit_out = self.vit_backbone(x)

        resnet_feature_map = resnet_out["feature_map"]
        resnet_pooled_feature = resnet_out["pooled_feature"]
        vit_tokens = vit_out["all_tokens"]

        cnn_tokens, cnn_global_token, cnn_spatial_tokens = self._build_cnn_tokens(
            feature_map=resnet_feature_map,
            pooled_feature=resnet_pooled_feature,
        )
        vit_tokens_projected = self.vit_token_projector(vit_tokens)

        fused_cnn_tokens = cnn_tokens
        fused_vit_tokens = vit_tokens_projected

        last_bridge_outputs: dict[str, torch.Tensor] = {}
        for bridge_block in self.token_fusion_blocks:
            last_bridge_outputs = bridge_block(
                cnn_tokens=fused_cnn_tokens,
                vit_tokens=fused_vit_tokens,
                return_gate=True,
                return_attn_weights=False,
            )
            fused_cnn_tokens = last_bridge_outputs["fused_cnn_tokens"]
            fused_vit_tokens = last_bridge_outputs["fused_vit_tokens"]

        matched_cnn_tokens = self.token_resampler(fused_cnn_tokens)
        matched_vit_tokens = self.token_resampler(fused_vit_tokens)
        fused_matched_tokens, matched_gate = self.token_dim_fusion(
            matched_cnn_tokens,
            matched_vit_tokens,
            return_gate=True,
        )

        pooled_feature = fused_matched_tokens.mean(dim=1)
        logits = self.classifier(pooled_feature)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "resnet_feature_map": resnet_feature_map,
            "resnet_pooled_feature": resnet_pooled_feature,
            "vit_tokens": vit_tokens,
            "cnn_global_token": cnn_global_token,
            "cnn_spatial_tokens": cnn_spatial_tokens,
            "cnn_tokens": cnn_tokens,
            "vit_tokens_projected": vit_tokens_projected,
            "fused_cnn_tokens": fused_cnn_tokens,
            "fused_vit_tokens": fused_vit_tokens,
            "matched_cnn_tokens": matched_cnn_tokens,
            "matched_vit_tokens": matched_vit_tokens,
            "fused_matched_tokens": fused_matched_tokens,
            "matched_gate": matched_gate,
            "fusion_gate": matched_gate,
            "fused_feature": pooled_feature,
        }

        for key, value in last_bridge_outputs.items():
            if key not in {"fused_cnn_tokens", "fused_vit_tokens"}:
                outputs[key] = value

        return outputs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行当前配置对应的前向路径 / Run the configured forward path."""
        if self.model_mode == "resnet_only":
            feature_dict = self.extract_branch_features(x)
            resnet_feature = feature_dict["resnet_feature"]
            logits = self.classifier(resnet_feature)
            return {
                "logits": logits,
                "resnet_feature": resnet_feature,
                "fused_feature": resnet_feature,
            }

        if self.model_mode == "vit_only":
            feature_dict = self.extract_branch_features(x)
            vit_feature = feature_dict["vit_feature"]
            logits = self.classifier(vit_feature)
            return {
                "logits": logits,
                "vit_feature": vit_feature,
                "fused_feature": vit_feature,
            }

        if self.model_mode == "dual":
            if self.fusion_type == "token_bridge":
                return self._run_token_bridge_forward(x)
            if self.fusion_type == "matched_token_gated":
                return self._run_matched_token_gated_forward(x)
            return self._run_global_dual_forward(x)

        raise RuntimeError(f"Unexpected model_mode={self.model_mode}")


def _demo_forward_dual_token_bridge() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    model = DualEncoderModel(
        num_classes=5,
        model_mode="dual",
        resnet_name="resnet18",
        vit_name="vit_b_32",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="token_bridge",
        fusion_dim=256,
        projector_hidden_dim=512,
        fusion_hidden_dim=256,
        dropout=0.1,
        token_num_heads=8,
        token_gate_hidden_dim=256,
        token_ffn_hidden_dim=512,
        token_use_gate=True,
        num_bridge_layers=1,
        summary_fusion_type="gated",
        use_cnn_pos_embed=True,
        cnn_pos_embed_base_size=7,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    print("==== DualEncoderModel Forward Test (Dual + TokenBridge) ====")
    print("Input shape            :", x.shape)
    print("CNN tokens shape       :", outputs["cnn_tokens"].shape)
    print("ViT projected shape    :", outputs["vit_tokens_projected"].shape)
    print("Fused CNN tokens shape :", outputs["fused_cnn_tokens"].shape)
    print("Fused ViT tokens shape :", outputs["fused_vit_tokens"].shape)
    print("Fused feature shape    :", outputs["fused_feature"].shape)
    print("Logits shape           :", outputs["logits"].shape)


if __name__ == "__main__":
    _demo_forward_dual_token_bridge()
