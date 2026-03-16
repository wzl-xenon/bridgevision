from __future__ import annotations

import torch

from src.models.dual_encoder_model import DualEncoderModel
from src.models.fusions.token_bridge_fusion import TokenBridgeFusion


def test_token_bridge_fusion_shapes() -> None:
    fusion = TokenBridgeFusion(
        feature_dim=128,
        num_heads=8,
        dropout=0.0,
        gate_hidden_dim=128,
        ffn_hidden_dim=256,
        use_gate=True,
        use_layernorm=True,
    )
    fusion.eval()

    cnn_tokens = torch.randn(2, 50, 128)
    vit_tokens = torch.randn(2, 197, 128)

    with torch.no_grad():
        outputs = fusion(
            cnn_tokens=cnn_tokens,
            vit_tokens=vit_tokens,
            return_gate=True,
            return_attn_weights=False,
        )

    assert outputs["fused_cnn_tokens"].shape == cnn_tokens.shape
    assert outputs["fused_vit_tokens"].shape == vit_tokens.shape
    assert outputs["cnn_gate"].shape == cnn_tokens.shape
    assert outputs["vit_gate"].shape == vit_tokens.shape


def test_dual_encoder_token_bridge_forward_shapes() -> None:
    model = DualEncoderModel(
        num_classes=5,
        model_mode="dual",
        resnet_name="resnet18",
        vit_name="vit_b_32",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="token_bridge",
        fusion_dim=128,
        projector_hidden_dim=256,
        fusion_hidden_dim=128,
        dropout=0.0,
        token_num_heads=8,
        token_gate_hidden_dim=128,
        token_ffn_hidden_dim=256,
        token_use_gate=True,
        num_bridge_layers=1,
        summary_fusion_type="gated",
        use_cnn_pos_embed=True,
        cnn_pos_embed_base_size=7,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(x)

    assert outputs["cnn_tokens"].ndim == 3
    assert outputs["vit_tokens_projected"].ndim == 3
    assert outputs["fused_cnn_tokens"].shape == outputs["cnn_tokens"].shape
    assert outputs["fused_vit_tokens"].shape == outputs["vit_tokens_projected"].shape
    assert outputs["cnn_readout"].shape == (2, 128)
    assert outputs["vit_readout"].shape == (2, 128)
    assert outputs["fused_feature"].shape == (2, 128)
    assert outputs["logits"].shape == (2, 5)


def test_dual_encoder_matched_token_gated_forward_shapes() -> None:
    model = DualEncoderModel(
        num_classes=5,
        model_mode="dual",
        resnet_name="resnet18",
        vit_name="vit_b_32",
        pretrained_backbones=False,
        freeze_backbones=False,
        projector_type="mlp",
        fusion_type="matched_token_gated",
        fusion_dim=128,
        projector_hidden_dim=256,
        fusion_hidden_dim=128,
        dropout=0.0,
        token_num_heads=8,
        token_gate_hidden_dim=128,
        token_ffn_hidden_dim=256,
        token_use_gate=True,
        num_bridge_layers=1,
        matched_token_count=12,
        summary_fusion_type="gated",
        use_cnn_pos_embed=True,
        cnn_pos_embed_base_size=7,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(x)

    assert outputs["matched_cnn_tokens"].shape == (2, 12, 128)
    assert outputs["matched_vit_tokens"].shape == (2, 12, 128)
    assert outputs["fused_matched_tokens"].shape == (2, 12, 128)
    assert outputs["matched_gate"].shape == (2, 12, 128)
    assert outputs["fused_feature"].shape == (2, 128)
    assert outputs["logits"].shape == (2, 5)


def test_resnet_only_forward_shapes() -> None:
    model = DualEncoderModel(
        num_classes=5,
        model_mode="resnet_only",
        resnet_name="resnet18",
        pretrained_backbones=False,
        freeze_backbones=False,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(x)

    assert outputs["resnet_feature"].shape == (2, 512)
    assert outputs["fused_feature"].shape == (2, 512)
    assert outputs["logits"].shape == (2, 5)


def test_vit_only_forward_shapes() -> None:
    model = DualEncoderModel(
        num_classes=5,
        model_mode="vit_only",
        vit_name="vit_b_32",
        pretrained_backbones=False,
        freeze_backbones=False,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(x)

    assert outputs["vit_feature"].shape == (2, 768)
    assert outputs["fused_feature"].shape == (2, 768)
    assert outputs["logits"].shape == (2, 5)
