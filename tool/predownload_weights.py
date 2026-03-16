from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)

# 触发两个 backbone 的默认 torchvision 预训练权重下载 / Trigger the
# default torchvision pretrained weight downloads for both backbones.
_ = resnet50(weights=ResNet50_Weights.DEFAULT)
_ = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

print("Pretrained weights downloaded.")
