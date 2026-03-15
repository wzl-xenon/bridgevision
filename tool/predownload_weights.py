from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)

# 下载 ResNet50 默认权重
_ = resnet50(weights=ResNet50_Weights.DEFAULT)

# 下载 ViT-B/16 默认权重
_ = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

print("Pretrained weights downloaded.")