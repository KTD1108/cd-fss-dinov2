import torch
import torch.nn as nn
import torchvision.models as models
from typing import List

class ResNetBackbone(nn.Module):
    """
    Backbone ResNet50 chuẩn (Supervised ImageNet) cho FSS.
    Trích xuất các tầng trung gian (Layer 2, Layer 3, Layer 4) để so sánh đặc trưng.
    Sự vượt trội của ResNet nằm ở 'Inductive Bias' giúp nhận diện kết cấu không gian cục bộ (Texture)
    cực kỳ tốt trên ảnh viễn thám, khắc phục hoàn toàn điểm yếu của DINOv2.
    """
    def __init__(self, backbone_name="resnet50", pretrained=True):
        super().__init__()
        if backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone_name == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Ta sẽ dùng 4 tầng: layer1, layer2, layer3, layer4
        # Nhưng để ép chúng về cùng 1 số chiều (như DINOv2: 384) để dùng chung với Fusion Module
        self.embed_dim = 384
        
        # Linear projections để ép kích thước channel
        self.proj1 = nn.Conv2d(256, self.embed_dim, 1)
        self.proj2 = nn.Conv2d(512, self.embed_dim, 1)
        self.proj3 = nn.Conv2d(1024, self.embed_dim, 1)
        self.proj4 = nn.Conv2d(2048, self.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x) # [B, 256, H/4, W/4]
        f2 = self.layer2(f1) # [B, 512, H/8, W/8]
        f3 = self.layer3(f2) # [B, 1024, H/16, W/16]
        f4 = self.layer4(f3) # [B, 2048, H/32, W/32]

        # Phóng to tất cả về H/4 (giống hệt tư tưởng của ConvNeXt)
        h, w = f1.shape[-2:]
        f2 = torch.nn.functional.interpolate(f2, size=(h, w), mode='bilinear', align_corners=False)
        f3 = torch.nn.functional.interpolate(f3, size=(h, w), mode='bilinear', align_corners=False)
        f4 = torch.nn.functional.interpolate(f4, size=(h, w), mode='bilinear', align_corners=False)

        # Ép channel về 384
        out1 = self.proj1(f1)
        out2 = self.proj2(f2)
        out3 = self.proj3(f3)
        out4 = self.proj4(f4)

        return [out1, out2, out3, out4]
