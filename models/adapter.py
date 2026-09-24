import torch
import torch.nn as nn
from typing import List

class ConvAdapterBlock(nn.Module):
    """
    Adapter 1x1 Conv Bottleneck cho một tầng feature map.
    Mô hình: 1x1 Conv -> BN -> ReLU -> 1x1 Conv
    Residual connection: output = input + adapter(input)
    """
    def __init__(self, in_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, adapter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

class MultiLevelAdapters(nn.Module):
    """
    Quản lý danh sách các Adapters cho từng tầng feature map thu được từ DINOv2 Backbone.
    """
    def __init__(self, num_levels: int = 4, in_dim: int = 384, adapter_dim: int = 64):
        super().__init__()
        self.adapters = nn.ModuleList([
            ConvAdapterBlock(in_dim, adapter_dim) for _ in range(num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Nhận danh sách các feature maps từ DINOv2 và áp dụng Adapter từng tầng.
        """
        adapted_features = []
        for feat, adapter in zip(features, self.adapters):
            adapted_features.append(adapter(feat))
        return adapted_features

if __name__ == "__main__":
    print("Testing MultiLevelAdapters module...")
    dummy_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    adapters = MultiLevelAdapters(num_levels=4, in_dim=384, adapter_dim=64)
    out_feats = adapters(dummy_feats)
    for idx, f in enumerate(out_feats):
        print(f"Adapted Feature Level {idx + 1} Shape: {f.shape}")
    print("Adapter test passed successfully!")
