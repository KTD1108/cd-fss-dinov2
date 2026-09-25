import torch
import torch.nn as nn
from typing import List

class ConvAdapterBlock(nn.Module):
    """
    Adapter 1x1 Conv Bottleneck cho một tầng feature map.
    Mô hình: 1x1 Conv -> BN -> ReLU -> 1x1 Conv -> BN
    Residual connection: output = input + adapter(input)
    Đặc biệt: Khởi tạo Zero-Init ở lớp conv cuối giúp bảo toàn 100% đặc trưng ban đầu của DINOv2.
    """
    def __init__(self, in_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, adapter_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(adapter_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(adapter_dim, in_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_dim)

        # 🌟 ZERO-INITIALIZATION: Đảm bảo tại bước 0 (trước TTA), adapter(x) = 0.
        # Đầu ra ban đầu = x + 0 = x, không làm hỏng không gian biểu diễn hoàn hảo của DINOv2.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return x + residual

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
    # Kiểm tra xem zero-init có cho kết quả bằng hệt đầu vào ban đầu không
    diff = sum([(out - inp).abs().max().item() for out, inp in zip(out_feats, dummy_feats)])
    print(f"Zero-init difference from input: {diff:.6f} (Expected: 0.000000)")
    assert diff < 1e-6, "Error: Zero-init is not exact!"
    print("Adapter test passed successfully!")
