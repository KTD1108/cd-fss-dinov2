import torch
import torch.nn as nn
from core.contrastive_head import ContrastiveFeatureTransformer, ClassContrastiveAdapters

class ConvAdapterBlock(nn.Module):
    """
    Adapter Bottleneck 1x1 Conv cho một tầng feature map với GroupNorm và Residual connection.
    """
    def __init__(self, in_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, adapter_dim, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, adapter_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(adapter_dim, in_dim, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(8, in_dim)

        # Zero-init
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.gn2.weight)
        nn.init.zeros_(self.gn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gn2(self.conv2(self.relu(self.gn1(self.conv1(x)))))

class MultiLevelAdapters(nn.Module):
    """
    Quản lý danh sách các Adapters cho từng tầng feature map thu được từ DINOv2 Backbone.
    """
    def __init__(self, num_levels: int = 4, in_dim: int = 384, adapter_dim: int = 64):
        super().__init__()
        self.adapters = nn.ModuleList([
            ConvAdapterBlock(in_dim, adapter_dim) for _ in range(num_levels)
        ])

    def forward(self, features):
        return [adapter(feat) for feat, adapter in zip(features, self.adapters)]

if __name__ == "__main__":
    print("Testing models/adapter.py...")
    adapters = MultiLevelAdapters(4, 384, 64)
    dummy_feats = [torch.randn(1, 384, 14, 14) for _ in range(4)]
    out = adapters(dummy_feats)
    diff = sum([(o - i).abs().max().item() for o, i in zip(out, dummy_feats)])
    print(f"Zero-init max deviation: {diff:.6f}")
    assert diff < 1e-6
    print("Adapter test passed!")
