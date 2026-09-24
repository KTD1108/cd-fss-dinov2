import torch
import torch.nn as nn
from typing import List, Dict

class DINOv2Backbone(nn.Module):
    """
    Module trích xuất Patch Tokens đa tầng từ DINOv2 Backbone (ViT-S/14, ViT-B/14, v.v.)
    Đã đóng băng toàn bộ weights và gradient của backbone.
    """
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        intermediate_layers: List[int] = [2, 5, 8, 11],
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.intermediate_layers = intermediate_layers

        # Nạp DINOv2 backbone từ PyTorch Hub
        if pretrained:
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        else:
            raise ValueError("Mô hình DINOv2 yêu cầu nạp pretrained weights.")

        # Đóng băng gradient
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Trích xuất multi-level spatial feature maps từ ảnh đầu vào.
        
        Args:
            x: Input tensor kích thước [B, 3, H, W]
            
        Returns:
            List các feature maps kích thước [B, embed_dim, H/patch_size, W/patch_size]
        """
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        with torch.no_grad():
            # Thử dùng hàm get_intermediate_layers của DINOv2
            try:
                features = self.backbone.get_intermediate_layers(
                    x,
                    n=self.intermediate_layers,
                    reshape=True,
                    return_class_token=False
                )
            except Exception:
                # Fallback nếu reshape không có trong phiên bản hub cũ
                raw_features = self.backbone.get_intermediate_layers(
                    x,
                    n=self.intermediate_layers,
                    return_class_token=False
                )
                features = []
                for feat in raw_features:
                    # feat shape: [B, N_patches, C]
                    feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, grid_h, grid_w)
                    features.append(feat)

        return features

if __name__ == "__main__":
    # Smoke test cho DINOv2Backbone
    print("Testing DINOv2Backbone module...")
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        model = DINOv2Backbone(backbone_name="dinov2_vits14", intermediate_layers=[2, 5, 8, 11])
        out_feats = model(dummy_input)
        print(f"Backbone Loaded: {model.backbone_name}")
        print(f"Embed Dimension: {model.embed_dim}")
        for idx, feat in enumerate(out_feats):
            print(f"Feature Level {idx + 1} Shape: {feat.shape}")
        print("Smoke test passed successfully!")
    except Exception as e:
        print(f"Smoke test failed with error: {e}")
