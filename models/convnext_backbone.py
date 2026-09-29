import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBackbone(nn.Module):
    """
    Wrapper cho ConvNeXt-V2 từ thư viện `timm`.
    Chuẩn hóa các feature maps từ các stage khác nhau về cùng số kênh (embed_dim)
    và cùng độ phân giải không gian (H/4, W/4) để tương thích 100% với DINOv2 pipeline.
    """
    def __init__(self, model_name='convnextv2_base.fcmae_ft_in22k_in1k', embed_dim=384):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Vui lòng cài đặt timm: pip install timm")

        # Load mô hình ConvNeXt-V2, bật tính năng trả về các lớp trung gian
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        self.embed_dim = embed_dim
        
        # Số kênh mặc định của 4 stages trong ConvNeXt-V2 Base: [128, 256, 512, 1024]
        # Nếu dùng bản 'tiny' hay 'large', số kênh sẽ khác, nhưng ta giả định 'base'
        feature_info = self.backbone.feature_info
        channels = [info['num_chs'] for info in feature_info]
        
        # Projector cho từng stage về chung embed_dim
        self.projectors = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in channels
        ])

    def forward(self, x):
        features = self.backbone(x)
        
        # Lấy kích thước không gian của Stage 0 (H/4, W/4) làm chuẩn
        target_size = features[0].shape[-2:]
        
        uniform_feats = []
        for i, feat in enumerate(features):
            # 1. Chiếu về số kênh chuẩn (384)
            proj_feat = self.projectors[i](feat)
            
            # 2. Phóng to (Upsample) các layer sâu về độ phân giải của Layer 0
            if i > 0:
                proj_feat = F.interpolate(proj_feat, size=target_size, mode='bilinear', align_corners=False)
                
            uniform_feats.append(proj_feat)
            
        return uniform_feats
