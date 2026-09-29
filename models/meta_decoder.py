import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FSSMetaDecoder(nn.Module):
    """
    Meta-Decoder chuyên dụng cho Few-Shot Segmentation.
    Nhận đặc trưng từ Backbone (DINOv2), tính toán Cosine Correlation với Support Prototype,
    sau đó dùng mạng CNN để giải mã (upsample) thành mặt nạ (Mask) với độ phân giải gốc.
    """
    def __init__(self, in_channels: int = 384, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        # Kênh đầu vào sẽ là in_channels (từ Query) + num_layers (từ 4 Correlation maps)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels + num_layers, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False), # Phóng to về kích thước gốc (14 * 2 * 7 = 196) (xấp xỉ)
            # Ta sẽ dùng F.interpolate ở cuối để ép size chính xác
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, q_feats: List[torch.Tensor], s_feats: List[torch.Tensor], s_mask: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        q_feats, s_feats: List các feature maps từ DINOv2 (4 layers)
        s_mask: Mask của support image [B, 1, H_img, W_img]
        """
        B, _, H, W = q_feats[0].shape
        
        # Đưa s_mask về cùng kích thước với feature map
        s_mask_resized = F.interpolate(s_mask.float(), size=(H, W), mode='nearest')
        
        correlations = []
        for q_l, s_l in zip(q_feats, s_feats):
            # Tính Prototype cho tầng này
            fg_count = s_mask_resized.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            s_proto = (s_l * s_mask_resized).sum(dim=(2, 3), keepdim=True) / fg_count # [B, C, 1, 1]
            
            # Chuẩn hóa L2
            q_norm = F.normalize(q_l, dim=1)
            s_proto_norm = F.normalize(s_proto, dim=1)
            
            # Tính Cosine Similarity Map
            corr = (q_norm * s_proto_norm).sum(dim=1, keepdim=True) # [B, 1, H, W]
            correlations.append(corr)
            
        # Nối (Concat) tất cả Correlation Maps với Feature Map gốc của Query (Layer 0)
        corr_tensor = torch.cat(correlations, dim=1) # [B, 4, H, W]
        fused_feats = torch.cat([q_feats[0], corr_tensor], dim=1) # [B, 384+4, H, W]
        
        # Đưa qua mạng giải mã CNN
        out = self.decoder(fused_feats) # [B, 1, H', W']
        
        # Phóng to về kích thước gốc của bức ảnh
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        
        # Sigmoid để ép xác suất về [0, 1]
        return torch.sigmoid(out)
