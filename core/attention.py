import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
import cv2
import numpy as np

def compute_dynamic_threshold_mask(prob_map: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa Min-Max và áp dụng Otsu Dynamic Thresholding.
    Tránh hiện tượng sập về 0% khi ảnh support có ít pixel foreground làm tỷ lệ kích hoạt bị thu nhỏ.
    
    prob_map: [B, 1, H, W] hoặc [H, W] nằm trong dải [0, 1]
    Returns:
        binary_mask: [B, 1, H, W] (0.0 hoặc 1.0)
    """
    if prob_map.dim() == 2:
        prob_map = prob_map.unsqueeze(0).unsqueeze(0)
    elif prob_map.dim() == 3:
        prob_map = prob_map.unsqueeze(1)

    B, _, H, W = prob_map.shape
    binary_masks = []

    for i in range(B):
        p = prob_map[i, 0].detach().cpu().numpy()
        p_min, p_max = float(p.min()), float(p.max())
        if p_max - p_min < 1e-6:
            binary_masks.append(torch.zeros((1, H, W), device=prob_map.device))
            continue

        # Chuẩn hóa min-max để mở rộng toàn bộ dải động về [0, 1]
        p_norm = (p - p_min) / (p_max - p_min + 1e-8)
        norm_uint8 = (p_norm * 255.0).astype(np.uint8)

        # Tính ngưỡng Otsu tự động
        thresh_val, _ = cv2.threshold(norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_otsu = thresh_val / 255.0

        # Ràng buộc an toàn: không để ngưỡng quá cực đoan
        thresh = float(np.clip(thresh_otsu, 0.25, 0.75))

        mask_np = (p_norm > thresh).astype(np.float32)
        binary_masks.append(torch.from_numpy(mask_np).unsqueeze(0).to(prob_map.device))

    return torch.stack(binary_masks, dim=0)

class DenseCrossAttention(nn.Module):
    """
    Tính Dense Cross-Attention giữa Query Feature Map và Support Feature Map
    dựa trên Support Mask theo cơ chế Dense Affinity Matrix.
    """
    def __init__(self, normalize: bool = True, temperature: float = 0.15):
        super().__init__()
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, query_feat: torch.Tensor, support_feat: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        """
        query_feat:   [B, C, H, W]
        support_feat: [B, C, H, W]
        support_mask: [B, 1, H_orig, W_orig] hoặc [B, 1, H, W]
        
        Returns:
            pred_dense_map: [B, 1, H, W] trong dải [0, 1]
        """
        B, C, H, W = query_feat.shape

        if self.normalize:
            Q = F.normalize(query_feat, dim=1).flatten(2).permute(0, 2, 1) # [B, HW, C]
            K = F.normalize(support_feat, dim=1).flatten(2)                 # [B, C, HW]
            scale = 1.0 / self.temperature
        else:
            Q = query_feat.flatten(2).permute(0, 2, 1) # [B, HW, C]
            K = support_feat.flatten(2)                 # [B, C, HW]
            scale = 1.0 / math.sqrt(C)

        # Ma trận tương quan Dense Affinity: [B, HW_query, HW_support]
        affinity = torch.bmm(Q, K) * scale
        attn = F.softmax(affinity, dim=-1) # Softmax trên toàn bộ không gian pixel support

        # Downsample support mask về kích thước feature (H, W)
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)
        
        mask_downsampled = F.interpolate(support_mask.float(), size=(H, W), mode="nearest")
        V = mask_downsampled.flatten(2).permute(0, 2, 1) # [B, HW_support, 1]

        # Tích hợp xác suất Foreground cho từng pixel Query: [B, 1, H, W]
        pred_map_dense = torch.bmm(attn, V).permute(0, 2, 1).reshape(B, 1, H, W)
        return pred_map_dense

class MultiLayerFusion(nn.Module):
    """
    Nội suy các correlation/pred maps từ nhiều tầng về kích thước ảnh gốc và lấy trung bình có trọng số.
    """
    def __init__(self, layer_weights: List[float] = [0.25, 0.25, 0.25, 0.25], normalize: bool = True, temperature: float = 0.15):
        super().__init__()
        self.cross_attn = DenseCrossAttention(normalize=normalize, temperature=temperature)
        weights_tensor = torch.tensor(layer_weights, dtype=torch.float32)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum())

    def forward(
        self,
        query_feats: List[torch.Tensor],
        support_feats: List[torch.Tensor],
        support_mask: torch.Tensor,
        target_size: tuple = (224, 224)
    ) -> torch.Tensor:
        
        pred_maps = []
        for q_f, s_f in zip(query_feats, support_feats):
            pred_l = self.cross_attn(q_f, s_f, support_mask) # [B, 1, H_feat, W_feat]
            # Upsample về target size (ảnh gốc)
            pred_l_upsampled = F.interpolate(pred_l, size=target_size, mode="bilinear", align_corners=False)
            pred_maps.append(pred_l_upsampled)

        stacked_preds = torch.stack(pred_maps, dim=1) # [B, L, 1, H, W]
        w = self.weights.view(1, -1, 1, 1, 1).to(stacked_preds.device)
        fused_pred = (stacked_preds * w).sum(dim=1)
        return fused_pred

class MetaDecoder(nn.Module):
    """
    Mạng giải mã có khả năng học (Learnable Decoder) dùng cho Meta-Training.
    """
    def __init__(self, in_channels: int = 384, num_layers: int = 4):
        super().__init__()
        self.cross_attn = DenseCrossAttention()
        
        decoder_in_dim = in_channels + num_layers
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(
        self,
        query_feats: List[torch.Tensor],
        support_feats: List[torch.Tensor],
        support_mask: torch.Tensor,
        target_size: tuple = (224, 224)
    ) -> torch.Tensor:
        
        pred_maps = []
        for q_f, s_f in zip(query_feats, support_feats):
            pred_l = self.cross_attn(q_f, s_f, support_mask)
            pred_maps.append(pred_l)
            
        stacked_preds = torch.cat(pred_maps, dim=1) 
        deepest_q_feat = query_feats[-1]
        
        decoder_input = torch.cat([deepest_q_feat, stacked_preds], dim=1)
        refined_logits = self.decoder(decoder_input)
        refined_mask = F.interpolate(refined_logits, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(refined_mask)

if __name__ == "__main__":
    print("Testing DenseCrossAttention, Otsu Dynamic Thresholding & MultiLayerFusion...")
    q_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    s_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    s_mask = (torch.rand(1, 1, 224, 224) > 0.5).float()

    fusion_module = MultiLayerFusion()
    fused_out = fusion_module(q_feats, s_feats, s_mask, target_size=(224, 224))
    binary_mask = compute_dynamic_threshold_mask(fused_out)
    print(f"Fused Prediction Shape: {fused_out.shape}, Mean Prob: {fused_out.mean().item():.4f}")
    print(f"Dynamic Threshold Mask Shape: {binary_mask.shape}, FG ratio: {binary_mask.mean().item():.4f}")
    print("All tests passed successfully!")
