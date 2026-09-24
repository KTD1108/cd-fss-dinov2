import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DenseCrossAttention(nn.Module):
    """
    Tính Dense Cross-Attention giữa Query Feature Map và Support Feature Map
    dựa trên Support Mask.
    """
    def __init__(self, temperature_scale: float = 20.0):
        super().__init__()
        self.temperature_scale = temperature_scale

    def forward(self, query_feat: torch.Tensor, support_feat: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        """
        query_feat:   [B, C, H, W]
        support_feat: [B, C, H, W]
        support_mask: [B, 1, H_orig, W_orig] hoặc [B, 1, H, W]
        
        Returns:
            pred_similarity_map: [B, 1, H, W]
        """
        B, C, H, W = query_feat.shape

        # Reshape Q, K về [B, N, C] với N = H * W
        Q = query_feat.flatten(2).permute(0, 2, 1) # [B, N, C]
        K = support_feat.flatten(2)                # [B, C, N]

        # Rescale & Normalization (Cosine similarity)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=1)

        # Cosine Correlation Matrix [B, N_query, N_support] scaled by temperature multiplier
        attn = torch.bmm(Q_norm, K_norm) * self.temperature_scale
        attn = F.softmax(attn, dim=-1) # Softmax trên không gian support

        # Downsample support mask về kích thước feature (H, W)
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)
        
        mask_downsampled = F.interpolate(support_mask.float(), size=(H, W), mode="nearest")
        V = mask_downsampled.flatten(2).permute(0, 2, 1) # [B, N, 1]

        # Dự đoán phân đoạn Query từ dense cross-attention: [B, 1, H, W]
        pred_map_dense = torch.bmm(attn, V).permute(0, 2, 1).reshape(B, 1, H, W)

        # Trích xuất Foreground Prototype & Background Prototype từ Support Feature
        s_fg_mask = mask_downsampled
        s_bg_mask = 1.0 - s_fg_mask

        s_fg_count = s_fg_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        s_bg_count = s_bg_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

        proto_fg = (support_feat * s_fg_mask).sum(dim=(2, 3), keepdim=True) / s_fg_count
        proto_bg = (support_feat * s_bg_mask).sum(dim=(2, 3), keepdim=True) / s_bg_count

        proto_fg_norm = F.normalize(proto_fg, dim=1)
        proto_bg_norm = F.normalize(proto_bg, dim=1)

        # Cosine similarity giữa Query Feature với Foreground và Background Prototype
        q_feat_norm = F.normalize(query_feat, dim=1)
        sim_fg = (q_feat_norm * proto_fg_norm).sum(dim=1, keepdim=True) * 10.0
        sim_bg = (q_feat_norm * proto_bg_norm).sum(dim=1, keepdim=True) * 10.0

        # Dual-Prototype Softmax Competition
        sim_concat = torch.cat([sim_bg, sim_fg], dim=1) # [B, 2, H, W]
        proto_prob = F.softmax(sim_concat, dim=1)[:, 1:2, :, :] # Lấy xác suất Foreground

        # Kết hợp ensemble giữa Dense Attention Map và Dual-Prototype Softmax Map
        pred_map = 0.5 * pred_map_dense + 0.5 * proto_prob
        return pred_map

class MultiLayerFusion(nn.Module):
    """
    Nội suy các correlation/pred maps từ nhiều tầng về kích thước ảnh gốc và lấy trung bình có trọng số.
    """
    def __init__(self, layer_weights: List[float] = [0.1, 0.2, 0.35, 0.35]):
        super().__init__()
        self.cross_attn = DenseCrossAttention()
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

        # Trọng số ưu tiên tầng ngữ nghĩa cao (Layer 8, 11)
        stacked_preds = torch.stack(pred_maps, dim=1) # [B, L, 1, H, W]
        w = self.weights.view(1, -1, 1, 1, 1).to(stacked_preds.device)
        fused_pred = (stacked_preds * w).sum(dim=1)
        return fused_pred

if __name__ == "__main__":
    print("Testing DenseCrossAttention & MultiLayerFusion...")
    q_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    s_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    s_mask = (torch.rand(1, 1, 224, 224) > 0.5).float()

    fusion_module = MultiLayerFusion()
    fused_out = fusion_module(q_feats, s_feats, s_mask, target_size=(224, 224))
    print(f"Fused Prediction Shape: {fused_out.shape}")
    print("Attention module tests passed successfully!")
