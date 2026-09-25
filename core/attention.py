import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.thresholding import compute_adaptive_threshold_mask

class DualCrossAttention(nn.Module):
    """
    Module tính toán tương quan hai chiều (Dual Foreground/Background Density Matching)
    giữa Query Feature Map và Support Feature Map dựa trên Support Mask.
    
    Cơ chế:
    So sánh độ tương đồng của từng pixel Query đồng thời với cả tập hợp Foreground 
    và tập hợp Background của Support, triệt tiêu độ lệch tỷ lệ diện tích và ngăn chặn sập ngưỡng.
    """
    def __init__(self, temperature: float = 0.15, proto_ratio: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.proto_ratio = proto_ratio

    def forward(
        self, 
        query_feat: torch.Tensor, 
        support_feat: torch.Tensor, 
        support_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        query_feat:   [B, C, H, W]
        support_feat: [B, C, H, W]
        support_mask: [B, 1, H_orig, W_orig] hoặc [B, 1, H, W]
        
        Returns:
            prob_map: [B, 1, H, W] xác suất Foreground chuẩn hóa thực thụ trong dải [0, 1]
        """
        B, C, H, W = query_feat.shape

        # Chuẩn hóa L2 dọc theo kênh C
        Q = F.normalize(query_feat, dim=1).flatten(2).permute(0, 2, 1) # [B, HW, C]
        K = F.normalize(support_feat, dim=1).flatten(2)                 # [B, C, HW]

        # Ma trận tương quan Cosine: [B, HW_q, HW_s]
        sim_matrix = torch.bmm(Q, K)

        # Downsample support mask về kích thước không gian của feature map (H, W)
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)
        m_down = F.interpolate(support_mask.float(), size=(H, W), mode="nearest")
        
        m_fg = m_down.flatten(2)          # [B, 1, HW_s]
        m_bg = (1.0 - m_down).flatten(2)  # [B, 1, HW_s]

        n_fg = m_fg.sum(dim=-1, keepdim=True).clamp(min=1.0) # [B, 1, 1]
        n_bg = m_bg.sum(dim=-1, keepdim=True).clamp(min=1.0) # [B, 1, 1]

        # 1. Dense Token Affinity: Tương quan trung bình đến từng token FG và BG
        sim_dense_fg = (sim_matrix * m_fg).sum(dim=-1) / n_fg.squeeze(-1) # [B, HW_q]
        sim_dense_bg = (sim_matrix * m_bg).sum(dim=-1) / n_bg.squeeze(-1) # [B, HW_q]

        # 2. Prototype Affinity: Tương quan đến Prototype đại diện FG và BG
        K_t = K.permute(0, 2, 1) # [B, HW_s, C]
        proto_fg = (K_t * m_fg.permute(0, 2, 1)).sum(dim=1) / n_fg.squeeze(-1) # [B, C]
        proto_bg = (K_t * m_bg.permute(0, 2, 1)).sum(dim=1) / n_bg.squeeze(-1) # [B, C]

        proto_fg = F.normalize(proto_fg, dim=-1).unsqueeze(-1) # [B, C, 1]
        proto_bg = F.normalize(proto_bg, dim=-1).unsqueeze(-1) # [B, C, 1]

        sim_proto_fg = torch.bmm(Q, proto_fg).squeeze(-1) # [B, HW_q]
        sim_proto_bg = torch.bmm(Q, proto_bg).squeeze(-1) # [B, HW_q]

        # Kết hợp Dense Affinity và Prototype Affinity
        score_fg = (1.0 - self.proto_ratio) * sim_dense_fg + self.proto_ratio * sim_proto_fg
        score_bg = (1.0 - self.proto_ratio) * sim_dense_bg + self.proto_ratio * sim_proto_bg

        # 3. Phân loại 2 lớp (Binary Softmax)
        logits = torch.stack([score_bg, score_fg], dim=1) / self.temperature # [B, 2, HW_q]
        prob_fg = F.softmax(logits, dim=1)[:, 1].reshape(B, 1, H, W)
        return prob_fg

# Giữ lại DenseCrossAttention như một alias tương thích ngược
DenseCrossAttention = DualCrossAttention

class MultiLayerFusion(nn.Module):
    """
    Kết hợp dự đoán từ nhiều tầng DINOv2 (Intermediate Layers) và upsample về kích thước ảnh gốc.
    """
    def __init__(
        self, 
        layer_weights: List[float] = [0.25, 0.25, 0.25, 0.25], 
        temperature: float = 0.15,
        proto_ratio: float = 0.5
    ):
        super().__init__()
        self.cross_attn = DualCrossAttention(temperature=temperature, proto_ratio=proto_ratio)
        weights_tensor = torch.tensor(layer_weights, dtype=torch.float32)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum())

    def forward(
        self,
        query_feats: List[torch.Tensor],
        support_feats: List[torch.Tensor],
        support_mask: torch.Tensor,
        target_size: tuple = (392, 392)
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

if __name__ == "__main__":
    print("Testing DualCrossAttention & MultiLayerFusion...")
    q_feats = [torch.randn(1, 384, 28, 28) for _ in range(4)]
    s_feats = [torch.randn(1, 384, 28, 28) for _ in range(4)]
    s_mask = torch.zeros(1, 1, 392, 392)
    s_mask[:, :, 50:150, 50:150] = 1.0 # 6.5% FG

    fusion = MultiLayerFusion()
    fused_prob = fusion(q_feats, s_feats, s_mask, target_size=(392, 392))
    bin_mask = compute_adaptive_threshold_mask(fused_prob)

    print(f"Fused Prob min: {fused_prob.min():.4f}, max: {fused_prob.max():.4f}, mean: {fused_prob.mean():.4f}")
    print(f"Binary Mask FG ratio: {bin_mask.mean():.4f}")
    assert bin_mask.mean() < 0.70, "Error: Mask collapsed to all-ones!"
    print("All tests passed successfully!")
