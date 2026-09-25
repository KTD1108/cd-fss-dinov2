import torch
import torch.nn as nn
import torch.nn.functional as F
from core.contrastive_head import dense_info_nce_loss, keep_var_loss, prototype_contrastive_loss

class DenseInfoNCELoss(nn.Module):
    """
    Module bọc quanh dense_info_nce_loss cho phép sử dụng theo phong cách nn.Module.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, orig_feats: torch.Tensor, trans_feats: torch.Tensor) -> torch.Tensor:
        return dense_info_nce_loss(orig_feats, trans_feats, self.temperature)

class FeatureStatVarianceLoss(nn.Module):
    """
    Module bọc quanh keep_var_loss cân bằng thống kê đặc trưng (Mean/Variance).
    """
    def __init__(self):
        super().__init__()

    def forward(self, orig_feats: torch.Tensor, trans_feats: torch.Tensor) -> torch.Tensor:
        return keep_var_loss(orig_feats, trans_feats)

class PrototypeContrastiveLoss(nn.Module):
    """
    Module bọc quanh prototype_contrastive_loss tách biệt Foreground/Background Prototypes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, orig_feats: torch.Tensor, trans_feats: torch.Tensor, mask: torch.Tensor, mask_trans: torch.Tensor) -> torch.Tensor:
        return prototype_contrastive_loss(orig_feats, trans_feats, mask, mask_trans)

# Aliases tương thích ngược
DenseContrastiveLoss = DenseInfoNCELoss
FeatureStatLoss = FeatureStatVarianceLoss
PrototypeAlignmentLoss = PrototypeContrastiveLoss

if __name__ == "__main__":
    print("Testing unified Loss modules...")
    f1 = torch.randn(1, 64, 16, 16)
    f2 = torch.randn(1, 64, 16, 16)
    m = torch.zeros(1, 1, 16, 16)
    m[:, :, 4:12, 4:12] = 1.0

    l_nce = DenseInfoNCELoss()(f1, f2)
    l_stat = FeatureStatVarianceLoss()(f1, f2)
    l_p = PrototypeContrastiveLoss()(f1, f2, m, m)

    print(f"Unified L_nce: {l_nce.item():.4f}, L_stat: {l_stat.item():.4f}, L_proto: {l_p.item():.4f}")
    print("All Loss tests passed!")
