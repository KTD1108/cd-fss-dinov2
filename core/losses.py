import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseContrastiveLoss(nn.Module):
    """
    Dense Contrastive Loss giữa feature gốc và feature đã qua Data Augmentation.
    Đo lường độ tương đồng Cosine giữa các vị trí tương ứng của hai feature maps.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        feat1, feat2: [B, C, H, W]
        """
        B, C, H, W = feat1.shape
        # Normalize features dọc theo kênh C
        f1 = F.normalize(feat1, dim=1) # [B, C, H, W]
        f2 = F.normalize(feat2, dim=1) # [B, C, H, W]

        # Cosine similarity giữa các pixel tương ứng
        cos_sim = torch.sum(f1 * f2, dim=1) # [B, H, W]
        
        # Loss = 1.0 - average cosine similarity
        loss = 1.0 - torch.mean(cos_sim)
        return torch.clamp(loss, min=0.0)

class FeatureStatLoss(nn.Module):
    """
    Loss cân bằng Mean & Variance giữa hai feature views (chuẩn bị chống Domain Shift).
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        feat1, feat2: [B, C, H, W]
        """
        mean1 = torch.mean(feat1, dim=[2, 3])
        mean2 = torch.mean(feat2, dim=[2, 3])
        var1 = torch.var(feat1, dim=[2, 3], unbiased=False)
        var2 = torch.var(feat2, dim=[2, 3], unbiased=False)

        loss_mean = F.mse_loss(mean1, mean2)
        loss_var = F.mse_loss(var1, var2)
        return loss_mean + loss_var

class PrototypeAlignmentLoss(nn.Module):
    """
    Class Alignment Loss (L_p): Ép Prototype Foreground/Background giữa các views tương đồng.
    """
    def __init__(self):
        super().__init__()

    def _get_prototypes(self, feat: torch.Tensor, mask: torch.Tensor):
        """
        feat: [B, C, H, W]
        mask: [B, 1, H, W] hoặc [B, H, W]
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Resize mask về cùng H, W với feat
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode="nearest")

        fg_mask = (mask > 0.5).float()
        bg_mask = (mask <= 0.5).float()

        fg_sum = torch.sum(fg_mask, dim=[2, 3], keepdim=True) + 1e-5
        bg_sum = torch.sum(bg_mask, dim=[2, 3], keepdim=True) + 1e-5

        proto_fg = torch.sum(feat * fg_mask, dim=[2, 3], keepdim=True) / fg_sum
        proto_bg = torch.sum(feat * bg_mask, dim=[2, 3], keepdim=True) / bg_sum

        return proto_fg, proto_bg

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        proto_fg1, proto_bg1 = self._get_prototypes(feat1, mask1)
        proto_fg2, proto_bg2 = self._get_prototypes(feat2, mask2)

        loss_fg = F.mse_loss(proto_fg1, proto_fg2)
        loss_bg = F.mse_loss(proto_bg1, proto_bg2)
        return loss_fg + loss_bg

if __name__ == "__main__":
    print("Testing Loss functions...")
    f1 = torch.randn(1, 384, 16, 16)
    f2 = torch.randn(1, 384, 16, 16)
    m1 = (torch.rand(1, 1, 224, 224) > 0.5).float()
    m2 = (torch.rand(1, 1, 224, 224) > 0.5).float()

    l_nce = DenseContrastiveLoss()(f1, f2)
    l_stat = FeatureStatLoss()(f1, f2)
    l_p = PrototypeAlignmentLoss()(f1, f2, m1, m2)

    print(f"L_nce: {l_nce.item():.4f}")
    print(f"L_stat: {l_stat.item():.4f}")
    print(f"L_p: {l_p.item():.4f}")
    print("Loss tests passed successfully!")
