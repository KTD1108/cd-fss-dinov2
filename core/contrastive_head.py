import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict

def ssim_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)

def keep_var_loss(orig_feats: torch.Tensor, trans_feats: torch.Tensor) -> torch.Tensor:
    """
    Giữ ổn định phân phối đặc trưng (mean và variance) giữa ảnh gốc và ảnh biến dạng.
    """
    mean_diff = orig_feats.mean(dim=(-2, -1)) - trans_feats.mean(dim=(-2, -1))
    var_diff = orig_feats.var(dim=(-2, -1)) - trans_feats.var(dim=(-2, -1))
    return torch.abs(mean_diff).mean() + torch.abs(var_diff).mean()

def dense_info_nce_loss(orig_feats: torch.Tensor, trans_feats: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    InfoNCE Loss đo lường sự bất biến của các pixel tương ứng giữa ảnh gốc và ảnh biến dạng.
    """
    B, C, H, W = trans_feats.shape
    o = orig_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
    t = trans_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # Normalize L2
    o_norm = F.normalize(o, dim=-1)
    t_norm = F.normalize(t, dim=-1)

    # Positive pair logits
    pos_logits = (o_norm * t_norm).sum(dim=-1) / temperature # [B, HW]
    
    # Subsample negative pairs to avoid OOM
    step = max(1, (H * W) // 128)
    o_sub = o_norm[:, ::step, :]
    t_sub = t_norm[:, ::step, :]
    all_logits = torch.bmm(o_sub, t_sub.transpose(1, 2)) / temperature

    max_logits = torch.max(all_logits, dim=-1, keepdim=True).values
    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(all_logits - max_logits), dim=-1, keepdim=True) + 1e-8)
    loss = -(pos_logits[:, ::step] - log_sum_exp.squeeze(-1))
    return loss.mean()

def prototype_contrastive_loss(orig_feats: torch.Tensor, trans_feats: torch.Tensor, mask: torch.Tensor, mask_trans: torch.Tensor) -> torch.Tensor:
    """
    Tách biệt Foreground Prototype khỏi Background Prototype qua các biến dạng.
    """
    B, C, H, W = orig_feats.shape
    m_o = F.interpolate(mask.float(), size=(H, W), mode='nearest')
    m_t = F.interpolate(mask_trans.float(), size=(H, W), mode='nearest')

    fg_count_o = m_o.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    bg_count_o = (1.0 - m_o).sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    fg_proto_o = (orig_feats * m_o).sum(dim=(2, 3)) / fg_count_o.squeeze()
    
    fg_count_t = m_t.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    bg_count_t = (1.0 - m_t).sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    fg_proto_t = (trans_feats * m_t).sum(dim=(2, 3)) / fg_count_t.squeeze()
    bg_proto_t = (trans_feats * (1.0 - m_t)).sum(dim=(2, 3)) / bg_count_t.squeeze()

    fg_proto_o = F.normalize(fg_proto_o, dim=-1)
    fg_proto_t = F.normalize(fg_proto_t, dim=-1)
    bg_proto_t = F.normalize(bg_proto_t, dim=-1)

    sim_pos = torch.exp(ssim_cos(fg_proto_o, fg_proto_t) / 0.5)
    sim_neg = torch.exp(ssim_cos(fg_proto_o, bg_proto_t) / 0.5)

    loss = -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))
    return loss.mean()

class ContrastiveFeatureTransformer(nn.Module):
    """
    Projector Head chiếu đặc trưng DINOv2 (384) xuống không gian biểu diễn (64)
    được tối ưu hóa qua cơ chế Adapt Before Comparison.
    """
    def __init__(self, in_channels: int = 384, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))

class ClassContrastiveAdapters:
    """
    Quản lý các bộ Contrastive Transformers theo từng Class ID (chuẩn ABCDFSS).
    Thích nghi trên tập dữ liệu mục tiêu ở episode đầu tiên của class và tái sử dụng cho các episode sau.
    """
    def __init__(self, num_classes: int = 6, num_layers: int = 4, in_dim: int = 384, out_dim: int = 64, lr: float = 0.01, epochs: int = 25):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.epochs = epochs

        # Danh sách mô hình cho từng class
        self.class_models: Dict[int, nn.ModuleList] = {}
        self.fitted_classes = set()

    def get_models_for_class(self, class_id: int, device: torch.device) -> nn.ModuleList:
        if class_id not in self.class_models:
            models = nn.ModuleList([
                ContrastiveFeatureTransformer(self.in_dim, self.out_dim)
                for _ in range(self.num_layers)
            ]).to(device)
            self.class_models[class_id] = models
        return self.class_models[class_id]

    def has_fitted(self, class_id: int) -> bool:
        return class_id in self.fitted_classes

    def fit_class(
        self,
        class_id: int,
        q_feats: List[torch.Tensor],
        q_aug_feats: List[torch.Tensor],
        s_feats: List[torch.Tensor],
        s_aug_feats: List[torch.Tensor],
        s_mask: torch.Tensor,
        s_aug_mask: torch.Tensor,
        device: torch.device
    ):
        models = self.get_models_for_class(class_id, device)
        models.train()
        optimizer = torch.optim.SGD(models.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        for ep in range(self.epochs):
            optimizer.zero_grad()
            total_loss = 0.0

            for l in range(self.num_layers):
                q_t = models[l](q_feats[l])
                q_aug_t = models[l](q_aug_feats[l])
                s_t = models[l](s_feats[l])
                s_aug_t = models[l](s_aug_feats[l])

                # InfoNCE Losses
                loss_q_nce = dense_info_nce_loss(q_t, q_aug_t)
                loss_s_nce = dense_info_nce_loss(s_t, s_aug_t)
                loss_stat = keep_var_loss(q_t, q_aug_t) + keep_var_loss(s_t, s_aug_t)
                loss_proto = prototype_contrastive_loss(s_t, s_aug_t, s_mask, s_aug_mask)

                total_loss += (loss_q_nce + loss_s_nce + 0.5 * loss_stat + 1.0 * loss_proto)

            total_loss.backward()
            optimizer.step()

        models.eval()
        self.fitted_classes.add(class_id)
        return total_loss.item()

    def transform(self, class_id: int, feats: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
        models = self.get_models_for_class(class_id, device)
        models.eval()
        with torch.no_grad():
            return [models[l](feats[l]) for l in range(self.num_layers)]
