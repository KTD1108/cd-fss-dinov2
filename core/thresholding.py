import torch
import numpy as np

def compute_adaptive_threshold_mask(
    prob_map: torch.Tensor,
    drop_least: float = 0.05,
    max_fg_ratio: float = 0.70,
    support_fg_ratio: float = None
) -> torch.Tensor:
    """
    Phân ngưỡng thích ứng dựa trên thuật toán Otsu kết hợp chốt chặn an toàn (Safeguard).
    Hỗ trợ chốt chặn giới hạn diện tích (Support-Ratio Guard) đặc biệt cho DeepGlobe.

    Args:
        prob_map: Tensor [B, 1, H, W] hoặc [H, W] chứa xác suất trong dải [0, 1].
        drop_least: Tỷ lệ loại bỏ các giá trị nhiễu ở cận dưới (mặc định 5%).
        max_fg_ratio: Tỷ lệ tối đa của Foreground (mặc định 70%) nhằm ngăn chặn
                      hiện tượng sập ngưỡng toàn bộ về 1 (Degenerate All-Ones).

    Returns:
        binary_mask: Tensor [B, 1, H, W] dạng nhị phân (0.0 hoặc 1.0).
    """
    if prob_map.dim() == 2:
        prob_map = prob_map.unsqueeze(0).unsqueeze(0)
    elif prob_map.dim() == 3:
        prob_map = prob_map.unsqueeze(1)

    B, _, H, W = prob_map.shape
    batch_masks = []

    for i in range(B):
        p_tensor = prob_map[i, 0] # [H, W] on GPU
        p_min, p_max = p_tensor.min(), p_tensor.max()

        if p_max - p_min < 1e-5:
            batch_masks.append(torch.zeros((1, H, W), device=prob_map.device))
            continue

        # Chuẩn hóa về [0, 255]
        p_norm = (p_tensor - p_min) / (p_max - p_min + 1e-8)
        norm_uint8 = (p_norm * 255.0).to(torch.uint8)

        # Tính histogram trên GPU (256 bins)
        hist = torch.histc(norm_uint8.float(), bins=256, min=0, max=255)
        
        # Loại bỏ nhiễu ở mức thấp nhất
        total_pixels = hist.sum()
        drop_count = int(total_pixels.item() * drop_least)
        cum_hist = torch.cumsum(hist, dim=0)
        valid_start = torch.searchsorted(cum_hist, drop_count).item()
        
        # Otsu's method (Vectorized on GPU)
        hist[0:valid_start] = 0
        total_pixels = hist.sum()
        
        weight_b = torch.cumsum(hist, dim=0)
        weight_f = total_pixels - weight_b
        
        # Tránh chia cho 0
        weight_b_safe = weight_b.clone()
        weight_b_safe[weight_b_safe == 0] = 1.0
        weight_f_safe = weight_f.clone()
        weight_f_safe[weight_f_safe == 0] = 1.0

        sum_b = torch.cumsum(hist * torch.arange(256, device=hist.device), dim=0)
        sum_f = sum_b[-1] - sum_b
        
        mean_b = sum_b / weight_b_safe
        mean_f = sum_f / weight_f_safe
        
        # Intra-class variance or Between-class variance
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        
        optimal_thresh_idx = torch.argmax(var_between).item()
        
        # Chuyển ngược ngưỡng về dải giá trị thực [p_min, p_max]
        otsu_thresh = float((optimal_thresh_idx / 255.0) * (p_max - p_min).item() + p_min.item())
        mean_thresh = p_tensor.mean().item()

        # 🛡️ CHỐT CHẶN 1 (ABCDFSS Appendix D): Ngưỡng phải tối thiểu bằng giá trị trung bình toàn ảnh
        thresh = max(otsu_thresh, mean_thresh)

        # Tạo mask nhị phân ban đầu
        bin_mask = (p_tensor >= thresh).float()

        # 🛡️ CHỐT CHẶN 2 (Sanity Guard): Ngăn chặn triệt để hiện tượng All-Ones thoái hóa
        fg_ratio = bin_mask.mean().item()
        
        # Nếu có thông tin từ Support, giới hạn diện tích Foreground tối đa của Query 
        # không vượt quá 2.0 lần diện tích Foreground của Support (Phòng thủ cực mạnh cho DeepGlobe)
        limit_fg_ratio = max_fg_ratio
        if support_fg_ratio is not None:
            limit_fg_ratio = min(max_fg_ratio, support_fg_ratio * 2.0 + 0.05)

        if fg_ratio > limit_fg_ratio:
            # Fallback quantile dựa trên limit_fg_ratio
            fallback_thresh = torch.quantile(p_tensor.view(-1), 1.0 - limit_fg_ratio).item()
            bin_mask = (p_tensor >= fallback_thresh).float()

        batch_masks.append(bin_mask.unsqueeze(0))

    return torch.stack(batch_masks, dim=0)

def compute_support_guided_threshold(
    support_prob_map: torch.Tensor,
    support_gt_mask: torch.Tensor,
    num_bins: int = 100
) -> float:
    """
    Tìm ngưỡng tối ưu trên ảnh Support (có ground truth) để tối đa hóa IoU.
    """
    s_prob = support_prob_map.view(-1)
    s_gt = support_gt_mask.view(-1)
    
    thresholds = torch.linspace(0.01, 0.99, steps=num_bins, device=s_prob.device)
    best_iou = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        pred = (s_prob >= thresh).float()
        intersection = (pred * s_gt).sum()
        union = pred.sum() + s_gt.sum() - intersection
        if union > 0:
            iou = (intersection / union).item()
            if iou > best_iou:
                best_iou = iou
                best_thresh = thresh.item()
                
    return best_thresh

if __name__ == "__main__":
    print("Testing compute_adaptive_threshold_mask...")
    dummy_prob = torch.rand(2, 1, 392, 392)
    mask = compute_adaptive_threshold_mask(dummy_prob)
    print(f"Mask Shape: {mask.shape}, FG ratio: {mask.mean().item():.4f}")
    assert mask.mean().item() <= 0.70, "Sanity check failed: FG ratio exceeds 70%!"
    print("All thresholding tests passed successfully!")
