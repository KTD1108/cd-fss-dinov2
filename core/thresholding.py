import torch
import numpy as np
import cv2

def compute_adaptive_threshold_mask(
    prob_map: torch.Tensor,
    drop_least: float = 0.05,
    max_fg_ratio: float = 0.70
) -> torch.Tensor:
    """
    Phân ngưỡng thích ứng dựa trên thuật toán Otsu kết hợp chốt chặn an toàn (Safeguard) 
    theo chuẩn bài báo ABCDFSS (Appendix D) và cơ chế chống sập ngưỡng thoái hóa.

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
        p_np = prob_map[i, 0].detach().cpu().numpy().astype(np.float32)
        p_min, p_max = float(p_np.min()), float(p_np.max())

        # Nếu bản đồ xác suất hoàn toàn đồng nhất
        if p_max - p_min < 1e-5:
            batch_masks.append(torch.zeros((1, H, W), device=prob_map.device))
            continue

        # Chuẩn hóa về [0, 255] để áp dụng Otsu
        p_norm = (p_np - p_min) / (p_max - p_min + 1e-8)
        norm_uint8 = (p_norm * 255.0).astype(np.uint8)

        # Loại bỏ các giá trị nhiễu ở percentile thấp nhất (tương tự segutils.py trong ABCDFSS)
        lower_bound = int(255 * drop_least)
        valid_vals = norm_uint8[norm_uint8 >= lower_bound]

        if len(valid_vals) == 0:
            valid_vals = norm_uint8

        thresh_val, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Chuyển ngược ngưỡng về dải giá trị thực [p_min, p_max]
        otsu_thresh = float((thresh_val / 255.0) * (p_max - p_min) + p_min)
        mean_thresh = float(np.mean(p_np))

        # 🛡️ CHỐT CHẶN 1 (ABCDFSS Appendix D): Ngưỡng phải tối thiểu bằng giá trị trung bình toàn ảnh
        # Thresh = max(otsu, mean) ngăn không cho background bị ngộ nhận thành foreground
        thresh = max(otsu_thresh, mean_thresh)

        # Tạo mask nhị phân ban đầu
        bin_mask = (p_np >= thresh).astype(np.float32)

        # 🛡️ CHỐT CHẶN 2 (Sanity Guard): Ngăn chặn triệt để hiện tượng All-Ones thoái hóa
        # Trong FSS, hiếm khi 1 lớp đối tượng chiếm > 70% toàn bộ khung hình
        fg_ratio = float(np.mean(bin_mask))
        if fg_ratio > max_fg_ratio:
            # Nếu tỷ lệ FG quá cao, kích hoạt fallback: chỉ giữ lại top 25% giá trị kích hoạt mạnh nhất
            fallback_thresh = float(np.percentile(p_np, 75))
            bin_mask = (p_np >= fallback_thresh).astype(np.float32)

        batch_masks.append(torch.from_numpy(bin_mask).unsqueeze(0).to(prob_map.device))

    return torch.stack(batch_masks, dim=0)

if __name__ == "__main__":
    print("Testing compute_adaptive_threshold_mask...")
    dummy_prob = torch.rand(2, 1, 392, 392)
    mask = compute_adaptive_threshold_mask(dummy_prob)
    print(f"Mask Shape: {mask.shape}, FG ratio: {mask.mean().item():.4f}")
    assert mask.mean().item() <= 0.70, "Sanity check failed: FG ratio exceeds 70%!"
    print("All thresholding tests passed successfully!")
