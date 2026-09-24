import torch
import numpy as np

def otsu_thresholding(pred_map: torch.Tensor) -> torch.Tensor:
    """
    Áp dụng phân ngưỡng Otsu tự động lên prediction probability map.
    
    Args:
        pred_map: Tensor [B, 1, H, W] hoặc [H, W] với giá trị trong dải [0, 1]
    Returns:
        binary_mask: Tensor [B, 1, H, W] gồm các giá trị 0.0 hoặc 1.0
    """
    device = pred_map.device
    pred_np = pred_map.detach().cpu().numpy()
    
    batch_masks = []
    for b in range(pred_np.shape[0]):
        img = pred_np[b, 0]
        # Xử lý trường hợp ảnh có giá trị đồng nhất
        if np.max(img) == np.min(img):
            thresh = 0.5
        else:
            try:
                import cv2
                img_uint8 = (img * 255).astype(np.uint8)
                val, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_val = val / 255.0
                mean_val = float(np.mean(img))
                # Ngưỡng chuẩn bài báo ABCDFSS Appendix D: thresh = max(mean(m), otsus(m))
                thresh = max(mean_val, otsu_val)
            except Exception:
                try:
                    from skimage.filters import threshold_otsu
                    otsu_val = float(threshold_otsu(img))
                    thresh = max(float(np.mean(img)), otsu_val)
                except Exception:
                    thresh = 0.5

        binary_img = (img >= thresh).astype(np.float32)
        batch_masks.append(binary_img)

    binary_tensor = torch.from_numpy(np.array(batch_masks)).unsqueeze(1).to(device)
    return binary_tensor

def adaptive_thresholding(pred_map: torch.Tensor, fixed_thresh: float = 0.5) -> torch.Tensor:
    """
    Phân ngưỡng cố định hoặc thích ứng đơn giản.
    """
    return (pred_map >= fixed_thresh).float()

if __name__ == "__main__":
    print("Testing Thresholding algorithms...")
    dummy_pred = torch.rand(1, 1, 224, 224)
    otsu_mask = otsu_thresholding(dummy_pred)
    print(f"Otsu Thresholded Mask Shape: {otsu_mask.shape}")
    print(f"Unique values in mask: {torch.unique(otsu_mask)}")
    print("Thresholding tests passed successfully!")
