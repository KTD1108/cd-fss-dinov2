import torch
import numpy as np
import cv2
import os

def save_prediction_visualization(
    query_img_tensor: torch.Tensor,
    gt_mask_tensor: torch.Tensor,
    pred_prob_map: torch.Tensor,
    crf_mask_tensor: torch.Tensor,
    save_path: str = "experiments/results_visualization.png"
):
    """
    Trực quan hóa và lưu lại so sánh: Ảnh Query | Ground Truth | Prediction Prob | DenseCRF Mask
    """
    # Inverse Normalize Image
    img = query_img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    if img.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert masks to uint8
    gt_mask = (gt_mask_tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    prob_map = (pred_prob_map[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    crf_mask = (crf_mask_tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

    # Color maps for visualization
    gt_color = cv2.applyColorMap(gt_mask, cv2.COLORMAP_JET)
    prob_color = cv2.applyColorMap(prob_map, cv2.COLORMAP_VIRIDIS)
    crf_color = cv2.applyColorMap(crf_mask, cv2.COLORMAP_JET)

    # Create side-by-side panel
    H, W, _ = img_bgr.shape
    panel = np.zeros((H, W * 4, 3), dtype=np.uint8)
    panel[:, :W] = img_bgr
    panel[:, W:2*W] = gt_color
    panel[:, 2*W:3*W] = prob_color
    panel[:, 3*W:] = crf_color

    # Add labels
    cv2.putText(panel, "Query Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "Ground Truth", (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "Pred Heatmap", (2*W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "DenseCRF Result", (3*W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, panel)
    print(f"Lưu kết quả trực quan tại: {save_path}")
    return save_path
