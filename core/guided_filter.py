import torch
import torch.nn as nn
import torch.nn.functional as F

class FastGuidedFilter(nn.Module):
    """
    Bộ lọc làm mịn có hướng dẫn (Guided Filter) chạy hoàn toàn trên GPU bằng PyTorch native.
    Sử dụng ảnh gốc RGB làm guidance map để nắn và làm sắc nét ranh giới vật thể 
    từ bản đồ xác suất thô của DINOv2 mà không cần cài thêm pydensecrf.
    """
    def __init__(self, r: int = 4, eps: float = 1e-2):
        super().__init__()
        self.r = r
        self.eps = eps

    def forward(self, rgb_guidance: torch.Tensor, coarse_prob: torch.Tensor) -> torch.Tensor:
        """
        rgb_guidance: [B, 3, H, W] tensor ảnh RGB trong dải [0, 1]
        coarse_prob:  [B, 1, H, W] tensor xác suất thô trong dải [0, 1]
        
        Returns:
            refined_prob: [B, 1, H, W] xác suất sắc nét theo viền RGB
        """
        # Chuyển ảnh RGB sang grayscale luminance
        I = 0.299 * rgb_guidance[:, 0:1] + 0.587 * rgb_guidance[:, 1:2] + 0.114 * rgb_guidance[:, 2:3]
        p = coarse_prob

        k = 2 * self.r + 1
        mean_I = F.avg_pool2d(I, k, stride=1, padding=self.r)
        mean_p = F.avg_pool2d(p, k, stride=1, padding=self.r)
        mean_Ip = F.avg_pool2d(I * p, k, stride=1, padding=self.r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = F.avg_pool2d(I * I, k, stride=1, padding=self.r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = F.avg_pool2d(a, k, stride=1, padding=self.r)
        mean_b = F.avg_pool2d(b, k, stride=1, padding=self.r)

        q = mean_a * I + mean_b
        return q.clamp(0.0, 1.0)

if __name__ == "__main__":
    gf = FastGuidedFilter()
    dummy_rgb = torch.rand(1, 3, 392, 392)
    dummy_prob = torch.rand(1, 1, 392, 392)
    out = gf(dummy_rgb, dummy_prob)
    print("GuidedFilter output shape:", out.shape)
    print("Test passed successfully!")
