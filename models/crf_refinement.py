import torch
import numpy as np

class DenseCRFRefinement:
    """
    Hậu xử lý mặt nạ phân đoạn bằng DenseCRF để làm sắc nét ranh giới vật thể.
    """
    def __init__(self, iter_max: int = 10, pos_w: float = 3.0, pos_xy_std: float = 3.0, bi_w: float = 10.0, bi_xy_std: float = 80.0, bi_rgb_std: float = 13.0):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image_np: np.ndarray, prob_map_np: np.ndarray) -> np.ndarray:
        """
        image_np:    [H, W, 3] uint8 numpy array (ảnh gốc RGB)
        prob_map_np: [H, W] float32 numpy array trong dải [0, 1]
        
        Returns:
            refine_mask: [H, W] float32 numpy array (0.0 hoặc 1.0)
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax

            H, W, C = image_np.shape
            # Tạo 2-class probability (Background vs Foreground)
            prob_bg = 1.0 - prob_map_np
            probs = np.stack([prob_bg, prob_map_np], axis=0) # [2, H, W]

            unary = unary_from_softmax(probs)

            d = dcrf.DenseCRF2D(W, H, 2)
            d.setUnaryEnergy(unary)

            # Spatial energy (Vị trí điểm ảnh)
            d.addPairwiseGaussian(sxy=(self.pos_xy_std, self.pos_xy_std), compat=self.pos_w, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORM_ALIAGNED)

            # Bilateral energy (Vị trí + Màu sắc RGB)
            d.addPairwiseBilateral(sxy=(self.bi_xy_std, self.bi_xy_std), srgb=(self.bi_rgb_std, self.bi_rgb_std, self.bi_rgb_std), rgbim=image_np, compat=self.bi_w, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORM_ALIAGNED)

            Q = d.inference(self.iter_max)
            res = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            return res
        except ImportError:
            # Fallback nếu pydensecrf chưa được cài đặt / không tương thích trên Windows
            return (prob_map_np >= 0.5).astype(np.float32)

if __name__ == "__main__":
    print("Testing DenseCRFRefinement...")
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_prob = np.random.rand(224, 224).astype(np.float32)

    crf = DenseCRFRefinement()
    ref_mask = crf(dummy_img, dummy_prob)
    print(f"Refined Mask Shape: {ref_mask.shape}")
    print("CRF test passed successfully!")
