import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class RandomShearAugmentation:
    """
    Tăng cường dữ liệu bằng Random Shearing cho TTA theo bài báo gốc ABCDFSS.
    """
    def __init__(self, shear_range: tuple = (-15, 15)):
        self.shear_range = shear_range

    def __call__(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor = None):
        """
        img_tensor:  [B, C, H, W]
        mask_tensor: [B, 1, H, W] (nếu có)
        """
        # Để đồng bộ góc shear ngẫu nhiên giữa ảnh và mask
        angle, translations, scale, shear = T.RandomAffine.get_params(
            degrees=[0, 0], translate=None, scale_ranges=None, shears=self.shear_range, img_size=img_tensor.shape[-2:]
        )

        aug_img = TF.affine(img_tensor, angle=angle, translate=translations, scale=scale, shear=shear)
        if mask_tensor is not None:
            aug_mask = TF.affine(mask_tensor, angle=angle, translate=translations, scale=scale, shear=shear, interpolation=T.InterpolationMode.NEAREST)
            return aug_img, aug_mask
        return aug_img

if __name__ == "__main__":
    print("Testing RandomShearAugmentation...")
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_mask = (torch.rand(1, 1, 224, 224) > 0.5).float()
    aug = RandomShearAugmentation()
    aug_img, aug_mask = aug(dummy_img, dummy_mask)
    print(f"Augmented Image Shape: {aug_img.shape}")
    print(f"Augmented Mask Shape: {aug_mask.shape}")
    print("Transforms test passed successfully!")
