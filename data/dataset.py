import torch
from torch.utils.data import Dataset
import numpy as np

class CDFSSEpisodeDataset(Dataset):
    """
    Dataset loader cho các bài toán Few-Shot Segmentation (1-shot episode).
    Mỗi episode gồm 1 ảnh Query (và Query Mask nếu đánh giá) + 1 ảnh Support + Support Mask.
    """
    def __init__(self, query_images, query_masks, support_images, support_masks, img_size: int = 224):
        self.query_images = query_images
        self.query_masks = query_masks
        self.support_images = support_images
        self.support_masks = support_masks
        self.img_size = img_size

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, idx):
        return {
            "query_img": self.query_images[idx],      # [3, H, W]
            "query_mask": self.query_masks[idx],    # [1, H, W]
            "support_img": self.support_images[idx],  # [3, H, W]
            "support_mask": self.support_masks[idx]  # [1, H, W]
        }

class FolderEpisodeDataset(Dataset):
    """
    Dataset loader nạp danh sách đường dẫn ảnh thực tế và mask thực tế từ đĩa.
    """
    def __init__(self, query_img_paths, query_mask_paths, support_img_paths, support_mask_paths, img_size: int = 224):
        import torchvision.transforms as T
        from PIL import Image

        self.query_img_paths = query_img_paths
        self.query_mask_paths = query_mask_paths
        self.support_img_paths = support_img_paths
        self.support_mask_paths = support_mask_paths
        self.img_size = img_size

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.query_img_paths)

    def _load_img(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform_img(img)

    def _load_mask(self, path):
        from PIL import Image
        mask = Image.open(path).convert("L")
        mask_t = self.transform_mask(mask)
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        return {
            "query_img": self._load_img(self.query_img_paths[idx]),
            "query_mask": self._load_mask(self.query_mask_paths[idx]),
            "support_img": self._load_img(self.support_img_paths[idx]),
            "support_mask": self._load_mask(self.support_mask_paths[idx])
        }

class FSS1000Dataset(Dataset):
    """
    Dataset loader chuyên dụng cho tập dữ liệu FSS-1000.
    Tự động quét các class folder (ví dụ: accordion, acorn...) và sample các episodes.
    """
    def __init__(self, root_dir: str, num_episodes: int = 100, img_size: int = 224, seed: int = 42, split: str = 'test'):
        import os
        import glob
        import random
        import torchvision.transforms as T
        from PIL import Image

        self.root_dir = root_dir
        self.num_episodes = num_episodes
        self.img_size = img_size

        # Lấy danh sách các class folders
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if len(all_classes) == 0:
            raise ValueError(f"Không tìm thấy class folder nào tại: {root_dir}")

        # Phân chia theo split (FSS-1000 chuẩn: 520 train, 240 val, 240 test)
        if len(all_classes) >= 1000:
            if split == 'train':
                self.classes = all_classes[:520]
            elif split == 'val':
                self.classes = all_classes[520:760]
            elif split == 'test':
                self.classes = all_classes[760:1000]
            else:
                self.classes = all_classes # all
        else:
            self.classes = all_classes

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        # Pre-sample các episodes cố định theo seed để kết quả đếm lại khớp 100%
        rng = random.Random(seed)
        self.episodes = []
        for _ in range(num_episodes):
            cls_name = rng.choice(self.classes)
            cls_dir = os.path.join(root_dir, cls_name)

            # Lấy tất cả các file .jpg/.png ảnh trong class
            img_files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png")))
            # Chỉ lọc lấy ảnh gốc (loại trừ các file mask trùng đuôi nếu có)
            valid_pairs = []
            for img_path in img_files:
                base, ext = os.path.splitext(img_path)
                mask_path = base + ".png" if ext != ".png" else base + ".png"
                if not os.path.exists(mask_path):
                    # thử mask với suffix khác nếu có
                    continue
                # Giả định FSS-1000: N.jpg và N.png
                if ext.lower() in ['.jpg', '.jpeg']:
                    mask_candidate = base + '.png'
                    if os.path.exists(mask_candidate):
                        valid_pairs.append((img_path, mask_candidate))

            if len(valid_pairs) < 2:
                continue

            q_idx, s_idx = rng.sample(range(len(valid_pairs)), 2)
            self.episodes.append((valid_pairs[q_idx], valid_pairs[s_idx]))

    def __len__(self):
        return len(self.episodes)

    def _load_img(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform_img(img)

    def _load_mask(self, path):
        from PIL import Image
        mask = Image.open(path).convert("L")
        mask_t = self.transform_mask(mask)
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        (q_img_path, q_mask_path), (s_img_path, s_mask_path) = self.episodes[idx]
        return {
            "query_img": self._load_img(q_img_path),
            "query_mask": self._load_mask(q_mask_path),
            "support_img": self._load_img(s_img_path),
            "support_mask": self._load_mask(s_mask_path)
        }

class ISICDataset(Dataset):
    """
    Dataset loader cho ảnh y tế ISIC (Skin Lesion).
    """
    def __init__(self, root_dir: str, num_episodes: int = 100, img_size: int = 224, seed: int = 42):
        import os, glob, random
        import torchvision.transforms as T
        from PIL import Image

        self.root_dir = root_dir
        self.categories = ['1', '2', '3']
        self.img_size = img_size

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        self.img_metadata = {}
        for cat in self.categories:
            img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'ISIC2018_Task1-2_Training_Input', cat, '*.jpg')))
            valid_pairs = []
            for img_path in img_paths:
                img_name = os.path.basename(img_path).replace(".jpg", "")
                mask_path = os.path.join(self.root_dir, 'ISIC2018_Task1_Training_GroundTruth', img_name + '_segmentation.png')
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            if len(valid_pairs) >= 2:
                self.img_metadata[cat] = valid_pairs

        rng = random.Random(seed)
        self.episodes = []
        available_cats = list(self.img_metadata.keys())
        if not available_cats:
            print(f"Warning: Không tìm thấy dữ liệu ISIC tại {root_dir}")
            return
            
        for _ in range(num_episodes):
            cat = rng.choice(available_cats)
            pairs = self.img_metadata[cat]
            q_idx, s_idx = rng.sample(range(len(pairs)), 2)
            self.episodes.append((pairs[q_idx], pairs[s_idx]))

    def __len__(self):
        return len(self.episodes)

    def _load_img(self, path):
        from PIL import Image
        return self.transform_img(Image.open(path).convert("RGB"))

    def _load_mask(self, path):
        from PIL import Image
        mask_t = self.transform_mask(Image.open(path).convert("L"))
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        (q_img_path, q_mask_path), (s_img_path, s_mask_path) = self.episodes[idx]
        return {
            "query_img": self._load_img(q_img_path),
            "query_mask": self._load_mask(q_mask_path),
            "support_img": self._load_img(s_img_path),
            "support_mask": self._load_mask(s_mask_path)
        }

class SUIMDataset(Dataset):
    """
    Dataset loader cho ảnh dưới nước SUIM.
    """
    def __init__(self, root_dir: str, num_episodes: int = 100, img_size: int = 224, seed: int = 42):
        import os, glob, random
        import torchvision.transforms as T
        from PIL import Image

        self.root_dir = root_dir
        self.categories = ['FV','HD','PF','RI','RO','SR','WR']
        self.img_size = img_size

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        self.img_metadata = {}
        for cat in self.categories:
            mask_paths = sorted(glob.glob(os.path.join(self.root_dir, 'masks', cat, '*.*')))
            valid_pairs = []
            for mask_path in mask_paths:
                img_name = os.path.basename(mask_path).split('.')[0] + '.jpg'
                img_path = os.path.join(self.root_dir, 'images', img_name)
                if os.path.exists(img_path):
                    valid_pairs.append((img_path, mask_path))
            if len(valid_pairs) >= 2:
                self.img_metadata[cat] = valid_pairs

        rng = random.Random(seed)
        self.episodes = []
        available_cats = list(self.img_metadata.keys())
        if not available_cats:
            print(f"Warning: Không tìm thấy dữ liệu SUIM tại {root_dir}")
            return
            
        for _ in range(num_episodes):
            cat = rng.choice(available_cats)
            pairs = self.img_metadata[cat]
            q_idx, s_idx = rng.sample(range(len(pairs)), 2)
            self.episodes.append((pairs[q_idx], pairs[s_idx]))

    def __len__(self):
        return len(self.episodes)

    def _load_img(self, path):
        from PIL import Image
        return self.transform_img(Image.open(path).convert("RGB"))

    def _load_mask(self, path):
        from PIL import Image
        mask_t = self.transform_mask(Image.open(path).convert("L"))
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        (q_img_path, q_mask_path), (s_img_path, s_mask_path) = self.episodes[idx]
        return {
            "query_img": self._load_img(q_img_path),
            "query_mask": self._load_mask(q_mask_path),
            "support_img": self._load_img(s_img_path),
            "support_mask": self._load_mask(s_mask_path)
        }

class LungDataset(Dataset):
    """
    Dataset loader cho ảnh X-Quang Phổi (Chest X-ray / Lung).
    """
    def __init__(self, root_dir: str, num_episodes: int = 100, img_size: int = 224, seed: int = 42):
        import os, glob, random
        import torchvision.transforms as T
        from PIL import Image

        self.root_dir = root_dir
        self.img_size = img_size

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        mask_paths = sorted(glob.glob(os.path.join(self.root_dir, 'masks', '*.png')))
        valid_pairs = []
        for mask_path in mask_paths:
            # abc_mask.png -> abc.png
            img_name = os.path.basename(mask_path)[:-9] + '.png' 
            img_path = os.path.join(self.root_dir, 'CXR_png', img_name)
            if os.path.exists(img_path):
                valid_pairs.append((img_path, mask_path))

        rng = random.Random(seed)
        self.episodes = []
        if len(valid_pairs) < 2:
            print(f"Warning: Không tìm thấy dữ liệu Lung tại {root_dir}")
            return
            
        for _ in range(num_episodes):
            q_idx, s_idx = rng.sample(range(len(valid_pairs)), 2)
            self.episodes.append((valid_pairs[q_idx], valid_pairs[s_idx]))

    def __len__(self):
        return len(self.episodes)

    def _load_img(self, path):
        from PIL import Image
        return self.transform_img(Image.open(path).convert("RGB"))

    def _load_mask(self, path):
        from PIL import Image
        mask_t = self.transform_mask(Image.open(path).convert("L"))
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        (q_img_path, q_mask_path), (s_img_path, s_mask_path) = self.episodes[idx]
        return {
            "query_img": self._load_img(q_img_path),
            "query_mask": self._load_mask(q_mask_path),
            "support_img": self._load_img(s_img_path),
            "support_mask": self._load_mask(s_mask_path)
        }

class DeepGlobeDataset(Dataset):
    """
    Dataset loader cho ảnh vệ tinh DeepGlobe.
    """
    def __init__(self, root_dir: str, num_episodes: int = 100, img_size: int = 224, seed: int = 42):
        import os, glob, random
        import torchvision.transforms as T
        from PIL import Image

        self.root_dir = root_dir
        self.img_size = img_size
        self.categories = ['1', '2', '3', '4', '5', '6']

        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        self.img_metadata = {}
        for cat in self.categories:
            origin_dir = os.path.join(self.root_dir, cat, 'test', 'origin')
            gt_dir = os.path.join(self.root_dir, cat, 'test', 'groundtruth')
            
            if not os.path.exists(origin_dir):
                continue
                
            img_paths = sorted(glob.glob(os.path.join(origin_dir, "*.jpg")))
            valid_pairs = []
            for img_path in img_paths:
                img_name = os.path.basename(img_path).replace(".jpg", ".png")
                mask_path = os.path.join(gt_dir, img_name)
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path))
            
            if len(valid_pairs) >= 2:
                self.img_metadata[cat] = valid_pairs

        rng = random.Random(seed)
        self.episodes = []
        available_cats = list(self.img_metadata.keys())
        if not available_cats:
            print(f"Warning: Không tìm thấy dữ liệu DeepGlobe tại {root_dir}")
            return
            
        for _ in range(num_episodes):
            cat = rng.choice(available_cats)
            pairs = self.img_metadata[cat]
            q_idx, s_idx = rng.sample(range(len(pairs)), 2)
            self.episodes.append((pairs[q_idx], pairs[s_idx]))

    def __len__(self):
        return len(self.episodes)

    def _load_img(self, path):
        from PIL import Image
        return self.transform_img(Image.open(path).convert("RGB"))

    def _load_mask(self, path):
        from PIL import Image
        mask_t = self.transform_mask(Image.open(path).convert("L"))
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        (q_img_path, q_mask_path), (s_img_path, s_mask_path) = self.episodes[idx]
        return {
            "query_img": self._load_img(q_img_path),
            "query_mask": self._load_mask(q_mask_path),
            "support_img": self._load_img(s_img_path),
            "support_mask": self._load_mask(s_mask_path)
        }

