import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.dinov2_backbone import DINOv2Backbone
from models.crf_refinement import DenseCRFRefinement
from core.contrastive_head import ClassContrastiveAdapters
from core.attention import MultiLayerFusion
from core.thresholding import compute_adaptive_threshold_mask
from core.guided_filter import FastGuidedFilter
from core.evaluator import Evaluator
from data.transforms import RandomShearAugmentation
from data.dataset import FSS1000Dataset, DeepGlobeDataset, ISICDataset, SUIMDataset, LungDataset
from core.visualizer import save_prediction_visualization

def run_benchmark(
    num_episodes: int = 10,
    dataset_root: str = None,
    dataset_name: str = "fss",
    config_path: str = "config/default_config.yaml",
    use_meta_decoder: bool = False,
    use_crf: bool = False,
    use_guided_filter: bool = True,
    adapt_to: str = "none",  # 'first-episode', 'every-episode', 'none'
    img_size: int = None
):
    print("=" * 65)
    mode_name = "META-TRAINED DECODER" if use_meta_decoder else "MULTI-LAYER ZERO-SHOT"
    print(f"BẮT ĐẦU CHẠY BENCHMARK DINOv2 ({mode_name}) CHO {num_episodes} EPISODES")
    if dataset_root and os.path.exists(dataset_root):
        print(f"Nguồn dữ liệu: {dataset_root} (Loại: {dataset_name})")
    else:
        print("Nguồn dữ liệu: Giả lập (Synthetic Random Episodes)")
    print(f"Làm sắc nét viền (Guided Filter): {'BẬT' if use_guided_filter else 'TẮT'}")
    print(f"Chế độ Thích nghi (Adaptation): {adapt_to.upper()}")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if img_size is None:
        img_size = cfg["dataset"].get("img_size", 392)
    print(f"Kích thước ảnh xử lý: {img_size}x{img_size} (Feature Map: {img_size//14}x{img_size//14} tokens)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device}")

    # Bản đồ số lượng class chuẩn theo benchmark CD-FSS
    dataset_classes = {"deepglobe": 6, "isic": 3, "lung": 1, "suim": 7, "fss": 1000}
    num_classes = dataset_classes.get(dataset_name.lower(), 10)

    # Chuẩn bị Dataset thực tế nếu truyền dataset_root
    real_dataset = None
    if dataset_root and os.path.exists(dataset_root):
        try:
            if dataset_name.lower() == "fss":
                real_dataset = FSS1000Dataset(root_dir=dataset_root, num_episodes=num_episodes, img_size=img_size, split='test')
            elif dataset_name.lower() == "deepglobe":
                real_dataset = DeepGlobeDataset(root_dir=dataset_root, num_episodes=num_episodes, img_size=img_size)
            elif dataset_name.lower() == "isic":
                real_dataset = ISICDataset(root_dir=dataset_root, num_episodes=num_episodes, img_size=img_size)
            elif dataset_name.lower() == "suim":
                real_dataset = SUIMDataset(root_dir=dataset_root, num_episodes=num_episodes, img_size=img_size)
            elif dataset_name.lower() == "lung":
                real_dataset = LungDataset(root_dir=dataset_root, num_episodes=num_episodes, img_size=img_size)
            else:
                raise ValueError(f"Không hỗ trợ dataset_name: {dataset_name}")
                
            print(f"Đã khởi tạo thành công {real_dataset.__class__.__name__} với {len(real_dataset)} episodes thực tế!")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo Dataset ({e}), chuyển sang chế độ giả lập.")

    # Load Backbone (eval mode, frozen)
    backbone = DINOv2Backbone(
        backbone_name=cfg["model"]["backbone_name"],
        intermediate_layers=cfg["model"]["intermediate_layers"]
    ).to(device)

    augmentation = RandomShearAugmentation()
    guided_filter = FastGuidedFilter(r=4, eps=1e-2, blend_alpha=0.5).to(device)
    
    # Quản lý Adapter theo từng Class ID (chuẩn bài báo ABCDFSS)
    class_adapters = ClassContrastiveAdapters(
        num_classes=num_classes,
        num_layers=len(cfg["model"]["intermediate_layers"]),
        in_dim=backbone.embed_dim,
        out_dim=64,
        lr=cfg["tta"]["lr"],
        epochs=cfg["tta"]["epochs"]
    )

    fusion_module = MultiLayerFusion(
        layer_weights=[0.25, 0.25, 0.25, 0.25], 
        temperature=0.15,
        proto_ratio=0.5
    ).to(device)
    print("✅ Đang sử dụng chế độ: MULTI-LAYER DUAL CROSS-ATTENTION (FG/BG Density Matching)")
        
    crf_refiner = DenseCRFRefinement()

    overall_evaluator = Evaluator(num_classes=num_classes)
    episode_mious = []
    episode_fb_ious = []

    # Chuẩn hóa ảnh RGB phục vụ Guided Filter
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for ep in range(1, num_episodes + 1):
        # 1. Nạp dữ liệu 1-shot episode (Query & Support) từ Dataset thực hoặc Giả lập
        class_id = (ep - 1) % num_classes
        if real_dataset is not None and len(real_dataset) >= ep:
            data = real_dataset[ep - 1]
            query_img = data["query_img"].unsqueeze(0).to(device)
            query_gt = data["query_mask"].unsqueeze(0).to(device)
            support_img = data["support_img"].unsqueeze(0).to(device)
            support_mask = data["support_mask"].unsqueeze(0).to(device)
            class_id = data.get("class_id", class_id)
        else:
            query_img = torch.randn(1, 3, img_size, img_size).to(device)
            query_gt = (torch.rand(1, 1, img_size, img_size) > 0.6).float().to(device)
            support_img = torch.randn(1, 3, img_size, img_size).to(device)
            support_mask = (torch.rand(1, 1, img_size, img_size) > 0.6).float().to(device)

        # 2. Trích xuất đặc trưng Backbone
        with torch.no_grad():
            q_feats = backbone(query_img)
            s_feats = backbone(support_img)

        # 3. Thích nghi theo lớp (Class-Wise Adaptation) nếu bật
        final_loss_val = 0.0
        if adapt_to != "none":
            need_fit = False
            if adapt_to == "first-episode":
                need_fit = not class_adapters.has_fitted(class_id)
            elif adapt_to == "every-episode":
                need_fit = True

            if need_fit:
                query_aug = augmentation(query_img)
                support_aug, support_mask_aug = augmentation(support_img, support_mask)

                with torch.no_grad():
                    q_aug_feats = backbone(query_aug)
                    s_aug_feats = backbone(support_aug)

                final_loss_val = class_adapters.fit_class(
                    class_id=class_id,
                    q_feats=q_feats,
                    q_aug_feats=q_aug_feats,
                    s_feats=s_feats,
                    s_aug_feats=s_aug_feats,
                    s_mask=support_mask,
                    s_aug_mask=support_mask_aug,
                    device=device
                )

            # Chiếu đặc trưng qua Adapter của class_id
            q_final = class_adapters.transform(class_id, q_feats, device)
            s_final = class_adapters.transform(class_id, s_feats, device)
        else:
            # Thuần Zero-Shot
            q_final = q_feats
            s_final = s_feats

        # 4. Inference & Evaluate
        with torch.no_grad():
            pred_prob_map = fusion_module(q_final, s_final, support_mask, target_size=(img_size, img_size))
            
            # Làm sắc nét viền vật thể bằng Fast Guided Filter
            if use_guided_filter:
                rgb_raw = (query_img * std_t + mean_t).clamp(0.0, 1.0)
                refined_prob = guided_filter(rgb_raw, pred_prob_map)
            else:
                refined_prob = pred_prob_map

            # 🛡️ Dynamic Adaptive Thresholding (Otsu + Mean Safeguard + Sanity Guard)
            pred_bin_mask = compute_adaptive_threshold_mask(refined_prob)

            if use_crf:
                img_np = (query_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                prob_np = refined_prob[0, 0].cpu().numpy()
                crf_mask_np = crf_refiner(img_np, prob_np)
                eval_mask = torch.from_numpy(crf_mask_np).unsqueeze(0).unsqueeze(0).to(device)
            else:
                eval_mask = pred_bin_mask

            ep_evaluator = Evaluator(num_classes=num_classes)
            ep_evaluator.update(eval_mask, query_gt, class_id=class_id)
            overall_evaluator.update(eval_mask, query_gt, class_id=class_id)
            
            ep_res = ep_evaluator.compute()
            episode_mious.append(ep_res["IoU_FG"])
            episode_fb_ious.append(ep_res["FB-IoU"])

        # Trực quan hóa cho Episode 1
        if ep == 1:
            save_prediction_visualization(
                query_img, query_gt, pred_prob_map, eval_mask,
                save_path="experiments/sample_episode_viz.png"
            )

        print(f"Episode [{ep:02d}/{num_episodes:02d}] (Class {class_id}) - FG-IoU: {ep_res['IoU_FG']*100:.2f}% | FB-IoU: {ep_res['FB-IoU']*100:.2f}%")

    overall_res = overall_evaluator.compute()
    mean_ep_miou = np.mean(episode_mious) * 100
    std_ep_miou = np.std(episode_mious) * 100
    mean_fb_iou = np.mean(episode_fb_ious) * 100

    print("\n" + "=" * 65)
    print("KẾT QUẢ BENCHMARK TỔNG HỢP:")
    print(f"  - mIoU Chuẩn Benchmark (Class-wise): {overall_res['mIoU'] * 100:.2f}%")
    print(f"  - Trung bình mIoU theo Episode:       {mean_ep_miou:.2f}% (+/- {std_ep_miou:.2f}%)")
    print(f"  - Trung bình FB-IoU:                  {mean_fb_iou:.2f}%")
    print(f"  - IoU Foreground Tích lũy:            {overall_res['IoU_FG'] * 100:.2f}%")
    print(f"  - IoU Background Tích lũy:            {overall_res['IoU_BG'] * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Benchmark for CD-FSS DINOv2")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episodes thử nghiệm")
    parser.add_argument("--dataset_root", type=str, default=None, help="Đường dẫn root tới dataset")
    parser.add_argument("--dataset_name", type=str, default="fss", choices=["fss", "deepglobe", "isic", "suim", "lung"], help="Tên bộ dữ liệu")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Đường dẫn file config")
    parser.add_argument("--use_meta_decoder", action="store_true", help="Bật cờ này để dùng MetaDecoder, nếu không sẽ dùng Zero-Shot")
    parser.add_argument("--use_crf", action="store_true", help="Bật cờ này để dùng hậu xử lý CRF")
    parser.add_argument("--no_guided_filter", action="store_true", help="Tắt Guided Filter")
    parser.add_argument("--adapt_to", type=str, default="none", choices=["first-episode", "every-episode", "none"], help="Cơ chế thích nghi")
    parser.add_argument("--img_size", type=int, default=None, help="Kích thước ảnh (vd: 392 hoặc 518)")
    args = parser.parse_args()

    run_benchmark(
        num_episodes=args.episodes, 
        dataset_root=args.dataset_root, 
        dataset_name=args.dataset_name,
        config_path=args.config,
        use_meta_decoder=args.use_meta_decoder,
        use_crf=args.use_crf,
        use_guided_filter=not args.no_guided_filter,
        adapt_to=args.adapt_to,
        img_size=args.img_size
    )
