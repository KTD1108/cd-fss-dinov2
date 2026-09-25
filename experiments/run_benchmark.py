import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.dinov2_backbone import DINOv2Backbone
from models.adapter import MultiLevelAdapters
from models.crf_refinement import DenseCRFRefinement
from core.losses import DenseContrastiveLoss, FeatureStatLoss, PrototypeAlignmentLoss
from core.attention import MetaDecoder, MultiLayerFusion
from core.thresholding import otsu_thresholding
from core.evaluator import Evaluator
from data.transforms import RandomShearAugmentation
from data.dataset import FSS1000Dataset, DeepGlobeDataset, ISICDataset, SUIMDataset, LungDataset
from core.visualizer import save_prediction_visualization

def run_benchmark(num_episodes: int = 10, dataset_root: str = None, dataset_name: str = "fss", config_path: str = "config/default_config.yaml", use_meta_decoder: bool = False):
    print("=" * 65)
    mode_name = "META-TRAINED DECODER" if use_meta_decoder else "ZERO-SHOT (TRAINING-FREE)"
    print(f"BẮT ĐẦU CHẠY BENCHMARK DINOv2 ({mode_name}) CHO {num_episodes} EPISODES")
    if dataset_root and os.path.exists(dataset_root):
        print(f"Nguồn dữ liệu: {dataset_root} (Loại: {dataset_name})")
    else:
        print("Nguồn dữ liệu: Giả lập (Synthetic Random Episodes)")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device}")

    # Chuẩn bị Dataset thực tế nếu truyền dataset_root
    real_dataset = None
    if dataset_root and os.path.exists(dataset_root):
        try:
            img_size = cfg["dataset"]["img_size"]
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

    l_nce = DenseContrastiveLoss().to(device)
    l_stat = FeatureStatLoss().to(device)
    l_p = PrototypeAlignmentLoss().to(device)
    augmentation = RandomShearAugmentation()
    
    if use_meta_decoder:
        # 🌟 Sử dụng MetaDecoder (có trọng số)
        fusion_module = MetaDecoder(in_channels=384, num_layers=4).to(device)
        weight_path = "models/weights/meta_decoder_best.pth"
        if os.path.exists(weight_path):
            fusion_module.load_state_dict(torch.load(weight_path, map_location=device))
            print("✅ Đang sử dụng chế độ: META-DECODER (Đã nạp trọng số học thuật)")
        else:
            print("⚠️ CẢNH BÁO: Không tìm thấy trọng số MetaDecoder. Đang chạy với trọng số ngẫu nhiên!")
        fusion_module.eval() # Khóa trọng số trong lúc Test-Time Adaptation
    else:
        # 🛡️ Sử dụng MultiLayerFusion (Không trọng số - Tốt cho Cross-Domain)
        fusion_module = MultiLayerFusion().to(device)
        print("✅ Đang sử dụng chế độ: ZERO-SHOT (MultiLayerFusion - Kháng Domain Shift cực tốt)")
        
    crf_refiner = DenseCRFRefinement()

    overall_evaluator = Evaluator()
    episode_mious = []
    episode_fb_ious = []

    for ep in range(1, num_episodes + 1):
        # 1. Reset Adapters cho mỗi Episode mới
        adapters = MultiLevelAdapters(
            num_levels=len(cfg["model"]["intermediate_layers"]),
            in_dim=backbone.embed_dim,
            adapter_dim=cfg["model"]["adapter_channels"]
        ).to(device)

        optimizer = optim.SGD(
            adapters.parameters(),
            lr=cfg["tta"]["lr"],
            momentum=cfg["tta"]["momentum"],
            weight_decay=cfg["tta"]["weight_decay"]
        )

        # 2. Nạp dữ liệu 1-shot episode (Query & Support) từ Dataset thực hoặc Giả lập
        if real_dataset is not None and len(real_dataset) >= ep:
            data = real_dataset[ep - 1]
            query_img = data["query_img"].unsqueeze(0).to(device)
            query_gt = data["query_mask"].unsqueeze(0).to(device)
            support_img = data["support_img"].unsqueeze(0).to(device)
            support_mask = data["support_mask"].unsqueeze(0).to(device)
        else:
            img_size = cfg["dataset"]["img_size"]
            query_img = torch.randn(1, 3, img_size, img_size).to(device)
            query_gt = (torch.rand(1, 1, img_size, img_size) > 0.6).float().to(device)
            support_img = torch.randn(1, 3, img_size, img_size).to(device)
            support_mask = (torch.rand(1, 1, img_size, img_size) > 0.6).float().to(device)

        # 3. Tạo Nhãn Giả (Pseudo-Label) bằng Zero-Shot trước khi TTA
        adapters.eval()
        with torch.no_grad():
            img_size = cfg["dataset"]["img_size"]
            q_raw = adapters(backbone(query_img))
            s_raw = adapters(backbone(support_img))
            pseudo_pred = fusion_module(q_raw, s_raw, support_mask, target_size=(img_size, img_size))
            
            # Lọc các pixel chắc chắn (Confident Pixels)
            pseudo_mask = torch.zeros_like(pseudo_pred)
            pseudo_mask[pseudo_pred > 0.6] = 1.0
            pseudo_mask[pseudo_pred < 0.4] = 0.0
            valid_pixels = ((pseudo_pred > 0.6) | (pseudo_pred < 0.4)).float()

        # 4. Test-Time Adaptation Loop với Self-Training
        adapters.train()
        for epoch in range(cfg["tta"]["epochs"]):
            optimizer.zero_grad()

            query_aug = augmentation(query_img)
            support_aug, support_mask_aug = augmentation(support_img, support_mask)

            q_feats = backbone(query_img)
            q_aug_feats = backbone(query_aug)
            s_feats = backbone(support_img)
            s_aug_feats = backbone(support_aug)

            q_adapted = adapters(q_feats)
            q_aug_adapted = adapters(q_aug_feats)
            s_adapted = adapters(s_feats)
            s_aug_adapted = adapters(s_aug_feats)

            total_loss = 0.0
            w_cfg = cfg["tta"]["loss_weights"]

            for l in range(len(q_adapted)):
                loss_nce_q = l_nce(q_adapted[l], q_aug_adapted[l])
                loss_nce_s = l_nce(s_adapted[l], s_aug_adapted[l])
                loss_stat_l = l_stat(q_adapted[l], q_aug_adapted[l]) + l_stat(s_adapted[l], s_aug_adapted[l])
                loss_p_l = l_p(s_adapted[l], s_aug_adapted[l], support_mask, support_mask_aug)

                total_loss += (w_cfg["w_nce_q"] * loss_nce_q +
                              w_cfg["w_nce_s"] * loss_nce_s +
                              w_cfg["w_stat"] * loss_stat_l +
                              w_cfg["w_p"] * loss_p_l)

            # Thêm Self-Attention Loss: Khả năng tự tái tạo lại Support Mask của Support Features
            # Thêm Self-Attention Loss: Khả năng tự tái tạo lại Support Mask của Support Features
            pred_s_self = fusion_module(s_adapted, s_adapted, support_mask, target_size=(img_size, img_size))
            loss_self_attn = torch.nn.functional.binary_cross_entropy(pred_s_self.clamp(1e-6, 1-1e-6), support_mask.float())
            total_loss += w_cfg.get("w_self_attn", 2.0) * loss_self_attn

            # 🔥 ĐÓNG GÓP MỚI: Pseudo-Label Self-Training Loss cho Query
            pred_q_pseudo = fusion_module(q_adapted, s_adapted, support_mask, target_size=(img_size, img_size))
            bce_loss = torch.nn.functional.binary_cross_entropy(pred_q_pseudo.clamp(1e-6, 1-1e-6), pseudo_mask, reduction='none')
            # Chỉ phạt trên các pixel mà model ban đầu đã rất tự tin
            loss_pseudo = (bce_loss * valid_pixels).sum() / (valid_pixels.sum() + 1e-8)
            total_loss += 5.0 * loss_pseudo  # Trọng số khá lớn để ép model học Query Domain

            total_loss.backward()
            optimizer.step()

        # 4. Inference & Evaluate
        adapters.eval()
        with torch.no_grad():
            q_feats_final = adapters(backbone(query_img))
            s_feats_final = adapters(backbone(support_img))

            img_size = cfg["dataset"]["img_size"]
            pred_prob_map = fusion_module(q_feats_final, s_feats_final, support_mask, target_size=(img_size, img_size))
            
            img_np = (query_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            prob_np = pred_prob_map[0, 0].cpu().numpy()
            crf_mask_np = crf_refiner(img_np, prob_np)
            crf_mask_tensor = torch.from_numpy(crf_mask_np).unsqueeze(0).unsqueeze(0).to(device)

            ep_evaluator = Evaluator()
            ep_evaluator.update(crf_mask_tensor, query_gt)
            overall_evaluator.update(crf_mask_tensor, query_gt)
            
            ep_res = ep_evaluator.compute()
            episode_mious.append(ep_res["mIoU"])
            episode_fb_ious.append(ep_res["FB-IoU"])

        # Trực quan hóa cho Episode 1
        if ep == 1:
            save_prediction_visualization(
                query_img, query_gt, pred_prob_map, crf_mask_tensor,
                save_path="experiments/sample_episode_viz.png"
            )

        print(f"Episode [{ep:02d}/{num_episodes:02d}] - mIoU: {ep_res['mIoU']*100:.2f}% | FB-IoU: {ep_res['FB-IoU']*100:.2f}% | Final Loss: {total_loss.item():.4f}")

    overall_res = overall_evaluator.compute()
    mean_miou = np.mean(episode_mious) * 100
    std_miou = np.std(episode_mious) * 100
    mean_fb_iou = np.mean(episode_fb_ious) * 100

    print("\n" + "=" * 65)
    print("KẾT QUẢ BENCHMARK TỔNG HỢP:")
    print(f"  - Trung bình mIoU:   {mean_miou:.2f}% (+/- {std_miou:.2f}%)")
    print(f"  - Trung bình FB-IoU: {mean_fb_iou:.2f}%")
    print(f"  - IoU Foreground:    {overall_res['IoU_FG'] * 100:.2f}%")
    print(f"  - IoU Background:    {overall_res['IoU_BG'] * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Benchmark for CD-FSS DINOv2")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episodes thử nghiệm")
    parser.add_argument("--dataset_root", type=str, default=None, help="Đường dẫn root tới dataset")
    parser.add_argument("--dataset_name", type=str, default="fss", choices=["fss", "deepglobe", "isic", "suim", "lung"], help="Tên bộ dữ liệu")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Đường dẫn file config")
    parser.add_argument("--use_meta_decoder", action="store_true", help="Bật cờ này để dùng MetaDecoder, nếu không sẽ dùng Zero-Shot")
    args = parser.parse_args()

    run_benchmark(
        num_episodes=args.episodes, 
        dataset_root=args.dataset_root, 
        dataset_name=args.dataset_name,
        config_path=args.config,
        use_meta_decoder=args.use_meta_decoder
    )

