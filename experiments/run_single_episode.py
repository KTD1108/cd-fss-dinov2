import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np

# Thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.dinov2_backbone import DINOv2Backbone
from models.adapter import MultiLevelAdapters
from models.crf_refinement import DenseCRFRefinement
from core.losses import DenseContrastiveLoss, FeatureStatLoss, PrototypeAlignmentLoss
from core.attention import MetaDecoder
from core.thresholding import otsu_thresholding
from core.evaluator import Evaluator
from data.transforms import RandomShearAugmentation

def run_single_episode(config_path: str = "config/default_config.yaml"):
    print("=" * 60)
    print("CHẠY THỬ NGHIỆM TTA VÀ DENSE CROSS-ATTENTION CHO 1 EPISODE")
    print("=" * 60)

    # 1. Nạp tệp cấu hình
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device đang sử dụng: {device}")

    # 2. Khởi tạo Mô hình & Adapter
    backbone = DINOv2Backbone(
        backbone_name=cfg["model"]["backbone_name"],
        intermediate_layers=cfg["model"]["intermediate_layers"]
    ).to(device)

    adapters = MultiLevelAdapters(
        num_levels=len(cfg["model"]["intermediate_layers"]),
        in_dim=backbone.embed_dim,
        adapter_dim=cfg["model"]["adapter_channels"]
    ).to(device)

    # 3. Khởi tạo Loss & Optimizer cho Adapter
    l_nce = DenseContrastiveLoss().to(device)
    l_stat = FeatureStatLoss().to(device)
    l_p = PrototypeAlignmentLoss().to(device)

    optimizer = optim.SGD(
        adapters.parameters(),
        lr=cfg["tta"]["lr"],
        momentum=cfg["tta"]["momentum"],
        weight_decay=cfg["tta"]["weight_decay"]
    )

    augmentation = RandomShearAugmentation()
    
    fusion_module = MetaDecoder(in_channels=384, num_layers=4).to(device)
    weight_path = "models/weights/meta_decoder_best.pth"
    if os.path.exists(weight_path):
        fusion_module.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ Đã tải thành công trọng số MetaDecoder từ quá trình Meta-Training!")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy trọng số MetaDecoder. Đang chạy với trọng số ngẫu nhiên!")
    fusion_module.eval()
    
    crf_refiner = DenseCRFRefinement()
    evaluator = Evaluator()

    # 4. Giả lập dữ liệu 1 Episode (Query & Support)
    print("\nGiả lập 1 cặp ảnh Query và Support (kích thước 224x224)...")
    query_img = torch.randn(1, 3, 224, 224).to(device)
    query_gt = (torch.rand(1, 1, 224, 224) > 0.6).float().to(device) # Ground Truth
    
    support_img = torch.randn(1, 3, 224, 224).to(device)
    support_mask = (torch.rand(1, 1, 224, 224) > 0.6).float().to(device)

    # 5. Vòng lặp Test-Time Adaptation (TTA) trong 25 epochs
    print(f"\nBắt đầu tối ưu TTA Adapter trong {cfg['tta']['epochs']} epochs...")
    adapters.train()
    
    for epoch in range(1, cfg["tta"]["epochs"] + 1):
        optimizer.zero_grad()

        # Tạo góc nhìn tăng cường (Augmented view) bằng Random Shearing
        query_aug = augmentation(query_img)
        support_aug, support_mask_aug = augmentation(support_img, support_mask)

        # Trích xuất DINOv2 features (Frozen)
        q_feats = backbone(query_img)
        q_aug_feats = backbone(query_aug)
        s_feats = backbone(support_img)
        s_aug_feats = backbone(support_aug)

        # Áp dụng Adapters
        q_adapted = adapters(q_feats)
        q_aug_adapted = adapters(q_aug_feats)
        s_adapted = adapters(s_feats)
        s_aug_adapted = adapters(s_aug_feats)

        # Tính tổng hợp các hàm loss TTA qua 4 tầng feature
        total_loss = 0.0
        w_cfg = cfg["tta"]["loss_weights"]

        for l in range(len(q_adapted)):
            loss_nce_q = l_nce(q_adapted[l], q_aug_adapted[l])
            loss_nce_s = l_nce(s_adapted[l], s_aug_adapted[l])
            loss_stat_l = l_stat(q_adapted[l], q_aug_adapted[l]) + l_stat(s_adapted[l], s_aug_adapted[l])
            loss_p_l = l_p(s_adapted[l], s_aug_adapted[l], support_mask, support_mask_aug)

            loss_l = (w_cfg["w_nce_q"] * loss_nce_q +
                      w_cfg["w_nce_s"] * loss_nce_s +
                      w_cfg["w_stat"] * loss_stat_l +
                      w_cfg["w_p"] * loss_p_l)
            
            total_loss += loss_l

        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{cfg['tta']['epochs']:02d}] - TTA Loss: {total_loss.item():.4f}")

    # 6. Suy luận (Inference) và Đánh giá (Evaluation)
    print("\nHoàn tất TTA. Tiến hành Dense Cross-Attention & Fusion...")
    adapters.eval()
    with torch.no_grad():
        q_feats_final = adapters(backbone(query_img))
        s_feats_final = adapters(backbone(support_img))

        # Cross-Attention Multi-Layer Fusion
        pred_prob_map = fusion_module(q_feats_final, s_feats_final, support_mask, target_size=(224, 224))

        # Phân ngưỡng Otsu
        binary_mask_otsu = otsu_thresholding(pred_prob_map)

        # DenseCRF Refinement (Hậu xử lý)
        img_np = (query_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        prob_np = pred_prob_map[0, 0].cpu().numpy()
        crf_mask_np = crf_refiner(img_np, prob_np)
        crf_mask_tensor = torch.from_numpy(crf_mask_np).unsqueeze(0).unsqueeze(0).to(device)

        # Tính toán chỉ số mIoU & FB-IoU
        evaluator.update(crf_mask_tensor, query_gt)
        metrics = evaluator.compute()

        # Lưu ảnh trực quan hóa
        from core.visualizer import save_prediction_visualization
        save_prediction_visualization(query_img, query_gt, pred_prob_map, crf_mask_tensor, save_path="experiments/single_episode_viz.png")

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ THỬ NGHIỆM EPISODE:")
    print(f"  - mIoU:   {metrics['mIoU'] * 100:.2f}%")
    print(f"  - FB-IoU: {metrics['FB-IoU'] * 100:.2f}%")
    print(f"  - IoU Foreground: {metrics['IoU_FG'] * 100:.2f}%")
    print(f"  - IoU Background: {metrics['IoU_BG'] * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_single_episode("config/default_config.yaml")
