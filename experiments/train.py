import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.dinov2_backbone import DINOv2Backbone
from core.attention import MetaDecoder
from core.evaluator import Evaluator
from data.dataset import FSS1000Dataset

def dice_loss(pred, target, smooth=1e-5):
    """
    Hàm mất mát Dice Loss cực kỳ hiệu quả cho phân đoạn ảnh
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def train(config_path: str = "config/default_config.yaml", dataset_root: str = "datasets/FSS-1000"):
    print("=" * 65)
    print("BẮT ĐẦU META-TRAINING DINOv2 DECODER TRÊN FSS-1000 (TẬP TRAIN)")
    print("=" * 65)
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device}")

    # 1. Chuẩn bị Dataset (Train & Val)
    img_size = cfg["dataset"]["img_size"]
    # Để train nhanh hơn, ta lấy ví dụ 5000 episodes cho train, 500 cho val
    train_dataset = FSS1000Dataset(root_dir=dataset_root, num_episodes=5000, img_size=img_size, split='train', seed=42)
    val_dataset = FSS1000Dataset(root_dir=dataset_root, num_episodes=500, img_size=img_size, split='val', seed=99)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Số lượng tập Train: {len(train_dataset)} | Tập Val: {len(val_dataset)}")

    # 2. Khởi tạo Mô hình
    backbone = DINOv2Backbone(
        backbone_name=cfg["model"]["backbone_name"],
        intermediate_layers=cfg["model"]["intermediate_layers"]
    ).to(device)
    backbone.eval() # Backbone LUÔN ĐÓNG BĂNG

    decoder = MetaDecoder(in_channels=384, num_layers=4).to(device)
    
    # Optimizer (AdamW)
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) # 10 epochs
    
    num_epochs = 10
    best_val_miou = 0.0

    # 3. Vòng lặp Huấn luyện (Epochs)
    for epoch in range(1, num_epochs + 1):
        decoder.train()
        train_loss = 0.0
        
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            q_img = batch["query_img"].to(device)
            q_mask = batch["query_mask"].to(device)
            s_img = batch["support_img"].to(device)
            s_mask = batch["support_mask"].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                q_feats = backbone(q_img)
                s_feats = backbone(s_img)
                
            # Đưa qua MetaDecoder
            pred_mask = decoder(q_feats, s_feats, s_mask, target_size=(img_size, img_size))
            
            # Tính Loss kết hợp BCE và Dice Loss
            bce_loss = F.binary_cross_entropy(pred_mask.clamp(1e-6, 1-1e-6), q_mask)
            d_loss = dice_loss(pred_mask, q_mask)
            loss = bce_loss + d_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        
        # 4. Validation
        decoder.eval()
        val_evaluator = Evaluator()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                q_img = batch["query_img"].to(device)
                q_mask = batch["query_mask"].to(device)
                s_img = batch["support_img"].to(device)
                s_mask = batch["support_mask"].to(device)
                
                q_feats = backbone(q_img)
                s_feats = backbone(s_img)
                
                pred_mask = decoder(q_feats, s_feats, s_mask, target_size=(img_size, img_size))
                
                # Binarize threshold = 0.5
                pred_binary = (pred_mask > 0.5).float()
                val_evaluator.update(pred_binary, q_mask)
                
        val_res = val_evaluator.compute()
        val_miou = val_res['mIoU'] * 100
        print(f"Val mIoU: {val_miou:.2f}% | Val FB-IoU: {val_res['FB-IoU'] * 100:.2f}%")
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            os.makedirs("models/weights", exist_ok=True)
            torch.save(decoder.state_dict(), "models/weights/meta_decoder_best.pth")
            print(f"🌟 Đã lưu model tốt nhất tại Epoch {epoch} với mIoU: {best_val_miou:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="datasets/FSS-1000")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    args = parser.parse_args()
    
    train(config_path=args.config, dataset_root=args.dataset_root)
