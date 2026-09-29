import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.dinov2_backbone import DINOv2Backbone
from models.meta_decoder import FSSMetaDecoder
from data.dataset import FSS1000Dataset

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

def train_meta_decoder(config_path: str = "config/default_config.yaml", dataset_root: str = "datasets/FSS-1000", epochs: int = 10, batch_size: int = 8):
    print("=" * 65)
    print("BẮT ĐẦU HUẤN LUYỆN META-DECODER TRÊN FSS-1000")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    img_size = cfg["dataset"].get("img_size", 392)
    
    print("1. Khởi tạo Tập dữ liệu FSS-1000 (Train & Val)...")
    try:
        train_dataset = FSS1000Dataset(root_dir=dataset_root, num_episodes=2000, img_size=img_size, split='train')
        val_dataset = FSS1000Dataset(root_dir=dataset_root, num_episodes=200, img_size=img_size, split='val')
    except Exception as e:
        print(f"Không thể khởi tạo Dataset thực: {e}. Tạo dữ liệu giả lập để kiểm thử thuật toán...")
        train_dataset = [{"query_img": torch.randn(3, img_size, img_size), "query_mask": torch.rand(1, img_size, img_size) > 0.5, "support_img": torch.randn(3, img_size, img_size), "support_mask": torch.rand(1, img_size, img_size) > 0.5} for _ in range(10)]
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("2. Đóng băng Backbone DINOv2 và Khởi tạo Meta-Decoder...")
    backbone = DINOv2Backbone(backbone_name=cfg["model"]["backbone_name"], intermediate_layers=cfg["model"]["intermediate_layers"]).to(device)
    backbone.eval() # Bắt buộc Eval
    
    decoder = FSSMetaDecoder(in_channels=backbone.embed_dim, num_layers=len(cfg["model"]["intermediate_layers"])).to(device)
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    
    print("3. Bắt đầu Vòng lặp Huấn luyện...")
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        decoder.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in pbar:
            if isinstance(batch, dict):
                q_img = batch["query_img"].to(device)
                q_gt = batch["query_mask"].float().to(device)
                s_img = batch["support_img"].to(device)
                s_mask = batch["support_mask"].float().to(device)
            else:
                continue

            with torch.no_grad():
                q_feats = backbone(q_img)
                s_feats = backbone(s_img)
                
            optimizer.zero_grad()
            
            # Giải mã
            pred_mask = decoder(q_feats, s_feats, s_mask, target_size=(img_size, img_size))
            
            # Loss = BCE + Dice
            loss = bce_loss(pred_mask, q_gt) + dice_loss(pred_mask, q_gt)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"➡️ Kết thúc Epoch {epoch} - Trung bình Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        decoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                if isinstance(batch, dict):
                    q_img = batch["query_img"].to(device)
                    q_gt = batch["query_mask"].float().to(device)
                    s_img = batch["support_img"].to(device)
                    s_mask = batch["support_mask"].float().to(device)
                    
                    q_feats = backbone(q_img)
                    s_feats = backbone(s_img)
                    pred_mask = decoder(q_feats, s_feats, s_mask, target_size=(img_size, img_size))
                    
                    loss = bce_loss(pred_mask, q_gt) + dice_loss(pred_mask, q_gt)
                    total_val_loss += loss.item()
                    
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"🎯 Trung bình Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(decoder.state_dict(), "weights/best_meta_decoder.pth")
            print("💾 Đã lưu mô hình tốt nhất!")
            
    print("\n✅ Huấn luyện hoàn tất. Sẵn sàng mang sang DeepGlobe để Test!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Meta-Decoder")
    parser.add_argument("--dataset_root", type=str, default="/dataset/FSS-1000", help="Đường dẫn tới FSS-1000")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch")
    args = parser.parse_args()
    
    train_meta_decoder(dataset_root=args.dataset_root, epochs=args.epochs, batch_size=args.batch_size)
