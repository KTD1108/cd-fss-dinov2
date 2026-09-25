import torch

class Evaluator:
    """
    Module đánh giá hiệu năng các chỉ số mIoU và FB-IoU (Foreground-Background IoU) cho CD-FSS.
    Hỗ trợ cả đo lường cấp Episode và Tích lũy Class-wise theo chuẩn các bài báo CD-FSS (ABCDFSS, PATNet).
    """
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.class_inter = [0.0] * self.num_classes
        self.class_union = [0.0] * self.num_classes
        self.active_classes = set()

    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, class_id: int = None):
        """
        pred_mask: [B, 1, H, W] hoặc [H, W] (0.0 hoặc 1.0)
        gt_mask:   [B, 1, H, W] hoặc [H, W] (0.0 hoặc 1.0)
        class_id:  id lớp ngữ nghĩa (tùy chọn)
        """
        pred_b = (pred_mask > 0.5).bool()
        gt_b = (gt_mask > 0.5).bool()

        tp = (pred_b & gt_b).sum().item()
        fp = (pred_b & ~gt_b).sum().item()
        fn = (~pred_b & gt_b).sum().item()
        tn = (~pred_b & ~gt_b).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        if class_id is not None and 0 <= class_id < self.num_classes:
            self.class_inter[class_id] += tp
            self.class_union[class_id] += (tp + fp + fn)
            self.active_classes.add(class_id)

    def compute(self) -> dict:
        iou_fg = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + 1e-6)
        fb_iou = (iou_fg + iou_bg) / 2.0
        
        # Nếu có thông tin class-wise, tính class mIoU chuẩn theo benchmark CD-FSS
        if len(self.active_classes) > 0:
            class_ious = [
                self.class_inter[c] / (self.class_union[c] + 1e-6)
                for c in self.active_classes
            ]
            class_miou = sum(class_ious) / len(class_ious)
        else:
            class_miou = iou_fg

        return {
            "mIoU": class_miou,
            "FB-IoU": fb_iou,
            "IoU_FG": iou_fg,
            "IoU_BG": iou_bg
        }

if __name__ == "__main__":
    print("Testing Evaluator module...")
    evaluator = Evaluator(num_classes=6)
    pred = (torch.rand(1, 1, 224, 224) > 0.5).float()
    gt = (torch.rand(1, 1, 224, 224) > 0.5).float()

    evaluator.update(pred, gt, class_id=1)
    res = evaluator.compute()
    print("Evaluation Results:", res)
    print("Evaluator test passed successfully!")
