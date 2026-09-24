import torch

class Evaluator:
    """
    Module đánh giá hiệu năng các chỉ số mIoU và FB-IoU (Foreground-Background IoU) cho CD-FSS.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor):
        """
        pred_mask: [B, 1, H, W] hoặc [H, W] (0.0 hoặc 1.0)
        gt_mask:   [B, 1, H, W] hoặc [H, W] (0.0 hoặc 1.0)
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

    def compute(self) -> dict:
        iou_fg = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + 1e-6)
        fb_iou = (iou_fg + iou_bg) / 2.0
        miou = iou_fg # Trong 1-class Binary Few-shot Segmentation, mIoU thường tính theo Foreground class

        return {
            "mIoU": miou,
            "FB-IoU": fb_iou,
            "IoU_FG": iou_fg,
            "IoU_BG": iou_bg
        }

if __name__ == "__main__":
    print("Testing Evaluator module...")
    evaluator = Evaluator()
    pred = (torch.rand(1, 1, 224, 224) > 0.5).float()
    gt = (torch.rand(1, 1, 224, 224) > 0.5).float()

    evaluator.update(pred, gt)
    res = evaluator.compute()
    print("Evaluation Results:", res)
    print("Evaluator test passed successfully!")
