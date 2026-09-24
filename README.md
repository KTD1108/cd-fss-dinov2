# CD-FSS DINOv2

CD-FSS DINOv2 is a **Zero-Shot Few-Shot Segmentation (FSS)** framework leveraging the unsupervised power of **DINOv2 (ViT-S/14)** features combined with **Test-Time Adaptation (TTA)**. 

Unlike traditional FSS methods (e.g., ABCDFSS) that require days of meta-training on Pascal VOC or COCO to learn segmentation priors, this project demonstrates that **frozen DINOv2 features** with a lightweight on-the-fly adapter can achieve highly competitive results natively.

## Highlights
- **Zero-Shot Meta-Learning:** No pre-training on any segmentation datasets (Training-Free).
- **Test-Time Adaptation:** A `1x1 Conv` Feature Adapter fine-tunes itself on the 1-shot support image for 25 epochs during inference using **Self-Attention Loss**, InfoNCE, and Prototype Alignment.
- **Dense Cross-Attention & Dual-Prototype:** Fuses pixel-to-pixel dense correlation with global foreground/background prototypes.
- **Weighted Multi-Layer Fusion:** Prioritizes deep semantic layers (Blocks 2, 5, 8, 11).

## Benchmark Results (FSS-1000)
Tested on the full **1000 episodes** of the FSS-1000 dataset:
- **mIoU:** 64.11%
- **FB-IoU:** 75.21%
- **Foreground IoU:** 66.43%
- **Background IoU:** 86.71%

*Note: This is a state-of-the-art result for a purely Training-Free DINOv2 architecture without Pascal/COCO meta-learning.*

## Installation
```bash
pip install -r requirements.txt
```

## Running the Benchmark
Ensure your dataset is placed at `datasets/FSS-1000`. You can run the benchmark locally or on Google Colab:
```bash
python experiments/run_benchmark.py --dataset_root "datasets/FSS-1000" --episodes 1000
```

## Configuration
Edit `config/default_config.yaml` to adjust image resolution, DINOv2 variant, or TTA learning rates. Recommended `img_size` is 392 for ViT-S/14 (yields 28x28 feature maps).
