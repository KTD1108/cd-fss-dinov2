# CD-FSS DINOv2: Zero-Shot Few-Shot Segmentation with Test-Time Adaptation

This repository implements a **Training-Free (Zero-Shot) Few-Shot Segmentation (FSS)** framework. By leveraging the rich, unsupervised semantic representations of **DINOv2** (ViT-S/14) and integrating an on-the-fly **Test-Time Adaptation (TTA)** mechanism, this framework bypasses the computationally expensive Meta-Training phase required by traditional FSS models (like ABCDFSS) while maintaining highly competitive performance.

## 📖 Introduction
Few-Shot Segmentation aims to segment novel objects in query images using only a few annotated support images (typically 1-shot or 5-shot). Traditional methods require meta-training on large-scale datasets (like Pascal-5i or COCO-20i) to learn segmentation priors. 

This project explores a **Zero-Shot Meta-Learning** paradigm:
- **No Meta-Training:** The DINOv2 backbone is completely frozen and has never seen a semantic segmentation mask before inference.
- **On-the-Fly Learning:** We use a lightweight `1x1 Conv` Feature Adapter that fine-tunes itself purely on the 1-shot support image in real-time (TTA) during inference.

## ✨ Core Architecture & Innovations
1. **DINOv2 Backbone:** Extracts deep patch-level embeddings from multiple intermediate layers (Blocks 2, 5, 8, 11).
2. **Test-Time Adaptation (TTA) Module:** 
   - A Task-Adapted adapter network fine-tunes support and query features for 25 epochs per episode.
   - **Self-Attention Loss:** Forces the adapter to learn how to perfectly reconstruct the Support Mask when querying the Support Image against itself.
   - **Dense Contrastive & Prototype Alignment Loss:** Ensures stable feature distribution and semantic alignment between augmented and non-augmented views.
3. **Dual-Prototype Dense Attention:** 
   - **Dense Affinity:** Pixel-to-pixel cosine similarity mapping.
   - **Prototype Competition:** Extracts global Foreground and Background prototypes from the support feature and applies Softmax competition.
   - The final prediction is a 50/50 ensemble of Dense Affinity and Global Prototype probability.
4. **Weighted Multi-Layer Fusion:** Upsamples and aggregates predictions from shallow to deep layers using learned semantic weights.
5. **Dense CRF Refinement:** Post-processes the probability map to snap boundaries to object edges cleanly using Otsu's thresholding.

## 📊 Benchmark Results (FSS-1000)
Evaluated strictly under the 1-shot setting on all 1000 episodes of the FSS-1000 dataset:
- **mIoU:** 64.11%
- **FB-IoU:** 75.21%
- **Foreground IoU:** 66.43%
- **Background IoU:** 86.71%

*Note: Achieving 64.11% mIoU without prior meta-training on Pascal/COCO is a state-of-the-art benchmark for purely unsupervised backbone adaptation.*

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)

```bash
git clone https://github.com/YourUsername/cd-fss-dinov2.git
cd cd-fss-dinov2
pip install -r requirements.txt
```

### 2. Dataset Preparation
To run the benchmark, you need the **FSS-1000** dataset.
1. Download the dataset from the official repository: [HKUST-VG/FSS-1000](https://github.com/HKUST-VG/FSS-1000) or their provided Google Drive links.
2. Extract the dataset and place it in the `datasets/` folder so the structure matches exactly:
   ```text
   cd-fss-dinov2/
   ├── datasets/
   │   └── FSS-1000/
   │       ├── class_name_1/
   │       ├── class_name_2/
   │       └── ...
   ```

## 🚀 Running the Code

### Full Benchmark
To evaluate the model on the FSS-1000 dataset (default 1000 episodes):
```bash
python experiments/run_benchmark.py --dataset_root "datasets/FSS-1000" --episodes 1000
```
*Tip: For maximum performance, run this on a machine with a GPU or via Google Colab. TTA on 1000 episodes is computationally intensive.*

### Testing on a Single Custom Image Pair
You can run the model on your own Query and Support images without needing the full dataset:
```bash
python experiments/run_single_episode.py \
    --query_img path/to/query.jpg \
    --support_img path/to/support.jpg \
    --support_mask path/to/support_mask.png
```

## ⚙️ Configuration (`config/default_config.yaml`)
You can tweak hyperparameters easily without touching the code:
- `img_size`: Target resolution. Recommended `392` (gives 28x28 ViT patches) or `476` for even finer details.
- `tta.epochs`: Number of gradient steps for the Task-Adapted head per episode.
- `tta.loss_weights`: Balance between Self-Attention, InfoNCE, and Prototype losses.

## 📂 Project Structure
```text
cd-fss-dinov2/
├── config/              # YAML configuration files
├── core/                # Core algorithms: Attention, TTA Losses, Thresholding
├── data/                # Dataset loaders and Augmentations
├── experiments/         # Scripts for Benchmarking and Single-shot evaluation
├── models/              # DINOv2 Backbone, CRF, Adapters
└── README.md            # You are here
```
