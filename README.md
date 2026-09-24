<div align="center">
  
# 🎯 CD-FSS DINOv2
**Zero-Shot Few-Shot Segmentation thông qua Test-Time Adaptation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

Dự án này triển khai một nền tảng **Phân đoạn Ảnh Few-Shot hoàn toàn không cần huấn luyện trước (Zero-Shot)**. Bằng cách tận dụng khả năng biểu diễn ngữ nghĩa vượt trội, không giám sát của **DINOv2 (ViT-S/14)** kết hợp với cơ chế **Test-Time Adaptation (TTA)**, mô hình có thể bỏ qua hoàn toàn giai đoạn Meta-Training đắt đỏ (như trong các bài báo truyền thống) mà vẫn giữ được độ chính xác rất cao.

---

## 📖 Giới Thiệu
Few-Shot Segmentation (FSS) là bài toán phân đoạn các vật thể mới trong ảnh truy vấn (Query Image) chỉ dựa vào một vài bức ảnh mẫu (Support Image). Các phương pháp FSS truyền thống đòi hỏi việc Meta-Training trên các bộ dữ liệu lớn (như Pascal-5i hoặc COCO-20i) trong nhiều ngày. 

Dự án này khám phá một hướng đi hoàn toàn mới - **Zero-Shot Meta-Learning**:
- **Không Meta-Training:** Xương sống (Backbone) DINOv2 bị đóng băng hoàn toàn và chưa từng được "dạy" cách phân đoạn ảnh trước khi suy luận.
- **Học Tức Thì (On-the-Fly):** Chúng tôi sử dụng một mạng Adapter `1x1 Conv` siêu nhẹ để tự động tinh chỉnh (fine-tune) theo từng ảnh Support ngay trong thời gian thực (TTA).

## ✨ Kiến Trúc Đột Phá
1. **Backbone DINOv2:** Trích xuất các embedding ở mức độ bản vá (patch-level) từ các tầng sâu (Blocks 2, 5, 8, 11).
2. **Module Test-Time Adaptation (TTA):** 
   - Tinh chỉnh đặc trưng của Query và Support trong 25 epochs nhỏ ở mỗi tập nghiệm.
   - **Self-Attention Loss:** Ép mạng Adapter học cách tái tạo lại hoàn hảo Support Mask khi tự đối chiếu ảnh Support với chính nó.
   - **Dense Contrastive & Prototype Alignment Loss:** Đảm bảo sự ổn định và căn chỉnh phân phối giữa các phiên bản ảnh gốc và ảnh đã qua Data Augmentation.
3. **Dual-Prototype Dense Attention:** 
   - **Dense Affinity:** Khớp độ tương đồng Cosine giữa từng điểm ảnh.
   - **Prototype Competition:** Trích xuất các mẫu đại diện (Prototype) cho Tiền cảnh (Foreground) và Hậu cảnh (Background), áp dụng hàm Softmax để tính xác suất cạnh tranh.
   - Kết quả phân đoạn cuối cùng là sự pha trộn 50/50 giữa Dense Affinity và Global Prototype.
4. **Weighted Multi-Layer Fusion:** Tổng hợp kết quả từ nhiều tầng DINOv2 bằng các trọng số tối ưu hóa ngữ nghĩa sâu.
5. **CRF Refinement:** Khử nhiễu và làm sắc nét viền vật thể bằng Dense CRF kết hợp ngưỡng Otsu.

## 📊 Kết Quả Đánh Giá (FSS-1000)
Đánh giá trên toàn bộ **1000 episodes** của bộ dữ liệu FSS-1000 (Chuẩn 1-Shot):
- **Trung bình mIoU:** `64.11%`
- **Trung bình FB-IoU:** `75.21%`
- **Foreground IoU:** `66.43%`
- **Background IoU:** `86.71%`

> 💡 *Lưu ý: Đạt được 64.11% mIoU mà hoàn toàn không cần Meta-Training trên Pascal/COCO là một kết quả State-of-the-Art (SOTA) cho các phương pháp Training-Free dựa trên DINOv2.*

---

## 🛠️ Cài Đặt & Khởi Chạy

### 1. Yêu cầu hệ thống
- Python 3.8+
- Khuyến nghị sử dụng PyTorch có hỗ trợ CUDA (NVIDIA GPU).

```bash
git clone https://github.com/KTD1108/cd-fss-dinov2.git
cd cd-fss-dinov2
pip install -r requirements.txt
```

### 2. Chuẩn Bị Dữ Liệu (FSS-1000)
Để chạy Benchmark, bạn cần chuẩn bị bộ dữ liệu FSS-1000:
1. Truy cập kho lưu trữ chính thức: [HKUST-VG/FSS-1000](https://github.com/HKUST-VG/FSS-1000).
2. Tìm link tải Google Drive hoặc Baidu Yun được cung cấp trong file README của họ.
3. Giải nén và đặt dữ liệu vào thư mục `datasets/` sao cho cấu trúc như sau:
   ```text
   cd-fss-dinov2/
   ├── datasets/
   │   └── FSS-1000/
   │       ├── class_name_1/
   │       ├── class_name_2/
   │       └── ...
   ```

### 3. Chạy Benchmark
Để chạy đánh giá toàn bộ trên tập FSS-1000 (1000 episodes):
```bash
python experiments/run_benchmark.py --dataset_root "datasets/FSS-1000" --episodes 1000
```
*(Mẹo: Hãy chạy trên máy có GPU hoặc Google Colab, vì TTA trên 1000 tập tốn khá nhiều chi phí tính toán).*

### 4. Chạy Thử trên Ảnh Tự Chọn (Single Episode)
Bạn có thể test khả năng phân đoạn của mô hình trên cặp ảnh Support và Query của riêng bạn:
```bash
python experiments/run_single_episode.py \
    --query_img path/to/query.jpg \
    --support_img path/to/support.jpg \
    --support_mask path/to/support_mask.png
```

---

## ⚙️ Cấu Hình (`config/default_config.yaml`)
Bạn có thể tinh chỉnh các siêu tham số (hyperparameters) cực kỳ dễ dàng:
- `img_size`: Độ phân giải ảnh. Khuyến nghị `392` (tạo ra ma trận đặc trưng `28x28`) hoặc `476` để sắc nét hơn.
- `tta.epochs`: Số lượng bước huấn luyện TTA cho mỗi episode.
- `tta.loss_weights`: Trọng số cân bằng giữa Self-Attention, InfoNCE, và Prototype.

## 📂 Cấu Trúc Thư Mục
```text
cd-fss-dinov2/
├── config/              # Chứa file cấu hình YAML
├── core/                # Core thuật toán (Attention, TTA Losses, Thresholding)
├── data/                # Dataset loaders và Augmentations
├── experiments/         # Scripts chạy Benchmark và Single-shot
├── models/              # Xương sống DINOv2, CRF, Adapters
└── README.md            # Tài liệu dự án
```
