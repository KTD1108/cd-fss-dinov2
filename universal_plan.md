# Kế hoạch Thiết kế Mạng CD-FSS Universal (Dựa trên DINOv2)
**Nhánh (Branch):** `v2-universal-adaptation`

## 1. Phân tích giới hạn hiện tại & Nguyên nhân cốt lõi
Dự án gốc ABCDFSS (dùng ResNet-50) đạt hiệu năng đồng đều trên mọi bộ dữ liệu (từ ảnh y tế ISIC, Lung đến viễn thám DeepGlobe) là nhờ **tháp đặc trưng không gian (Spatial Feature Pyramid)**. ResNet-50 trích xuất đặc trưng ở nhiều mức độ phân giải (stride 4, 8, 16, 32). Với ảnh $400 \times 400$, stride 4 tạo ra feature map $100 \times 100$, giúp bắt các biên rất mảnh trong DeepGlobe.

Ngược lại, DINOv2 (ViT-S/14) chia ảnh thành các patch $14 \times 14$ cố định. Toàn bộ các block từ nông đến sâu đều chỉ có độ phân giải $28 \times 28$ (với ảnh $392 \times 392$). DINOv2 cực mạnh về **ngữ nghĩa (semantics)** nhưng bị thô về **không gian (spatial resolution)**. Đó là lý do nó làm tốt trên ISIC/Lung nhưng tụt hậu trên DeepGlobe (vật thể nhỏ, phức tạp).

## 2. Giải pháp Universal (Hiệu quả trên mọi bộ dữ liệu)
Để mô hình thực sự "Universal", chúng ta cần kết hợp sức mạnh ngữ nghĩa của DINOv2 với khả năng phân giải không gian và khả năng thích nghi miền (Cross-Domain).

### 2.1. Multi-Scale & Multi-Layer Adaptive Fusion (Dung hợp Đa tỷ lệ và Đa tầng Thích ứng)
- **Vấn đề**: Các layer khác nhau của DINOv2 học các thông tin khác nhau. Trọng số cố định `[0.25, 0.25, 0.25, 0.25]` không phù hợp cho mọi miền dữ liệu.
- **Giải pháp**: 
  1. Biến trọng số các layer (Layer Weights) thành **tham số học được (Learnable Parameters)** và tối ưu hóa chúng ngay trong quá trình Test-Time Adaptation (TTA) thông qua Entropy Minimization.
  2. Thực hiện **Multi-Scale Inference**: Trích xuất đặc trưng của Query/Support ở 2 độ phân giải (ví dụ: $392 \times 392$ và $518 \times 518$) sau đó tổng hợp Dense Affinity Map. Điều này giúp DINOv2 nhìn được cả cấu trúc tổng thể (Lung) và chi tiết nhỏ (DeepGlobe).

### 2.2. Nâng cấp Test-Time Adaptation (TTA) 
- **Vấn đề**: ABCDFSS chỉ dùng Contrastive Loss (InfoNCE) và Prototype Loss. Trên DINOv2, các đặc trưng đã rất tốt, việc cố gắng ép các phân bố có thể làm hỏng biểu diễn.
- **Giải pháp**: Bổ sung **Target Entropy Minimization (Giảm thiểu Entropy mục tiêu)** trên ảnh Query. Entropy Minimization ép mô hình đưa ra các dự đoán cực đoan (tự tin cao) cho ảnh Query, ép các đường biên phân đoạn (decision boundary) đi qua vùng có mật độ dữ liệu thấp, cực kỳ hữu ích cho việc tách nền/vật thể trong miền dữ liệu mới.

### 2.3. Tối ưu hóa Bộ Lọc Biên (Adaptive Edge Refiner)
- **Vấn đề**: Guided Filter hiện tại dùng tham số `r=4, eps=1e-2` cố định. Viễn thám cần `eps` nhỏ để bám sát chi tiết, y tế cần `eps` lớn để làm mịn các vùng nhiễu.
- **Giải pháp**: Xây dựng `AdaptiveGuidedFilter` có khả năng tự động nội suy tham số `r` và `eps` dựa trên đặc tính của ảnh (ví dụ: dựa trên phương sai của ảnh RGB hoặc dải entropy của dự đoán). 

### 2.4. Khóa (Lock) Background & Đẩy Foreground trong Attention
- Background thường bao gồm rất nhiều loại vật thể khác nhau (đa dạng nội hàm), trong khi Foreground đồng nhất hơn.
- Cải tiến `DualCrossAttention` bằng cách cho phép Temperature $\tau_{\text{FG}}$ và $\tau_{\text{BG}}$ khác biệt hoặc tự học, giúp mô hình mềm dẻo hơn trong việc nới lỏng hay siết chặt khoảng cách đo lường độ tương đồng.

## 3. Kế hoạch Triển khai Chi tiết

**Giai đoạn 1: Nâng cấp Kiến trúc Cốt lõi (Core Architecture)**
1. Sửa đổi `core/attention.py`: Thêm Multi-Scale Inference và Learnable Weights cho `MultiLayerFusion`.
2. Sửa đổi `core/contrastive_head.py`: Bổ sung loss Entropy Minimization cho quá trình TTA trên Query.
3. Sửa đổi `core/guided_filter.py`: Thêm cơ chế tự động điều chỉnh siêu tham số dựa trên ảnh đầu vào.

**Giai đoạn 2: Tích hợp & Pipeline**
1. Cập nhật `experiments/run_benchmark.py` để hỗ trợ Multi-Scale (cho phép truyền mảng `img_sizes = [392, 518]`).
2. Sửa đổi Pipeline TTA để học cả biến số Fusion Weights của mô hình.

**Giai đoạn 3: Thực nghiệm Toàn diện (Benchmarking)**
1. Chạy đánh giá trên **DeepGlobe** (Kiểm tra xem Multi-scale có giải quyết được điểm yếu độ phân giải không).
2. Chạy đánh giá trên **ISIC** và **Lung** (Kiểm tra xem Semantic của DINOv2 có được phát huy tối đa nhờ Learnable Weights không).
3. Lập bảng so sánh đối chuẩn (Ablation Study).

---
*Kế hoạch này đảm bảo tính học thuật cao (Academic Novelty), không sao chép ABCDFSS, mà xây dựng một framework Cross-Domain FSS riêng biệt khai thác triệt để sức mạnh của Vision Transformer Foundation Models.*
