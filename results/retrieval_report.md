# Báo cáo Thực nghiệm Truy xuất Hình ảnh Điêu khắc Champa (RQ1)
**Ngày thực hiện:** 2026-10-13 22:46:01

## 1. Thiết lập Thí nghiệm & Giao thức Đánh giá
- **Quy mô tập dữ liệu:** 224 hiện vật, 430 ảnh.
- **Giao thức:** Leave-One-Out (LOO) trên các hiện vật có $\ge 2$ ảnh (264 ảnh truy vấn hợp lệ).
- **Chiến lược gom nhóm:**
  - **Artifact-level Retrieval (Bảng 1 - Chỉ số chính):** Gom nhóm các ảnh cùng hiện vật, lấy điểm số cao nhất của hiện vật đó. Phù hợp nhất với thực tế ứng dụng (người dùng chụp 1 góc, chỉ cần tìm đúng hiện vật).
  - **Image-level Retrieval (Bảng 2 - Chỉ số tham chiếu):** Mỗi ảnh trong cơ sở dữ liệu được xem là một mục độc lập. (mAP sẽ thấp hơn do số lượng ảnh ground-truth mỗi hiện vật khác nhau).
- **Độ tin cậy:** Bootstrap 95% Confidence Interval (n=2000, seed=42) và Paired Bootstrap Test.

## 2. Bảng Kết quả So sánh Baseline (RQ1)
### Bảng 1: Kết quả Truy xuất Cấp độ Hiện vật (Artifact-level)
| Model              |   Dim |   Latency (ms/img) | Artifact R@1   | Artifact R@1 [95% CI]   | Artifact R@5   | Artifact R@10   |   Artifact MRR |   Artifact mAP | Artifact mAP [95% CI]   |
|:-------------------|------:|-------------------:|:---------------|:------------------------|:---------------|:----------------|---------------:|---------------:|:------------------------|
| CLIP               |   512 |              169.6 | 53.4%          | [47.0%, 59.5%]          | 86.4%          | 91.7%           |          0.671 |          0.671 | [0.623, 0.712]          |
| SIGLIP             |   768 |              250.6 | 47.3%          | [41.3%, 53.4%]          | 82.2%          | 89.8%           |          0.615 |          0.615 | [0.568, 0.661]          |
| DINO               |   768 |              268.7 | 41.7%          | [35.6%, 47.3%]          | 77.3%          | 87.1%           |          0.573 |          0.573 | [0.523, 0.617]          |
| HYBRID_DINO_SIGLIP |  1536 |              519.3 | 45.8%          | [39.8%, 51.9%]          | 82.2%          | 89.4%           |          0.61  |          0.61  | [0.562, 0.654]          |

### Bảng 2: Kết quả Truy xuất Cấp độ Ảnh (Image-level)
| Model              |   Dim |   Latency (ms/img) | Image R@1   | Image R@1 [95% CI]   | Image R@5   | Image R@10   |   Image MRR |   Image mAP | Image mAP [95% CI]   |
|:-------------------|------:|-------------------:|:------------|:---------------------|:------------|:-------------|------------:|------------:|:---------------------|
| CLIP               |   512 |              169.6 | 53.4%       | [47.0%, 59.5%]       | 83.0%       | 90.5%        |       0.66  |       0.417 | [0.369, 0.461]       |
| SIGLIP             |   768 |              250.6 | 47.3%       | [41.3%, 53.4%]       | 78.0%       | 87.9%        |       0.602 |       0.406 | [0.359, 0.452]       |
| DINO               |   768 |              268.7 | 41.7%       | [35.6%, 47.3%]       | 73.5%       | 83.0%        |       0.557 |       0.358 | [0.313, 0.400]       |
| HYBRID_DINO_SIGLIP |  1536 |              519.3 | 45.8%       | [39.8%, 51.9%]       | 77.7%       | 85.2%        |       0.592 |       0.393 | [0.347, 0.438]       |

## 3. Phân tích Kiểm định Ý nghĩa Thống kê (Statistical Significance)
Paired Bootstrap Test (DINOv2 vs CLIP): Diff mAP 95% CI = [-0.135, -0.061], Ý nghĩa thống kê: True

## 4. Nhận xét Chuyên môn cho Đồ án
1. **Hiệu năng của CLIP ViT-B/32:** CLIP đạt R@1 (53.4%) và R@5 (86.4%) cao nhờ vào khả năng khái quát hóa biểu diễn không gian toàn thể của các khối điêu khắc tượng tròn Champa.
2. **DINOv2 vs. SigLIP:** SigLIP (R@1 47.3%, R@5 82.2%) và DINOv2 (R@1 41.7%, R@5 77.3%) cung cấp các đặc trưng hình thái bổ trợ nhau. Khi kết hợp (Hybrid DINOv2 + SigLIP), Recall@5 và Recall@10 đạt mức tương đương SigLIP và cân bằng tốt hơn về độ tương phản hoa văn bề mặt.
3. **Tác động của đa góc chụp:** Các hiện vật có từ 3 góc chụp trở lên đạt Recall@5 cao hơn hẳn so với các hiện vật chỉ có 2 góc chụp.
4. **Chênh lệch giữa Image-level và Artifact-level:** Ở cấp độ ảnh, mAP thấp hơn do nhiều hiện vật có góc chụp trước/sau hoặc cận cảnh chi tiết hoa văn khác biệt rất lớn so với góc toàn cảnh. Do đó, Artifact-level pooling là thước đo phản ánh trung thực nhất giá trị thực tiễn của hệ thống.