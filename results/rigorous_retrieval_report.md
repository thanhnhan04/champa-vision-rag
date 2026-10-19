# Báo Cáo Thực Nghiệm Truy Xuất Đa Góc Chụp (RQ1 & RQ2)

**Ngày thực hiện:** 2026-10-13 23:03:26  
**Thiết bị tính toán:** CPU (Intel/AMD x86_64, Single-node Local)  
**Phương pháp thống kê:** Cluster Bootstrap 95% Confidence Interval (Lấy mẫu có hoàn lại theo Hiện vật, n=2000 lần, seed=42).

---

## 1. So Sánh 3 Kịch Bản Dữ Liệu
1. **Kịch bản 1 (Raw Data):** Dữ liệu thô ban đầu (430 ảnh, 224 hiện vật). Chứa nhiễu phân mảnh định danh (The Met vs Commons) và album tổng hợp (33 ảnh Flickr Champa Stone bị gộp chung).
2. **Kịch bản 2 (Cleaned Labels - Khuyến nghị chính):** Đã làm sạch nhãn, tách album tổng hợp thành từng hiện vật độc lập và gộp các mã thực thể trùng chéo (428 ảnh, 247 hiện vật).
3. **Kịch bản 3 (Cleaned Strict Cross-View):** Trên tập Cleaned, loại bỏ bất kỳ ảnh nào trong gallery có Cosine Sim > 0.95 với query (loại trừ burst shots / duplicate scans) nhằm kiểm tra năng lực nhận diện góc chụp khác thực sự.

### Bảng Kết Quả Định Lượng (RQ1 & RQ2)
| Kịch bản          | Model                            |   Queries | Artifact R@1   | Artifact R@1 [95% CI]   | Artifact R@5   |   Artifact mAP | Artifact mAP [95% CI]   |
|:------------------|:---------------------------------|----------:|:---------------|:------------------------|:---------------|---------------:|:------------------------|
| 1. Raw            | CLIP                             |       264 | 53.4%          | [43.4%, 64.4%]          | 86.4%          |          0.671 | [0.593, 0.753]          |
| 1. Raw            | SigLIP                           |       264 | 47.3%          | [35.7%, 60.9%]          | 82.2%          |          0.616 | [0.525, 0.719]          |
| 1. Raw            | DINOv2                           |       264 | 41.3%          | [31.2%, 54.3%]          | 77.3%          |          0.572 | [0.485, 0.674]          |
| 1. Raw            | Hybrid (DINOv2 + SigLIP)         |       264 | 45.1%          | [34.2%, 57.7%]          | 82.2%          |          0.607 | [0.522, 0.704]          |
| 1. Raw            | Ensemble (CLIP + DINOv2)         |       264 | 47.7%          | [36.1%, 60.9%]          | 81.4%          |          0.622 | [0.531, 0.722]          |
| 1. Raw            | Ensemble (CLIP + SigLIP)         |       264 | 49.6%          | [37.7%, 63.2%]          | 83.7%          |          0.647 | [0.556, 0.745]          |
| 1. Raw            | Weighted (0.7 CLIP + 0.3 DINOv2) |       264 | 50.0%          | [37.9%, 63.6%]          | 84.1%          |          0.644 | [0.548, 0.744]          |
| 2. Cleaned        | CLIP                             |       244 | 70.9%          | [63.0%, 80.4%]          | 89.3%          |          0.796 | [0.734, 0.862]          |
| 2. Cleaned        | SigLIP                           |       244 | 67.2%          | [57.9%, 78.6%]          | 88.5%          |          0.763 | [0.688, 0.848]          |
| 2. Cleaned        | DINOv2                           |       244 | 61.5%          | [51.8%, 72.5%]          | 86.1%          |          0.729 | [0.659, 0.808]          |
| 2. Cleaned        | Hybrid (DINOv2 + SigLIP)         |       244 | 66.4%          | [57.2%, 77.2%]          | 88.9%          |          0.764 | [0.698, 0.840]          |
| 2. Cleaned        | Ensemble (CLIP + DINOv2)         |       244 | 68.9%          | [59.8%, 79.5%]          | 88.5%          |          0.778 | [0.711, 0.857]          |
| 2. Cleaned        | Ensemble (CLIP + SigLIP)         |       244 | 69.7%          | [60.5%, 80.6%]          | 88.9%          |          0.786 | [0.713, 0.864]          |
| 2. Cleaned        | Weighted (0.7 CLIP + 0.3 DINOv2) |       244 | 71.3%          | [62.0%, 82.3%]          | 89.3%          |          0.795 | [0.726, 0.871]          |
| 3. Cleaned Strict | CLIP                             |       244 | 42.2%          | [32.3%, 51.1%]          | 67.6%          |          0.541 | [0.434, 0.630]          |
| 3. Cleaned Strict | SigLIP                           |       244 | 53.3%          | [43.8%, 62.9%]          | 74.6%          |          0.625 | [0.536, 0.708]          |
| 3. Cleaned Strict | DINOv2                           |       244 | 38.9%          | [30.0%, 48.5%]          | 70.9%          |          0.533 | [0.455, 0.614]          |
| 3. Cleaned Strict | Hybrid (DINOv2 + SigLIP)         |       244 | 47.1%          | [38.7%, 56.4%]          | 74.6%          |          0.592 | [0.513, 0.668]          |
| 3. Cleaned Strict | Ensemble (CLIP + DINOv2)         |       244 | 50.8%          | [42.5%, 60.1%]          | 75.4%          |          0.618 | [0.539, 0.693]          |
| 3. Cleaned Strict | Ensemble (CLIP + SigLIP)         |       244 | 48.8%          | [38.6%, 58.9%]          | 72.1%          |          0.598 | [0.502, 0.685]          |
| 3. Cleaned Strict | Weighted (0.7 CLIP + 0.3 DINOv2) |       244 | 52.9%          | [44.0%, 62.2%]          | 75.8%          |          0.63  | [0.546, 0.708]          |

---

## 2. Ma Trận Kiểm Định Ghép Cặp (Paired Cluster Bootstrap Test)
*Kiểm định độ chênh lệch mAP cấp độ Hiện vật trên Kịch bản 2 (Cleaned Labels). p-value được tính trực tiếp từ phân phối hiệu số thực nghiệm:*

| Model A                  | Model B                          |   Diff mAP (A - B) | 95% CI of Diff   |   Empirical p-value | Significant (p < 0.05)   |
|:-------------------------|:---------------------------------|-------------------:|:-----------------|--------------------:|:-------------------------|
| CLIP                     | SigLIP                           |              0.032 | [-0.010, +0.072] |               0.145 | Không ý nghĩa (ns)       |
| CLIP                     | DINOv2                           |              0.067 | [+0.025, +0.109] |               0.001 | Có ý nghĩa (*)           |
| CLIP                     | Hybrid (DINOv2 + SigLIP)         |              0.031 | [-0.006, +0.066] |               0.102 | Không ý nghĩa (ns)       |
| CLIP                     | Ensemble (CLIP + DINOv2)         |              0.017 | [-0.017, +0.050] |               0.336 | Không ý nghĩa (ns)       |
| CLIP                     | Ensemble (CLIP + SigLIP)         |              0.009 | [-0.020, +0.039] |               0.555 | Không ý nghĩa (ns)       |
| CLIP                     | Weighted (0.7 CLIP + 0.3 DINOv2) |              0     | [-0.029, +0.027] |               0.962 | Không ý nghĩa (ns)       |
| SigLIP                   | DINOv2                           |              0.035 | [-0.012, +0.082] |               0.149 | Không ý nghĩa (ns)       |
| SigLIP                   | Hybrid (DINOv2 + SigLIP)         |             -0.001 | [-0.034, +0.028] |               0.999 | Không ý nghĩa (ns)       |
| SigLIP                   | Ensemble (CLIP + DINOv2)         |             -0.015 | [-0.051, +0.020] |               0.414 | Không ý nghĩa (ns)       |
| SigLIP                   | Ensemble (CLIP + SigLIP)         |             -0.023 | [-0.040, -0.003] |               0.025 | Có ý nghĩa (*)           |
| SigLIP                   | Weighted (0.7 CLIP + 0.3 DINOv2) |             -0.032 | [-0.062, +0.002] |               0.063 | Không ý nghĩa (ns)       |
| DINOv2                   | Hybrid (DINOv2 + SigLIP)         |             -0.035 | [-0.060, -0.012] |               0.002 | Có ý nghĩa (*)           |
| DINOv2                   | Ensemble (CLIP + DINOv2)         |             -0.05  | [-0.080, -0.026] |               0     | Có ý nghĩa (*)           |
| DINOv2                   | Ensemble (CLIP + SigLIP)         |             -0.057 | [-0.104, -0.009] |               0.015 | Có ý nghĩa (*)           |
| DINOv2                   | Weighted (0.7 CLIP + 0.3 DINOv2) |             -0.066 | [-0.102, -0.035] |               0     | Có ý nghĩa (*)           |
| Hybrid (DINOv2 + SigLIP) | Ensemble (CLIP + DINOv2)         |             -0.014 | [-0.032, +0.001] |               0.069 | Không ý nghĩa (ns)       |
| Hybrid (DINOv2 + SigLIP) | Ensemble (CLIP + SigLIP)         |             -0.022 | [-0.053, +0.012] |               0.195 | Không ý nghĩa (ns)       |
| Hybrid (DINOv2 + SigLIP) | Weighted (0.7 CLIP + 0.3 DINOv2) |             -0.031 | [-0.053, -0.008] |               0.009 | Có ý nghĩa (*)           |
| Ensemble (CLIP + DINOv2) | Ensemble (CLIP + SigLIP)         |             -0.008 | [-0.041, +0.028] |               0.67  | Không ý nghĩa (ns)       |
| Ensemble (CLIP + DINOv2) | Weighted (0.7 CLIP + 0.3 DINOv2) |             -0.017 | [-0.032, -0.001] |               0.038 | Có ý nghĩa (*)           |
| Ensemble (CLIP + SigLIP) | Weighted (0.7 CLIP + 0.3 DINOv2) |             -0.009 | [-0.036, +0.017] |               0.506 | Không ý nghĩa (ns)       |

---

## 3. Các Phát Hiện Khoa Học Cốt Lõi

1. **Hiệu năng sau khi làm sạch nhãn (Raw vs. Cleaned):**
   - Khi tách album tổng hợp Flickr (33 ảnh không còn coi nhau là ground-truth sai) và gộp The Met/Commons đúng thực thể, khoảng tin cậy Cluster Bootstrap phản ánh đúng mức độ dao động theo hiện vật.
2. **Kiểm định thống kê giữa các cặp mô hình:**
   - CLIP vượt trội hơn DINOv2 đơn lẻ một cách có ý nghĩa thống kê (p < 0.05).
   - Tuy nhiên, so sánh giữa CLIP và SigLIP cho thấy khoảng tin cậy hiệu số chứa giá trị 0 và p > 0.05, nghĩa là chưa đủ bằng chứng thống kê để khẳng định CLIP vượt trội hơn SigLIP.
3. **Tổ hợp Ensemble (CLIP + DINOv2 / Weighted):**
   - Mô hình Weighted Combination (0.7 CLIP + 0.3 DINOv2) và Ensemble (1280 chiều) duy trì Recall@5 ở mức rất cao và bổ sung thông tin chi tiết bề mặt của DINOv2.
4. **Kịch bản Strict Cross-View (Loại ảnh Sim > 0.95):**
   - Khi loại bỏ triệt để các ảnh chụp liên tiếp / góc chụp gần như trùng lặp trong gallery, Recall@1 phản ánh đúng độ khó thực tế của bài toán nhận dạng cổ vật qua các góc nhìn khác biệt hoàn toàn (trước, sau, nghiêng).
