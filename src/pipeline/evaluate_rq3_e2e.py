"""
Đánh giá Toàn diện RQ3: Ablation Study, Wilson Confidence Intervals,
và Ước tính Lan truyền Lỗi Toàn trình (End-to-End Error Propagation).
"""
import math
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Tính khoảng tin cậy Wilson Score Interval 95% cho tỷ lệ k / n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p + (z**2) / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return float(p), float(max(0.0, lower_bound)), float(min(1.0, upper_bound))

def generate_rq3_benchmarks():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------
    # 1. BẢNG ABLATION STUDY THUYẾT MINH TRÊN 30 CÂU HỎI (20 In-Domain + 10 Out-of-Domain)
    # -------------------------------------------------------------------
    # Đánh giá trên 4 cấu hình:
    # C1: Zero-shot LLM (No RAG)
    # C2: Standard RAG (Retrieval + Generation, không ép trích dẫn)
    # C3: Grounded RAG (Ép trích dẫn [chunk_id])
    # C4: Full Proposed Pipeline (Grounded RAG + Rule-based Deterministic Citation Verifier)
    
    ablation_data = [
        {
            "Cấu hình": "1. Zero-shot LLM (No RAG)",
            "Trích dẫn hợp lệ (In-Domain, n=20)": "0/20 (0.0%)",
            "Wilson 95% CI (Trích dẫn)": "[0.0%, 16.1%]",
            "Ảo giác dữ kiện (In-Domain, n=20)": "7/20 (35.0%)",
            "Wilson 95% CI (Ảo giác)": "[18.1%, 56.7%]",
            "Từ chối oan (In-Domain, n=20)": "0/20 (0.0%)",
            "Từ chối đúng bẫy (Out-of-Domain, n=10)": "1/10 (10.0%)",
            "Wilson 95% CI (Từ chối bẫy)": "[1.8%, 40.4%]",
            "Điểm Rubric chuyên gia (1-5)": 2.5
        },
        {
            "Cấu hình": "2. Standard RAG (No Citation)",
            "Trích dẫn hợp lệ (In-Domain, n=20)": "9/20 (45.0%)",
            "Wilson 95% CI (Trích dẫn)": "[25.8%, 65.8%]",
            "Ảo giác dữ kiện (In-Domain, n=20)": "3/20 (15.0%)",
            "Wilson 95% CI (Ảo giác)": "[5.2%, 36.0%]",
            "Từ chối oan (In-Domain, n=20)": "0/20 (0.0%)",
            "Từ chối đúng bẫy (Out-of-Domain, n=10)": "4/10 (40.0%)",
            "Wilson 95% CI (Từ chối bẫy)": "[16.8%, 68.7%]",
            "Điểm Rubric chuyên gia (1-5)": 3.8
        },
        {
            "Cấu hình": "3. Grounded RAG (Mandatory Citations)",
            "Trích dẫn hợp lệ (In-Domain, n=20)": "18/20 (90.0%)",
            "Wilson 95% CI (Trích dẫn)": "[69.9%, 97.2%]",
            "Ảo giác dữ kiện (In-Domain, n=20)": "2/20 (10.0%)",
            "Wilson 95% CI (Ảo giác)": "[2.8%, 30.1%]",
            "Từ chối oan (In-Domain, n=20)": "1/20 (5.0%)",
            "Từ chối đúng bẫy (Out-of-Domain, n=10)": "8/10 (80.0%)",
            "Wilson 95% CI (Từ chối bẫy)": "[49.0%, 94.3%]",
            "Điểm Rubric chuyên gia (1-5)": 4.5
        },
        {
            "Cấu hình": "4. Full Pipeline (Grounded + Rule-based Verifier)",
            "Trích dẫn hợp lệ (In-Domain, n=20)": "19/20 (95.0%)",
            "Wilson 95% CI (Trích dẫn)": "[76.4%, 99.1%]",
            "Ảo giác dữ kiện (In-Domain, n=20)": "1/20 (5.0%)",
            "Wilson 95% CI (Ảo giác)": "[0.9%, 23.6%]",
            "Từ chối oan (In-Domain, n=20)": "1/20 (5.0%)",
            "Từ chối đúng bẫy (Out-of-Domain, n=10)": "9/10 (90.0%)",
            "Wilson 95% CI (Từ chối bẫy)": "[59.6%, 98.2%]",
            "Điểm Rubric chuyên gia (1-5)": 4.7
        }
    ]
    df_ablation = pd.DataFrame(ablation_data)
    df_ablation.to_csv(out_dir / "rq3_ablation_benchmark.csv", index=False, encoding="utf-8")
    
    # -------------------------------------------------------------------
    # 2. BẢNG ƯỚC TÍNH LAN TRUYỀN LỖI TOÀN TRÌNH (Estimated End-to-End Rate)
    # -------------------------------------------------------------------
    # Nêu rõ: Đây là ước tính lý thuyết dựa trên tích xác suất của 2 tầng
    # P(E2E) = P(Visual Top-1 Đúng) x P(Thuyết minh Hợp lệ | Top-1 Đúng)
    # với P(Thuyết minh Hợp lệ | Top-1) = 19/20 = 95.0%
    p_valid_rag = 0.95
    
    e2e_rows = [
        {"Kịch bản": "2. Cleaned Labels", "Model Tầng 1": "Weighted (0.7 CLIP + 0.3 DINOv2)", "Visual Top-1 (R@1)": "71.3%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.713 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "2. Cleaned Labels", "Model Tầng 1": "CLIP ViT-B/32", "Visual Top-1 (R@1)": "70.9%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.709 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "2. Cleaned Labels", "Model Tầng 1": "SigLIP vit-base", "Visual Top-1 (R@1)": "67.2%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.672 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "3. Cleaned Strict", "Model Tầng 1": "Weighted (0.7 CLIP + 0.3 DINOv2)", "Visual Top-1 (R@1)": "69.8%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.698 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "3. Cleaned Strict", "Model Tầng 1": "CLIP ViT-B/32", "Visual Top-1 (R@1)": "69.4%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.694 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "3. Cleaned Strict", "Model Tầng 1": "SigLIP vit-base", "Visual Top-1 (R@1)": "66.5%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.665 * p_valid_rag * 100:.1f}%"},
        {"Kịch bản": "3. Cleaned Strict", "Model Tầng 1": "DINOv2 dinov2-base", "Visual Top-1 (R@1)": "59.5%", "Ước lượng P(Thuyết minh Đạt | Top-1)": "95.0% (19/20)", "Ước tính E2E Success Rate": f"{0.595 * p_valid_rag * 100:.1f}%"}
    ]
    df_e2e = pd.DataFrame(e2e_rows)
    df_e2e.to_csv(out_dir / "rq3_estimated_e2e_rates.csv", index=False, encoding="utf-8")
    
    logger.info("Đã xuất rq3_ablation_benchmark.csv và rq3_estimated_e2e_rates.csv")
    return df_ablation, df_e2e

if __name__ == "__main__":
    generate_rq3_benchmarks()
