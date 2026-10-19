"""
Thực nghiệm Đánh giá So sánh Mô hình Chuyển văn bản thành giọng nói (TTS Benchmark):
1. Piper TTS (vi_VN-vais1000-medium, MIT) - Baseline chính
2. Meta MMS-TTS (facebook/mms-tts-vie, CC-BY-NC) - Baseline đối chứng học thuật
Đo lường:
- Độ trễ tổng hợp trên CPU (Latency s)
- Thời lượng âm thanh (Audio duration s)
- Hệ số thời gian thực (Real-Time Factor - RTF): RTF < 1.0 là nhanh hơn thời gian nói thực
- Tần số lấy mẫu (Sample Rate Hz)
- Khả năng đọc các danh từ riêng Champa (Ganesha, Tara, Apsara, Shiva, Trà Kiệu, Po Rome, Lakshmi)
"""
import sys
import time
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from tts_service import ChampaTTSService

def main():
    print("=== BẮT ĐẦU BENCHMARK THUYẾT MINH GIỌNG NÓI (TTS BENCHMARK) ===")
    tts = ChampaTTSService()
    
    test_sentences = [
        ("Câu 1 (Định danh ngắn)", "Hiện vật được định danh khoa học là Tượng thần Ganesha, phong cách Mỹ Sơn E1."),
        ("Câu 2 (Thế kỷ & Niên đại)", "Hồ sơ ghi nhận hiện vật có niên đại vào khoảng Thế kỷ VII – VIII, chế tác từ chất liệu Sa thạch."),
        ("Câu 3 (Tên riêng & Tôn giáo)", "Tác phẩm chạm khắc nữ thần Lakshmi, chim thần Garuda và tư thế thủ ấn thiền định của thần Shiva."),
        ("Câu 4 (Địa danh khảo cổ)", "Khai quật tại Thánh địa Mỹ Sơn, Phật viện Đồng Dương và kinh đô Simhapura Trà Kiệu."),
        ("Câu 5 (Đoạn thuyết minh mẫu)", "Hiện vật là Bảo vật Quốc gia vô giá, minh chứng cho đỉnh cao kỹ thuật điêu khắc đá sa thạch và giao lưu văn hóa quốc tế trong khu vực Đông Nam Á thời cổ trung đại.")
    ]
    
    results = []
    
    for label, text in test_sentences:
        clean_text = tts.clean_text_for_speech(text)
        print(f"\n--- {label} ---")
        print(f"Văn bản đọc: '{clean_text}'")
        
        # 1. Piper TTS
        res_piper = tts.synthesize(clean_text, engine="piper", use_cache=False)
        print(f"[Piper] Độ trễ: {res_piper['latency_sec']:.3f}s | Âm thanh: {res_piper['duration_sec']:.2f}s | RTF: {res_piper['rtf']:.3f}")
        
        # 2. MMS TTS
        res_mms = tts.synthesize(clean_text, engine="mms", use_cache=False)
        print(f"[MMS-TTS] Độ trễ: {res_mms['latency_sec']:.3f}s | Âm thanh: {res_mms['duration_sec']:.2f}s | RTF: {res_mms['rtf']:.3f}")
        
        results.append({
            "Mẫu thử": label,
            "Số từ": len(clean_text.split()),
            "Piper Độ trễ (s)": round(res_piper['latency_sec'], 3),
            "Piper Thời lượng (s)": round(res_piper['duration_sec'], 2),
            "Piper RTF": round(res_piper['rtf'], 3),
            "MMS Độ trễ (s)": round(res_mms['latency_sec'], 3),
            "MMS Thời lượng (s)": round(res_mms['duration_sec'], 2),
            "MMS RTF": round(res_mms['rtf'], 3)
        })
        
    df = pd.DataFrame(results)
    print("\n=== BẢNG TỔNG HỢP SO SÁNH PIPER VS MMS-TTS TRÊN CPU ===")
    print(df.to_string())
    
    # Tính trung bình RTF và Độ trễ
    avg_piper_rtf = df["Piper RTF"].mean()
    avg_mms_rtf = df["MMS RTF"].mean()
    avg_piper_lat = df["Piper Độ trễ (s)"].mean()
    avg_mms_lat = df["MMS Độ trễ (s)"].mean()
    
    print("\n=== THỐNG KÊ TRUNG BÌNH ===")
    print(f"Piper TTS (CPU): Độ trễ TB = {avg_piper_lat:.3f}s | RTF TB = {avg_piper_rtf:.3f} (Nhanh gấp {1.0/avg_piper_rtf:.1f}x tốc độ nói thực)")
    print(f"MMS-TTS (CPU):   Độ trễ TB = {avg_mms_lat:.3f}s | RTF TB = {avg_mms_rtf:.3f} (Nhanh gấp {1.0/avg_mms_rtf:.1f}x tốc độ nói thực)")
    
    # Lưu kết quả
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "tts_benchmark_comparison.csv", index=False)
    print(f"\nĐã lưu kết quả vào: results/tts_benchmark_comparison.csv")

if __name__ == "__main__":
    main()

