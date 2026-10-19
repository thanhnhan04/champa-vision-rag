import json
from pathlib import Path
import logging

def run_quality_control():
    logger = logging.getLogger(__name__)
    logger.info("Đang chạy kiểm tra chất lượng (QC)...")
    
    artifacts_file = Path("data/processed/artifacts.jsonl")
    images_file = Path("data/processed/images.jsonl")
    
    if not artifacts_file.exists() or not images_file.exists():
        logger.error("Không tìm thấy dữ liệu processed.")
        return
        
    artifacts = [json.loads(line) for line in artifacts_file.read_text('utf-8').strip().split('\n') if line]
    images = [json.loads(line) for line in images_file.read_text('utf-8').strip().split('\n') if line]
    
    # 1. Số lượng cơ bản
    total_artifacts = len(artifacts)
    total_images = len(images)
    
    # Số ảnh trên mỗi hiện vật
    img_per_artifact = {}
    for img in images:
        a_id = img['artifact_id']
        img_per_artifact[a_id] = img_per_artifact.get(a_id, 0) + 1
        
    # 2. Tỷ lệ bản ghi thiếu trường (Missing fields)
    missing_stats = {
        'period': sum(1 for a in artifacts if not a.get('period')),
        'material': sum(1 for a in artifacts if not a.get('material')),
        'origin_site': sum(1 for a in artifacts if not a.get('origin_site')),
        'dimensions': sum(1 for a in artifacts if not a.get('dimensions')),
        'description': sum(1 for a in artifacts if not a.get('description')),
    }
    
    # Check thin metadata (description < 30 chars)
    thin_description = [a for a in artifacts if not a.get('description') or len(a.get('description', '')) < 30]
    # 3. Ảnh lỗi hoặc quá nhỏ (giả sử đã bị downloader xóa file_path)
    invalid_images = [img for img in images if not img.get('file_path')]
    
    # Viết báo cáo
    report_content = [
        "# Báo cáo Kiểm tra Chất lượng Dữ liệu (QC Report)",
        f"**Thời gian:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Thống kê chung",
        f"- Tổng số hiện vật (Artifacts): {total_artifacts}",
        f"- Tổng số hình ảnh (Images): {total_images}",
        f"- Trung bình ảnh/hiện vật: {total_images/total_artifacts:.1f}",
        "",
        "## 2. Phân tích độ khuyết thiếu (Missing Data)",
        "| Trường | Số bản ghi thiếu | Tỷ lệ (%) |",
        "|---|---|---|",
    ]
    
    for field, count in missing_stats.items():
        pct = (count / total_artifacts) * 100
        report_content.append(f"| {field} | {count} | {pct:.1f}% |")
        
    report_content.extend([
        "",
        "## 2.1. Metadata mỏng (Thin Metadata)",
        f"Có {len(thin_description)} hiện vật có mô tả quá ngắn (< 30 ký tự) hoặc bị trống. Tỷ lệ: {(len(thin_description)/total_artifacts)*100:.1f}%",
        "",
        "## 3. Hình ảnh không đạt chuẩn (lỗi, hỏng, hoặc bị chặn)",
        f"Có {len(invalid_images)} ảnh không đạt chuẩn và đã bị loại bỏ file vật lý:"
    ])
    
    for img in invalid_images[:10]:
        report_content.append(f"- `[{img['image_id']}]` từ URL: {img['image_url']}")
    if len(invalid_images) > 10:
        report_content.append("... (xem chi tiết trong log tải ảnh)")
        
    Path("data/processed/qc_report.md").write_text('\n'.join(report_content), 'utf-8')
    logger.info("Đã xuất báo cáo QC ra data/processed/qc_report.md")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_quality_control()

