"""
Chuyển đổi dữ liệu từ artifacts.jsonl và images.jsonl thành data/metadata.csv
chuẩn hóa theo yêu cầu của đồ án nghiên cứu.
"""
import json
import csv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_metadata_csv():
    logger.info("Bắt đầu tạo data/metadata.csv...")
    art_file = Path("data/processed/artifacts.jsonl")
    img_file = Path("data/processed/images.jsonl")
    out_csv = Path("data/metadata.csv")
    
    if not art_file.exists() or not img_file.exists():
        logger.error("Không tìm thấy dữ liệu processed.")
        return
        
    artifacts = {a['artifact_id']: a for a in [json.loads(line) for line in art_file.read_text('utf-8').strip().split('\n') if line]}
    images = [json.loads(line) for line in img_file.read_text('utf-8').strip().split('\n') if line]
    
    rows = []
    for img in images:
        aid = img['artifact_id']
        art = artifacts.get(aid, {})
        
        # Lấy thông tin era (ưu tiên ground truth, fallback sang inferred)
        inferred = art.get('inferred_metadata') or {}
        era = art.get('period') or inferred.get('extracted_period') or ''
        mat = art.get('material') or inferred.get('extracted_material') or ''
        
        # Đường dẫn ảnh tương đối chuẩn
        img_path = img.get('file_path') or f"data/images/{img['image_id']}.jpg"
        img_p = Path(img_path)
        if not img_p.exists():
            # Thử tìm file tương ứng trong data/images
            found = list(Path("data/images").glob(f"{img['image_id']}.*"))
            if found:
                img_path = str(found[0].as_posix())
                
        row = {
            'artifact_id': aid,
            'image_id': img['image_id'],
            'image_path': img_path,
            'title': art.get('name_en') or art.get('name_vi') or aid,
            'name_vi_suggested': inferred.get('suggested_name_vi') or '',
            'era': era,
            'material': mat,
            'origin_site': art.get('origin_site') or inferred.get('extracted_origin_site') or '',
            'current_location': art.get('current_location') or '',
            'description': art.get('description') or '',
            'source': art.get('source_name') or '',
            'license': img.get('license') or art.get('license') or '',
            'width': img.get('width') or 0,
            'height': img.get('height') or 0,
            'sha256': img.get('sha256') or '',
            'phash': img.get('phash') or ''
        }
        rows.append(row)
        
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"Đã tạo thành công data/metadata.csv với {len(rows)} dòng ảnh ({len(artifacts)} hiện vật).")

if __name__ == "__main__":
    build_metadata_csv()

