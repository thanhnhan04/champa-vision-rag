import json
import csv
from pathlib import Path
import logging

def remove_duplicates():
    logger = logging.getLogger(__name__)
    logger.info("Bắt đầu loại bỏ ảnh trùng lặp...")
    
    images_file = Path("data/processed/images.jsonl")
    report_file = Path("data/processed/duplicates_report.csv")
    cleaned_images_file = Path("data/processed/images_cleaned.jsonl")
    
    if not images_file.exists() or not report_file.exists():
        logger.error("Không tìm thấy images.jsonl hoặc duplicates_report.csv")
        return
        
    duplicates_to_remove = set()
    with open(report_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Luôn giữ lại image1_id, loại bỏ image2_id
            duplicates_to_remove.add(row['image2_id'])
            
    logger.info(f"Tổng số ảnh cần loại bỏ do trùng lặp: {len(duplicates_to_remove)}")
    
    kept_images = []
    removed_count = 0
    with open(images_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            img = json.loads(line)
            if img['image_id'] in duplicates_to_remove:
                removed_count += 1
            else:
                kept_images.append(img)
                
    # Ghi đè file ảnh (hoặc tạo file cleaned để an toàn)
    with open(images_file, "w", encoding="utf-8") as f:
        for img in kept_images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")
            
    logger.info(f"Đã loại bỏ {removed_count} ảnh. Số ảnh còn lại: {len(kept_images)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    remove_duplicates()

