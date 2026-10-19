import json
import csv
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def run_deduplication():
    logger.info("Bắt đầu kiểm tra và khử trùng lặp...")
    
    images_file = Path("data/processed/images.jsonl")
    if not images_file.exists():
        logger.error("Không tìm thấy images.jsonl")
        return
        
    images = []
    with open(images_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                images.append(json.loads(line))
                
    # Tìm trùng lặp dựa trên SHA256 (bản sao y hệt)
    sha256_groups = defaultdict(list)
    # Tìm trùng lặp dựa trên pHash (bản sao gần giống, crop, resize)
    phash_groups = defaultdict(list)
    
    for img in images:
        if img.get('sha256'):
            sha256_groups[img['sha256']].append(img)
        if img.get('phash'):
            phash_groups[img['phash']].append(img)
            
    duplicates = []
    
    # Gom nhóm các bản sao y hệt
    for sha, group in sha256_groups.items():
        if len(group) > 1:
            for i in range(1, len(group)):
                duplicates.append({
                    'image1_id': group[0]['image_id'],
                    'image2_id': group[i]['image_id'],
                    'type': 'exact_match_sha256',
                    'hash': sha
                })
                
    # Gom nhóm các bản sao gần giống (chưa có trong exact_match)
    # (Trong thực tế pHash có thể so khoảng cách Hamming, nhưng ở mức đơn giản ta so chuỗi giống hệt nhau)
    for phash, group in phash_groups.items():
        if len(group) > 1:
            for i in range(1, len(group)):
                # Tránh lặp lại những cặp đã bị bắt bởi sha256
                exists = any((d['image1_id'] == group[0]['image_id'] and d['image2_id'] == group[i]['image_id']) or 
                             (d['image1_id'] == group[i]['image_id'] and d['image2_id'] == group[0]['image_id']) for d in duplicates)
                if not exists:
                    duplicates.append({
                        'image1_id': group[0]['image_id'],
                        'image2_id': group[i]['image_id'],
                        'type': 'perceptual_match_phash',
                        'hash': phash
                    })
                    
    # Xuất báo cáo
    report_path = Path("data/processed/duplicates_report.csv")
    with open(report_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image1_id', 'image2_id', 'type', 'hash'])
        writer.writeheader()
        writer.writerows(duplicates)
        
    logger.info(f"Đã phát hiện {len(duplicates)} cặp ảnh nghi ngờ trùng lặp. Đã xuất báo cáo ra {report_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_deduplication()

