import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def refine_data():
    artifacts_file = Path("data/processed/artifacts.jsonl")
    images_file = Path("data/processed/images.jsonl")
    
    if not artifacts_file.exists() or not images_file.exists():
        return
        
    artifacts = []
    with open(artifacts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                artifacts.append(json.loads(line))
                
    images = []
    with open(images_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                images.append(json.loads(line))

    # 1. Lọc các hiện vật không thuộc Champa Việt Nam (như Chamba Ấn Độ)
    valid_artifacts = []
    invalid_ids = set()
    
    for a in artifacts:
        # Gom text để kiểm tra
        full_text = " ".join([str(v) for k, v in a.items() if v and isinstance(v, str)]).lower()
        
        if "india" in full_text or "punjab" in full_text or "chamba," in full_text:
            invalid_ids.add(a['artifact_id'])
            logger.info(f"Loại bỏ hiện vật Ấn Độ: {a.get('name_en')}")
            continue
            
        valid_artifacts.append(a)
        
    valid_images = [img for img in images if img['artifact_id'] not in invalid_ids]
    
    # Ghi lại
    with open(artifacts_file, "w", encoding="utf-8") as f:
        for a in valid_artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open(images_file, "w", encoding="utf-8") as f:
        for i in valid_images:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

    # 2. Thống kê chi tiết thuyết minh
    desc_lengths = []
    detailed_count = 0
    
    for a in valid_artifacts:
        desc = a.get('description') or ""
        desc_lengths.append(len(desc))
        if len(desc) > 150: # Hơn 150 ký tự được coi là có chút chi tiết
            detailed_count += 1
            
    avg_len = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
    logger.info(f"Tổng số hiện vật hợp lệ: {len(valid_artifacts)}")
    logger.info(f"Độ dài mô tả trung bình: {avg_len:.0f} ký tự")
    logger.info(f"Số hiện vật có mô tả chi tiết (>150 ký tự): {detailed_count}/{len(valid_artifacts)} ({(detailed_count/len(valid_artifacts))*100:.1f}%)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    refine_data()

