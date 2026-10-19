import json
import re
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def clean_name(name: str) -> str:
    """Loại bỏ các hậu tố hình ảnh (DSC, view 1, 01, đuôi mở rộng) để tìm tên gốc của hiện vật."""
    if not name:
        return ""
    
    # Xóa đuôi file nếu có
    name = re.sub(r'\.(jpg|png|jpeg|gif)$', '', name, flags=re.IGNORECASE)
    
    # Xóa các mẫu đánh số cuối chuỗi như " - DSC01883", " 01", "(1)", "view 1"
    name = re.sub(r'(\s*-\s*DSC\d+|\s+view\s+\d+|\s+\d{1,2}$|\s*\(\d+\)$)', '', name, flags=re.IGNORECASE)
    
    # Xóa khoảng trắng thừa và lowercase để so sánh
    return name.strip().lower()

def run_merger():
    logger.info("Bắt đầu gộp ảnh theo hiện vật (Merge Artifacts)...")
    
    artifacts_file = Path("data/processed/artifacts.jsonl")
    images_file = Path("data/processed/images.jsonl")
    
    if not artifacts_file.exists() or not images_file.exists():
        logger.error("Không tìm thấy dữ liệu processed.")
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
                
    # Nhóm các artifact_id có cùng tên gốc
    base_name_to_ids = defaultdict(list)
    id_to_artifact = {}
    
    for a in artifacts:
        a_id = a['artifact_id']
        id_to_artifact[a_id] = a
        
        # Nếu là Wikidata (thường chuẩn) thì gom theo Q-id
        if a_id.startswith("Q"):
            base_name_to_ids[a_id].append(a_id)
        else:
            base_name = clean_name(a['name_en'] or a['name_vi'] or a_id)
            base_name_to_ids[base_name].append(a_id)
            
    # Tạo bản đồ ánh xạ ID bị gộp -> ID chính
    id_mapping = {}
    merged_artifacts = []
    merged_count = 0
    
    for base_name, group in base_name_to_ids.items():
        primary_id = group[0]
        primary_artifact = id_to_artifact[primary_id]
        
        # Có thể cải thiện metadata bằng cách ghép description nếu chúng khác nhau
        
        merged_artifacts.append(primary_artifact)
        
        for duplicate_id in group[1:]:
            id_mapping[duplicate_id] = primary_id
            merged_count += 1
            
    logger.info(f"Đã gộp {merged_count} hiện vật trùng lặp logic.")
    
    # Cập nhật images.jsonl
    updated_images = []
    for img in images:
        old_id = img['artifact_id']
        if old_id in id_mapping:
            img['artifact_id'] = id_mapping[old_id]
        updated_images.append(img)
        
    # Ghi lại kết quả
    with open(artifacts_file, "w", encoding="utf-8") as f:
        for a in merged_artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open(images_file, "w", encoding="utf-8") as f:
        for img in updated_images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")
            
    logger.info("Hoàn tất ghi đè artifacts.jsonl và images.jsonl sau khi gộp.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_merger()

