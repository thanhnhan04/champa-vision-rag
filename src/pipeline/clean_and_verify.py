"""
Script dọn dẹp và chuẩn hóa dữ liệu Champa:
1. Loại bỏ các hiện vật không thuộc Champa (13 hiện vật Pháp từ The Met do trùng từ 'Champ', 1 Angkor từ Commons).
2. Khôi phục các hiện vật Champa xịn bị lọc nhầm trước đó (Sư tử CMA, Bệ thờ 9 vị thần Trà Kiệu, Shiva Guimet).
3. Khử trùng lặp ảnh trong images.jsonl và liên kết chính xác file_path trên ổ đĩa.
4. Xuất báo cáo thống kê chuyên sâu.
"""
import json
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

def run_clean_and_verify():
    art_file = Path("data/processed/artifacts.jsonl")
    img_file = Path("data/processed/images.jsonl")
    img_dir = Path("data/images")
    
    artifacts = [json.loads(line) for line in art_file.read_text('utf-8').strip().split('\n') if line]
    images = [json.loads(line) for line in img_file.read_text('utf-8').strip().split('\n') if line]
    
    # 1. Danh sách ID phi Champa cần loại bỏ
    non_champa_art_ids = {
        'MET_229492', 'MET_380720', 'MET_383731', 'MET_286932', 'MET_222206', 
        'MET_362249', 'MET_34014', 'MET_644408', 'MET_830588', 'MET_336976', 
        'MET_389675', 'MET_680821', 'MET_682098', 'COMMONS_64805124'
    }
    
    clean_artifacts = [a for a in artifacts if a['artifact_id'] not in non_champa_art_ids]
    
    # 2. Khôi phục CMA_150722 (Sư tử Champa)
    cma_file = Path("data/raw/cma/150722.json")
    if cma_file.exists():
        from src.sources.cma import parse_cma_item
        art_cma, imgs_cma = parse_cma_item(json.loads(cma_file.read_text('utf-8')))
        if art_cma and not any(a['artifact_id'] == art_cma.artifact_id for a in clean_artifacts):
            clean_artifacts.append(art_cma.model_dump(mode='json'))
            for im in imgs_cma:
                images.append(im.model_dump(mode='json'))
                
    # 3. Khôi phục các hiện vật Commons bị lọc oan
    from src.sources.commons import parse_commons_file
    allowed_licenses = ['cc-by-sa-4.0', 'cc-by-4.0', 'cc-by-3.0', 'cc-by-2.0', 'cc0-1.0', 'publicdomain', 'pd']
    for pid in ['40764307', '37214323']:
        p_file = Path(f"data/raw/commons/{pid}.json")
        if p_file.exists():
            art_c, img_c = parse_commons_file(json.loads(p_file.read_text('utf-8')), allowed_licenses)
            if art_c and not any(a['artifact_id'] == art_c.artifact_id for a in clean_artifacts):
                clean_artifacts.append(art_c.model_dump(mode='json'))
                if img_c:
                    images.append(img_c.model_dump(mode='json'))
                    
    valid_art_ids = {a['artifact_id'] for a in clean_artifacts}
    
    # 4. Khử trùng lặp images.jsonl
    seen_imgs = set()
    clean_images = []
    for img in images:
        iid = img['image_id']
        aid = img['artifact_id']
        if aid in valid_art_ids and iid not in seen_imgs:
            seen_imgs.add(iid)
            # Khớp file_path vật lý
            fp = img.get('file_path')
            if not fp or not Path(fp).exists():
                possible = list(img_dir.glob(f"{iid}.*"))
                if possible and possible[0].stat().st_size > 0:
                    img['file_path'] = str(possible[0].as_posix())
            clean_images.append(img)
            
    # Ghi đè file chuẩn hóa
    with open(art_file, 'w', encoding='utf-8') as f:
        for a in clean_artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
            
    with open(img_file, 'w', encoding='utf-8') as f:
        for i in clean_images:
            f.write(json.dumps(i, ensure_ascii=False) + '\n')
            
    logger.info(f"Đã làm sạch và chuẩn hóa: {len(clean_artifacts)} hiện vật, {len(clean_images)} hình ảnh.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_clean_and_verify()

