"""
Làm sạch nhãn và gộp thực thể (Entity Resolution & Deduplication)
theo góp ý chuyên môn của hội đồng:
1. Tách các album Flickr/Wikimedia bị gộp sai (Disaggregation: COMMONS_95036088, COMMONS_117983740, COMMONS_95035915).
2. Gộp các mã artifact_id trùng lặp cùng một hiện vật vật lý (The Met vs Commons, BTLS TP.HCM đa tác giả).
3. Loại bỏ ảnh phong cảnh không phải hiện vật (sân bảo tàng).
4. Xuất file data/metadata_cleaned.csv và embeddings/ids_cleaned.json.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bảng ánh xạ gộp mã hiện vật (Canonical Entity Mapping)
EXACT_CANONICAL_MAP = {
    # 1. The Met vs Wikimedia Commons (cùng ảnh / cùng hiện vật Sim = 1.0)
    "COMMONS_57865447": "MET_38905",      # Head of Shiva (lingakosha)
    "COMMONS_57865141": "MET_38645",      # Seated Male Figure
    "COMMONS_57103262": "MET_38314",      # Standing Shiva / Dvarapala
    
    # 2. BTLS TP.HCM: Các ảnh chụp cùng hiện vật nhưng mang mã Commons khác nhau
    "COMMONS_117983858": "COMMONS_40764319",  # Vũ công Trà Kiệu BTLS TP.HCM (DSC06143)
    "COMMONS_149130784": "COMMONS_40764325",  # Shiva, Dai Huu, Binh Dinh, 15th c.
    "COMMONS_5656048": "COMMONS_40593219",    # Shiva, Binh Dinh
    "COMMONS_149131012": "COMMONS_149131017", # Tượng trang phục sứt mẻ
    "COMMONS_149130607": "COMMONS_117983814", # Điêu khắc Sư tử / Nhân sư Champa
    "COMMONS_149130804": "COMMONS_40764312",  # Tượng Lakshmi
    "COMMONS_149130801": "COMMONS_149130801", # Giữ nguyên tượng tu sĩ
}

# Ánh xạ cụ thể từng ảnh trong các album Flickr tổng hợp sang hiện vật tương ứng
SPECIFIC_IMAGE_MAP = {
    # Album Flickr "Champa Stone (9980724305)" - vốn bị gán sai chung mã COMMONS_95036088
    "IMG_95036111": "COMMONS_40764328",  # Trán cửa tu sĩ Po Rome
    "IMG_95036109": "COMMONS_40764312",  # Lakshmi thế kỷ 12-13
    "IMG_95036096": "COMMONS_40764322",  # Bệ tháp Mẫm, Bình Định
    "IMG_95036053": "COMMONS_40764315",  # Lakshmi Trà Kiệu
    "IMG_95036100": "COMMONS_40764326",  # Sư tử Tháp Mẫm
    "IMG_95036093": "COMMONS_40764304",  # Kimnari Tháp Mẫm
    "IMG_95036095": "COMMONS_40764320",  # Trán cửa Tháp Mẫm
    "IMG_95036024": "COMMONS_54853103",  # Dancers (BTLS. 5940 5930)

    # Album Wikimedia "Cổ vật Champa" - vốn bị gán chung COMMONS_117983740
    "IMG_117983813": "COMMONS_40764312",  # Lakshmi thế kỷ 12-13
    "IMG_117983810": "COMMONS_40764315",  # Lakshmi Trà Kiệu
    "IMG_117983815": "COMMONS_149130801", # Tượng tu sĩ
    "IMG_117983811": "COMMONS_149130801", # Tượng tu sĩ (góc khác)
    "IMG_117983816": "COMMONS_117983814", # Nhân sư Champa

    # Album "Champa Ceramics" - vốn bị gán chung COMMONS_95035915
    "IMG_95035928": "COMMONS_51413781",  # Architectural decoration BTLS 22963
    "IMG_95035925": "COMMONS_51413781",  # Architectural decoration BTLS 22964
    "IMG_95035922": "COMMONS_51413476",  # Bowl BTLS 22850
}

# Các ảnh phong cảnh / không phải hiện vật cần loại bỏ
NON_ARTIFACT_IMAGES = {
    "IMG_52933741", # La cour du musée Cham de Da Nang (sân bảo tàng ngoài trời)
    "IMG_24671212", # Le musée de sculpture Cham (toàn cảnh sân bảo tàng)
}

def clean_metadata_and_ids():
    df = pd.read_csv("data/metadata.csv")
    initial_len = len(df)
    
    # 1. Loại bỏ ảnh phong cảnh không phải hiện vật điêu khắc
    df = df[~df["image_id"].isin(NON_ARTIFACT_IMAGES)].copy()
    logger.info(f"Đã loại bỏ {initial_len - len(df)} ảnh phong cảnh sân bảo tàng (không phải điêu khắc).")
    
    # 2. Chuẩn hóa mã artifact_id
    cleaned_rows = []
    album_disagg_counts = 0
    canonical_merged_counts = 0
    
    for _, row in df.iterrows():
        img_id = row["image_id"]
        old_art = row["artifact_id"]
        
        # Nếu có map theo ảnh cụ thể (tách album tổng hợp)
        if img_id in SPECIFIC_IMAGE_MAP:
            new_art = SPECIFIC_IMAGE_MAP[img_id]
            album_disagg_counts += 1
        # Nếu nằm trong album tổng hợp nhưng không có đối ứng -> tạo id riêng rẽ cho từng ảnh
        elif old_art in ["COMMONS_95036088", "COMMONS_117983740", "COMMONS_95035915"]:
            new_art = f"SINGLE_{img_id}"
            album_disagg_counts += 1
        # Nếu nằm trong danh sách gộp canonical
        elif old_art in EXACT_CANONICAL_MAP:
            new_art = EXACT_CANONICAL_MAP[old_art]
            canonical_merged_counts += 1
        else:
            new_art = old_art
            
        row_dict = row.to_dict()
        row_dict["artifact_id_raw"] = old_art
        row_dict["artifact_id"] = new_art
        cleaned_rows.append(row_dict)
        
    cleaned_df = pd.DataFrame(cleaned_rows)
    
    # Lưu metadata sạch
    cleaned_df.to_csv("data/metadata_cleaned.csv", index=False, encoding="utf-8")
    logger.info(f"Đã lưu data/metadata_cleaned.csv ({len(cleaned_df)} ảnh, {cleaned_df['artifact_id'].nunique()} hiện vật).")
    logger.info(f" - Tách album tổng hợp: {album_disagg_counts} ảnh.")
    logger.info(f" - Gộp định danh trùng chéo (The Met / Commons / BTLS): {canonical_merged_counts} ảnh.")
    
    # Cập nhật danh sách embeddings tương ứng (giữ nguyên thứ tự để dùng với file .npy)
    raw_ids = json.load(open("embeddings/ids.json", "r", encoding="utf-8"))
    
    cleaned_ids = []
    valid_indices = []
    for idx, item in enumerate(raw_ids):
        img_id = item["image_id"]
        if img_id in NON_ARTIFACT_IMAGES:
            continue
        valid_indices.append(idx)
        matching_row = cleaned_df[cleaned_df["image_id"] == img_id].iloc[0]
        cleaned_ids.append({
            "image_id": img_id,
            "artifact_id": matching_row["artifact_id"],
            "artifact_id_raw": matching_row["artifact_id_raw"]
        })
        
    Path("embeddings/ids_cleaned.json").write_text(json.dumps(cleaned_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save("embeddings/valid_cleaned_indices.npy", np.array(valid_indices))
    logger.info(f"Đã lưu embeddings/ids_cleaned.json ({len(cleaned_ids)} ảnh) và embeddings/valid_cleaned_indices.npy.")

if __name__ == "__main__":
    clean_metadata_and_ids()

