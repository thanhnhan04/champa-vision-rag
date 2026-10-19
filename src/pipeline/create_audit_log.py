"""
Tạo Bảng Kiểm toán Làm sạch Dữ liệu & Phân rã Thực thể (Label Audit Log).
Bảo đảm tính minh bạch, phi vòng lặp (Non-circular) và có căn cứ ngoại sinh rõ ràng.
"""
import pandas as pd
from pathlib import Path

AUDIT_RECORDS = [
    # 1. Loại bỏ ảnh phong cảnh sân bảo tàng (không phải điêu khắc)
    {
        "image_id": "IMG_52933741",
        "original_artifact_id": "COMMONS_52933741",
        "cleaned_artifact_id": "EXCLUDED",
        "action": "REMOVE_NON_ARTIFACT",
        "audit_basis": "Visual Inspection",
        "notes": "Ảnh toàn cảnh sân vườn ngoài trời Bảo tàng Điêu khắc Chăm Đà Nẵng, không phải hiện vật điêu khắc."
    },
    {
        "image_id": "IMG_24671212",
        "original_artifact_id": "COMMONS_24671212",
        "cleaned_artifact_id": "EXCLUDED",
        "action": "REMOVE_NON_ARTIFACT",
        "audit_basis": "Visual Inspection",
        "notes": "Ảnh toàn cảnh hành lang sân bảo tàng, không phải hiện vật điêu khắc cụ thể."
    },

    # 2. Gộp thực thể The Met và Wikimedia Commons (Trùng lặp ngoại sinh)
    {
        "image_id": "IMG_57865447",
        "original_artifact_id": "COMMONS_57865447",
        "cleaned_artifact_id": "MET_38905",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "The Met Accession No. 2002.502",
        "notes": "Đầu tượng thần Shiva (lingakosha), Wikimedia lấy ảnh từ The Met Open Access (DP291638)."
    },
    {
        "image_id": "IMG_57865141",
        "original_artifact_id": "COMMONS_57865141",
        "cleaned_artifact_id": "MET_38645",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "The Met Accession No. 1987.218.17",
        "notes": "Tượng Seated Male Figure, Wikimedia Commons lấy ảnh từ The Met Open Access (1987 218 17 B)."
    },
    {
        "image_id": "IMG_57103262",
        "original_artifact_id": "COMMONS_57103262",
        "cleaned_artifact_id": "MET_38314",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "The Met Accession No. 1987.218.1",
        "notes": "Tượng Standing Shiva / Dvarapala, Wikimedia Commons lấy ảnh từ The Met Open Access (DP311726)."
    },

    # 3. Phân rã 33 ảnh album Flickr "Champa Stone (9980724305)"
    {
        "image_id": "IMG_95036111",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764328",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Po Rome)",
        "notes": "Trán cửa tu sĩ Po Rome tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764328."
    },
    {
        "image_id": "IMG_95036109",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764312",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Lakshmi)",
        "notes": "Tượng nữ thần Lakshmi thế kỷ XII-XIII tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764312."
    },
    {
        "image_id": "IMG_95036096",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764322",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Tháp Mẫm base)",
        "notes": "Khối gờ chỉ chân bệ Tháp Mẫm tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764322."
    },
    {
        "image_id": "IMG_95036053",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764315",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Lakshmi Trà Kiệu)",
        "notes": "Tượng Lakshmi phong cách Trà Kiệu thế kỷ X tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764315."
    },
    {
        "image_id": "IMG_95036100",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764326",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Lion Tháp Mẫm)",
        "notes": "Tượng sư tử Tháp Mẫm thế kỷ XII-XIII tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764326."
    },
    {
        "image_id": "IMG_95036093",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764304",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Kimnari)",
        "notes": "Tượng Kimnari nửa người nửa chim Tháp Mẫm tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764304."
    },
    {
        "image_id": "IMG_95036095",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_40764320",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Antefix ascetic)",
        "notes": "Khối trang trí antefix tu sĩ Tháp Mẫm tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764320."
    },
    {
        "image_id": "IMG_95036024",
        "original_artifact_id": "COMMONS_95036088",
        "cleaned_artifact_id": "COMMONS_54853103",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Inventory No. BTLS. 5940 5930",
        "notes": "Phù điêu vũ nữ Champa số hiệu BTLS. 5940 5930 tại BTLS TP.HCM, đối chiếu khớp với COMMONS_54853103."
    },

    # 4. Phân rã các ảnh trong album Wikimedia COMMONS_117983740
    {
        "image_id": "IMG_117983813",
        "original_artifact_id": "COMMONS_117983740",
        "cleaned_artifact_id": "COMMONS_40764312",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Lakshmi)",
        "notes": "Tượng Lakshmi thế kỷ XII-XIII, đối chiếu khớp với COMMONS_40764312."
    },
    {
        "image_id": "IMG_117983810",
        "original_artifact_id": "COMMONS_117983740",
        "cleaned_artifact_id": "COMMONS_40764315",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Gallery Label (Lakshmi Trà Kiệu)",
        "notes": "Tượng Lakshmi Trà Kiệu, đối chiếu khớp với COMMONS_40764315."
    },
    {
        "image_id": "IMG_117983815",
        "original_artifact_id": "COMMONS_117983740",
        "cleaned_artifact_id": "COMMONS_149130801",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Visual Inspection",
        "notes": "Tượng tu sĩ khổ hạnh, đối chiếu khớp với COMMONS_149130801."
    },
    {
        "image_id": "IMG_117983811",
        "original_artifact_id": "COMMONS_117983740",
        "cleaned_artifact_id": "COMMONS_149130801",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Visual Inspection",
        "notes": "Tượng tu sĩ khổ hạnh góc nghiêng, đối chiếu khớp với COMMONS_149130801."
    },
    {
        "image_id": "IMG_117983816",
        "original_artifact_id": "COMMONS_117983740",
        "cleaned_artifact_id": "COMMONS_117983814",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Visual Inspection",
        "notes": "Tượng nhân sư Champa, đối chiếu khớp với COMMONS_117983814."
    },

    # 5. Gộp các ảnh cùng hiện vật tại BTLS TP.HCM mang mã Commons phân mảnh
    {
        "image_id": "IMG_117983858",
        "original_artifact_id": "COMMONS_117983858",
        "cleaned_artifact_id": "COMMONS_40764319",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "Museum Inventory No. BTLS. 5912-5913",
        "notes": "Tượng vũ công Trà Kiệu thế kỷ X tại BTLS TP.HCM (DSC06143), hai nhiếp ảnh gia đặt tên khác nhau."
    },
    {
        "image_id": "IMG_149130784",
        "original_artifact_id": "COMMONS_149130784",
        "cleaned_artifact_id": "COMMONS_40764325",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "Museum Gallery Label (Shiva Đại Hữu)",
        "notes": "Tượng Shiva Đại Hữu thế kỷ XV tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764325."
    },
    {
        "image_id": "IMG_5656048",
        "original_artifact_id": "COMMONS_5656048",
        "cleaned_artifact_id": "COMMONS_40593219",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "Museum Gallery Label (Shiva Bình Định)",
        "notes": "Tượng Shiva Bình Định tại Bảo tàng Lịch sử Quốc gia, đối chiếu khớp với COMMONS_40593219."
    },
    {
        "image_id": "IMG_149130804",
        "original_artifact_id": "COMMONS_149130804",
        "cleaned_artifact_id": "COMMONS_40764312",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "Museum Gallery Label (Lakshmi)",
        "notes": "Tượng Lakshmi thế kỷ XII-XIII tại BTLS TP.HCM, đối chiếu khớp với COMMONS_40764312."
    },
    {
        "image_id": "IMG_149130607",
        "original_artifact_id": "COMMONS_149130607",
        "cleaned_artifact_id": "COMMONS_117983814",
        "action": "MERGE_CANONICAL_MUSEUM",
        "audit_basis": "Visual Inspection",
        "notes": "Điêu khắc sư tử / nhân sư Champa tại BTLS TP.HCM, đối chiếu khớp với COMMONS_117983814."
    },
    {
        "image_id": "IMG_95035928",
        "original_artifact_id": "COMMONS_95035915",
        "cleaned_artifact_id": "COMMONS_51413781",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Inventory No. BTLS. 22963",
        "notes": "Đồ án kiến trúc gốm BTLS. 22963 tại BTLS TP.HCM, đối chiếu khớp với COMMONS_51413781."
    },
    {
        "image_id": "IMG_95035922",
        "original_artifact_id": "COMMONS_95035915",
        "cleaned_artifact_id": "COMMONS_51413476",
        "action": "DISAGGREGATE_ALBUM",
        "audit_basis": "Museum Inventory No. BTLS. 22850",
        "notes": "Bát gốm Champa BTLS. 22850 tại BTLS TP.HCM, đối chiếu khớp với COMMONS_51413476."
    }
]

def generate_audit_log():
    # Các ảnh còn lại trong 2 album Flickr/Wikimedia không có đối chứng -> gán mã độc lập SINGLE_<image_id>
    # để bảo đảm chúng không bị gộp sai thành cùng 1 hiện vật
    df_raw = pd.read_csv("data/metadata.csv")
    existing_audited = {r["image_id"] for r in AUDIT_RECORDS}
    
    for _, row in df_raw.iterrows():
        img_id = row["image_id"]
        art_id = row["artifact_id"]
        if img_id not in existing_audited and art_id in ["COMMONS_95036088", "COMMONS_117983740", "COMMONS_95035915"]:
            AUDIT_RECORDS.append({
                "image_id": img_id,
                "original_artifact_id": art_id,
                "cleaned_artifact_id": f"SINGLE_{img_id}",
                "action": "DISAGGREGATE_ALBUM",
                "audit_basis": "Album Disaggregation (Unmatched single item)",
                "notes": f"Ảnh trong album tổng hợp {art_id} không có ảnh chụp đối chứng, phân rã thành hiện vật đơn để tránh gán ground-truth sai."
            })
            
    df_audit = pd.DataFrame(AUDIT_RECORDS)
    out_file = Path("data/label_audit_log.csv")
    df_audit.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Đã lưu bảng kiểm toán nhãn vào {out_file} ({len(df_audit)} bản ghi kiểm toán).")
    print(df_audit["action"].value_counts().to_string())

if __name__ == "__main__":
    generate_audit_log()

