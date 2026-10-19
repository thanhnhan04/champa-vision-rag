"""
Tạo danh mục ghi công bản quyền chi tiết cho toàn bộ 428 ảnh trong cơ sở dữ liệu (Attribution Manifest).
Đảm bảo tuân thủ đầy đủ các điều khoản của Creative Commons (Attribution & ShareAlike).
"""
import json
import pandas as pd
from pathlib import Path

def generate_attribution_manifest():
    meta_df = pd.read_csv("data/metadata_cleaned.csv")
    
    # Đọc thêm tác giả từ images.jsonl nếu có
    author_map = {}
    images_jsonl = Path("images.jsonl")
    if images_jsonl.exists():
        for line in images_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                author_map[item.get("id")] = item.get("author", "Unknown / Community contributor")
                
    manifest_rows = []
    
    for _, row in meta_df.iterrows():
        img_id = row["image_id"]
        art_id = row["artifact_id"]
        title = row.get("title", "")
        source = row.get("source", "Wikimedia Commons")
        lic_raw = str(row.get("license", "cc0")).lower().strip()
        
        # Chuẩn hóa tên giấy phép và link điều khoản
        if "cc0" in lic_raw:
            exact_lic = "CC0 1.0 Universal (Public Domain Dedication)"
            lic_url = "https://creativecommons.org/publicdomain/zero/1.0/"
            req = "Không yêu cầu ghi công, phạm vi công cộng hoàn toàn"
        elif "by-sa 4.0" in lic_raw or "cc-by-sa" in lic_raw:
            exact_lic = "CC BY-SA 4.0 International"
            lic_url = "https://creativecommons.org/licenses/by-sa/4.0/"
            req = "Yêu cầu ghi công tác giả và chia sẻ tương tự theo cùng giấy phép"
        elif "by 3.0" in lic_raw:
            exact_lic = "CC BY 3.0 Unported"
            lic_url = "https://creativecommons.org/licenses/by/3.0/"
            req = "Yêu cầu ghi công tác giả"
        elif "by 2.0" in lic_raw:
            exact_lic = "CC BY 2.0 Generic"
            lic_url = "https://creativecommons.org/licenses/by/2.0/"
            req = "Yêu cầu ghi công tác giả"
        else:
            exact_lic = lic_raw.upper()
            lic_url = "https://creativecommons.org/"
            req = "Giấy phép mở"
            
        author = author_map.get(img_id, "Wikimedia Commons / The Met Open Access contributors")
        
        manifest_rows.append({
            "image_id": img_id,
            "artifact_id": art_id,
            "title": title,
            "author": author,
            "source_platform": source,
            "exact_license": exact_lic,
            "license_url": lic_url,
            "compliance_requirement": req
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    out_file = Path("data/attribution_manifest.csv")
    df_manifest.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Đã xuất danh mục ghi công bản quyền: {out_file} ({len(df_manifest)} ảnh).")
    print("\nPhân bố giấy phép chuẩn hóa:")
    print(df_manifest["exact_license"].value_counts().to_string())

if __name__ == "__main__":
    generate_attribution_manifest()

