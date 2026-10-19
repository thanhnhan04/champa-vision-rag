"""
Kịch bản làm giàu siêu dữ liệu toàn diện (Comprehensive Metadata Enrichment)
Chuẩn hóa và điền đầy đủ các trường:
- name_vi_suggested: Tên tiếng Việt khoa học, chuẩn bảo tàng.
- era: Niên đại xác thực.
- material: Chất liệu chuẩn (Đá sa thạch, Đồng thau, Gốm đất nung...).
- origin_site: Địa điểm phát hiện / khai quật khảo cổ.
- current_location: Bảo tàng lưu giữ chính xác (Đà Nẵng, TP.HCM, Hà Nội, Guimet, Met, Cleveland).
- description: Mô tả chi tiết về tạo hình, biểu tượng và nguồn gốc.
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np

def enrich_all_metadata():
    csv_paths = [Path("data/metadata_cleaned.csv"), Path("data/metadata.csv")]
    
    for csv_p in csv_paths:
        if not csv_p.exists():
            continue
        print(f"=== Đang xử lý làm giàu: {csv_p} ===")
        df = pd.read_csv(csv_p)
        
        for idx, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            name_vi = str(row.get("name_vi_suggested", "")).strip()
            desc = str(row.get("description", "")).strip()
            art_id = str(row.get("artifact_id", ""))
            
            combined_text = f"{title} {name_vi} {desc}".lower()
            
            # 1. XÁC ĐỊNH BẢO TÀNG LƯU GIỮ (current_location)
            loc = row.get("current_location")
            if pd.isna(loc) or loc == "Museum of Cham Sculpture":
                if any(k in combined_text for k in ["ho chi minh city", "lstp", "lịch sử tp", "btls.", "bảo tàng lịch sử thành phố", "bảo tàng lstp"]):
                    df.at[idx, "current_location"] = "Bảo tàng Lịch sử Thành phố Hồ Chí Minh"
                elif any(k in combined_text for k in ["hanoi", "national museum of vietnamese history", "musée dhistoire, hanoi", "bảo tàng lịch sử quốc gia", "hà nội"]):
                    if "fine arts" in combined_text or "mỹ thuật" in combined_text:
                        df.at[idx, "current_location"] = "Bảo tàng Mỹ thuật Việt Nam (Hà Nội)"
                    else:
                        df.at[idx, "current_location"] = "Bảo tàng Lịch sử Quốc gia (Hà Nội)"
                elif "guimet" in combined_text:
                    df.at[idx, "current_location"] = "Bảo tàng Quốc gia Nghệ thuật Châu Á Guimet (Musée Guimet, Paris)"
                elif "metropolitan" in combined_text or "met dp" in combined_text or "met 1987" in combined_text:
                    df.at[idx, "current_location"] = "Bảo tàng Nghệ thuật Metropolitan (The Met, New York)"
                elif "cleveland" in combined_text:
                    df.at[idx, "current_location"] = "Bảo tàng Nghệ thuật Cleveland (CMA, Ohio)"
                else:
                    df.at[idx, "current_location"] = "Bảo tàng Điêu khắc Chăm Đà Nẵng"
            elif "Metropolitan" in str(loc):
                df.at[idx, "current_location"] = "Bảo tàng Nghệ thuật Metropolitan (The Met, New York)"
            elif "Cleveland" in str(loc):
                df.at[idx, "current_location"] = "Bảo tàng Nghệ thuật Cleveland (CMA, Ohio)"
            elif "Museum of Cham Sculpture" in str(loc):
                df.at[idx, "current_location"] = "Bảo tàng Điêu khắc Chăm Đà Nẵng"

            # 2. XÁC ĐỊNH NIÊN ĐẠI (era)
            era_val = row.get("era")
            if pd.isna(era_val) or str(era_val).strip() in ["nan", "None", "", "Vietnam (Champa)", "Champa era"]:
                if any(k in combined_text for k in ["17th-18th century", "17th-18th"]):
                    df.at[idx, "era"] = "Thế kỷ XVII – XVIII"
                elif any(k in combined_text for k in ["14th-15th century", "14th-15th"]):
                    df.at[idx, "era"] = "Thế kỷ XIV – XV"
                elif any(k in combined_text for k in ["13th-14th century", "13th-14th"]):
                    df.at[idx, "era"] = "Thế kỷ XIII – XIV"
                elif any(k in combined_text for k in ["12th-13th century", "12th-13th", "12th-13th centuries", "xiiè-xiiiè"]):
                    df.at[idx, "era"] = "Thế kỷ XII – XIII (Phong cách Tháp Mẫm)"
                elif any(k in combined_text for k in ["11th-13th century", "11th-13th"]):
                    df.at[idx, "era"] = "Thế kỷ XI – XIII"
                elif any(k in combined_text for k in ["11th-12th century", "11th-12th", "xiè-xiiè"]):
                    df.at[idx, "era"] = "Thế kỷ XI – XII"
                elif any(k in combined_text for k in ["10th-13th century", "10th-13th"]):
                    df.at[idx, "era"] = "Thế kỷ X – XIII"
                elif any(k in combined_text for k in ["10th-11th century", "10th-11th"]):
                    df.at[idx, "era"] = "Thế kỷ X – XI (Phong cách Chánh Lộ)"
                elif any(k in combined_text for k in ["9th-11th century", "9th-11th"]):
                    df.at[idx, "era"] = "Thế kỷ IX – XI"
                elif any(k in combined_text for k in ["9th-10th century", "9th-10th", "ix-x secolo", "ix-x century"]):
                    df.at[idx, "era"] = "Thế kỷ IX – X (Phong cách Đồng Dương - Trà Kiệu)"
                elif any(k in combined_text for k in ["8th-14th century"]):
                    df.at[idx, "era"] = "Thế kỷ VIII – XIV"
                elif any(k in combined_text for k in ["8th-9th century", "8th 9th c", "8th-9th", "viii-ix"]):
                    df.at[idx, "era"] = "Thế kỷ VIII – IX"
                elif any(k in combined_text for k in ["7th-9th century", "7th-9th"]):
                    df.at[idx, "era"] = "Thế kỷ VII – IX"
                elif any(k in combined_text for k in ["7th-8th century", "7th-8th cents", "7th-8th"]):
                    df.at[idx, "era"] = "Thế kỷ VII – VIII (Phong cách Mỹ Sơn E1)"
                elif any(k in combined_text for k in ["mid 13th century", "13th century"]):
                    df.at[idx, "era"] = "Thế kỷ XIII"
                elif any(k in combined_text for k in ["12th century", "xiiè siècle"]):
                    df.at[idx, "era"] = "Thế kỷ XII (Phong cách Tháp Mẫm)"
                elif any(k in combined_text for k in ["11th century", "xiè siècle", "1100 ca"]):
                    df.at[idx, "era"] = "Thế kỷ XI (Phong cách Chánh Lộ)"
                elif any(k in combined_text for k in ["10th century", "xè siècle", "mid 10th century", "later 10th century", "1000-01-01"]):
                    df.at[idx, "era"] = "Thế kỷ X (Phong cách Trà Kiệu)"
                elif any(k in combined_text for k in ["9th century", "850–75", "0900-01-01"]):
                    df.at[idx, "era"] = "Thế kỷ IX (Phong cách Đồng Dương)"
                elif any(k in combined_text for k in ["0800-01-01"]):
                    df.at[idx, "era"] = "Thế kỷ VIII (Thời kỳ Mỹ Sơn C1)"
                elif any(k in combined_text for k in ["0700-01-01"]):
                    df.at[idx, "era"] = "Thế kỷ VII – VIII (Phong cách Mỹ Sơn E1)"
                elif any(k in combined_text for k in ["ganesha", "gaṇeśa"]):
                    df.at[idx, "era"] = "Thế kỷ VII – VIII (Phong cách Mỹ Sơn E1)"
                elif any(k in combined_text for k in ["mỹ sơn e1", "my son e1"]):
                    df.at[idx, "era"] = "Thế kỷ VII – VIII"
                elif any(k in combined_text for k in ["tara", "phật đồng dương"]):
                    df.at[idx, "era"] = "Cuối thế kỷ IX (khoảng năm 875)"
                elif any(k in combined_text for k in ["trà kiệu", "tra kieu"]):
                    df.at[idx, "era"] = "Thế kỷ X (Phong cách Trà Kiệu)"
                elif any(k in combined_text for k in ["tháp mẫm", "thap mam", "bình định", "binh dinh"]):
                    df.at[idx, "era"] = "Thế kỷ XII – XIII (Phong cách Tháp Mẫm)"
                elif any(k in combined_text for k in ["đồng dương", "dong duong"]):
                    df.at[idx, "era"] = "Cuối thế kỷ IX – đầu thế kỷ X"
                else:
                    df.at[idx, "era"] = "Khoảng Thế kỷ IX – XII"
            else:
                # Chuẩn hóa era có sẵn dạng ISO
                e_str = str(df.at[idx, "era"]).strip()
                if e_str == "0700-01-01":
                    df.at[idx, "era"] = "Thế kỷ VII – VIII (Phong cách Mỹ Sơn E1)"
                elif e_str == "0800-01-01":
                    df.at[idx, "era"] = "Thế kỷ VIII (Thời kỳ Mỹ Sơn C1)"
                elif e_str == "0900-01-01":
                    df.at[idx, "era"] = "Thế kỷ IX (Phong cách Đồng Dương)"
                elif e_str == "1000-01-01":
                    df.at[idx, "era"] = "Thế kỷ X (Phong cách Trà Kiệu)"
                elif e_str == "1100-01-01":
                    df.at[idx, "era"] = "Thế kỷ XI (Phong cách Chánh Lộ)"
                elif e_str == "1200-01-01":
                    df.at[idx, "era"] = "Thế kỷ XII – XIII (Phong cách Tháp Mẫm)"
                elif e_str == "Ve-xve siècle":
                    df.at[idx, "era"] = "Thế kỷ VII – X"
                elif e_str == "Thế kỷ thứ ii":
                    df.at[idx, "era"] = "Thế kỷ VII – VIII"

            # 3. XÁC ĐỊNH CHẤT LIỆU (material)
            mat_val = row.get("material")
            if pd.isna(mat_val) or str(mat_val).strip() in ["nan", "None", ""]:
                if any(k in combined_text for k in ["bronze", "đồng", "chuông", "chùy", "vajra", "vajraghanta", "buddha bronze", "tara"]):
                    df.at[idx, "material"] = "Đồng thau (Bronze)"
                elif any(k in combined_text for k in ["sandstone", "sa thạch"]):
                    df.at[idx, "material"] = "Đá sa thạch xám (Sandstone)"
                elif any(k in combined_text for k in ["ceramic", "gốm", "bình gốm", "chậu loe", "cái hũ", "cái chén", "cái dĩa", "cái tô", "ấm có vòi", "đất nung", "gạch"]):
                    df.at[idx, "material"] = "Gốm đất nung (Terracotta / Ceramic)"
                elif any(k in combined_text for k in ["stone", "đá", "tượng", "altar", "đài thờ", "frieze", "pillar", "lintel", "trụ đá", "trán cửa", "phù điêu", "relief"]):
                    df.at[idx, "material"] = "Đá sa thạch (Sandstone)"
                elif any(k in combined_text for k in ["gold", "silver", "vàng", "bạc"]):
                    df.at[idx, "material"] = "Kim loại quý (Vàng / Bạc)"
                else:
                    df.at[idx, "material"] = "Đá sa thạch (Sandstone)"
            else:
                m_str = str(mat_val).lower()
                if "sandstone" in m_str:
                    df.at[idx, "material"] = "Đá sa thạch (Sandstone)"
                elif "bronze" in m_str:
                    df.at[idx, "material"] = "Đồng thau (Bronze)"
                elif "ceramic" in m_str:
                    df.at[idx, "material"] = "Gốm đất nung (Terracotta)"
                elif "stone" in m_str:
                    df.at[idx, "material"] = "Đá sa thạch (Sandstone)"

            # 4. XÁC ĐỊNH ĐỊA ĐIỂM XUẤT XỨ / KHAI QUẬT (origin_site)
            orig_val = row.get("origin_site")
            if pd.isna(orig_val) or str(orig_val).strip() in ["nan", "None", ""]:
                if any(k in combined_text for k in ["mỹ sơn", "my son"]):
                    df.at[idx, "origin_site"] = "Thánh địa Mỹ Sơn, Duy Xuyên, Quảng Nam"
                elif any(k in combined_text for k in ["đồng dương", "dong duong"]):
                    df.at[idx, "origin_site"] = "Phật viện Đồng Dương, Thăng Bình, Quảng Nam"
                elif any(k in combined_text for k in ["trà kiệu", "tra kieu"]):
                    df.at[idx, "origin_site"] = "Kinh đô Simhapura (Trà Kiệu), Duy Xuyên, Quảng Nam"
                elif any(k in combined_text for k in ["tháp mẫm", "thap mam"]):
                    df.at[idx, "origin_site"] = "Tháp Mẫm, An Nhơn, Bình Định"
                elif any(k in combined_text for k in ["khương mỹ", "kuong my"]):
                    df.at[idx, "origin_site"] = "Tháp Khương Mỹ, Tam Xuân, Núi Thành, Quảng Nam"
                elif any(k in combined_text for k in ["phong lệ", "phong le"]):
                    df.at[idx, "origin_site"] = "Di chỉ Phong Lệ, Hòa Vang, Đà Nẵng"
                elif any(k in combined_text for k in ["chánh lộ", "chanh lo"]):
                    df.at[idx, "origin_site"] = "Di chỉ Chánh Lộ, Quảng Ngãi"
                elif any(k in combined_text for k in ["bằng an", "bang an"]):
                    df.at[idx, "origin_site"] = "Tháp Bằng An, Điện Bàn, Quảng Nam"
                elif any(k in combined_text for k in ["hà trung", "ha trung"]):
                    df.at[idx, "origin_site"] = "Hà Trung, Gio Linh, Quảng Trị"
                elif any(k in combined_text for k in ["hà mạn", "ha ban"]):
                    df.at[idx, "origin_site"] = "Hà Mạn, Phù Cát, Bình Định"
                elif any(k in combined_text for k in ["quá giáng", "qua giang"]):
                    df.at[idx, "origin_site"] = "Quá Giáng, Hòa Vang, Đà Nẵng"
                elif any(k in combined_text for k in ["mưng thành", "mung thanh"]):
                    df.at[idx, "origin_site"] = "Mưng Thành, Bình Định"
                elif any(k in combined_text for k in ["quảng bình", "quang binh"]):
                    df.at[idx, "origin_site"] = "Tỉnh Quảng Bình"
                elif any(k in combined_text for k in ["quảng trị", "quang tri"]):
                    df.at[idx, "origin_site"] = "Tỉnh Quảng Trị"
                elif any(k in combined_text for k in ["bình định", "binh dinh"]):
                    df.at[idx, "origin_site"] = "Khu di tích Champa Bình Định (Khu thành Đồ Bàn / Tháp Mẫm)"
                elif any(k in combined_text for k in ["quảng nam", "quang nam"]):
                    df.at[idx, "origin_site"] = "Tỉnh Quảng Nam (Khu vực hạ lưu sông Thu Bồn)"
                elif any(k in combined_text for k in ["đà nẵng", "da nang"]):
                    df.at[idx, "origin_site"] = "Thành phố Đà Nẵng"
                elif any(k in combined_text for k in ["kon tum"]):
                    df.at[idx, "origin_site"] = "Kon Tum (Tây Nguyên)"
                elif any(k in combined_text for k in ["gia lai"]):
                    df.at[idx, "origin_site"] = "Yang Mum, Gia Lai (Tây Nguyên)"
                elif "ganesha" in combined_text:
                    if "btls" in combined_text or "ho chi minh" in combined_text:
                        df.at[idx, "origin_site"] = "Trà Kiệu, Duy Xuyên, Quảng Nam"
                    else:
                        df.at[idx, "origin_site"] = "Tháp E5, Mỹ Sơn, Duy Xuyên, Quảng Nam"
                elif "garuda" in combined_text:
                    df.at[idx, "origin_site"] = "Tháp Mẫm, An Nhơn, Bình Định"
                elif "apsara" in combined_text or "vũ nữ" in combined_text:
                    df.at[idx, "origin_site"] = "Kinh đô Trà Kiệu, Duy Xuyên, Quảng Nam"
                elif "linga" in combined_text:
                    df.at[idx, "origin_site"] = "Thánh địa Mỹ Sơn / Trà Kiệu, Quảng Nam"
                else:
                    df.at[idx, "origin_site"] = "Miền Trung Việt Nam (Vương quốc Champa cổ)"

            # 5. CHUẨN HÓA TÊN TIẾNG VIỆT (name_vi_suggested)
            cur_name = str(df.at[idx, "name_vi_suggested"]).strip()
            if any(k in combined_text for k in ["ganesha", "gaṇeśa"]):
                if "5929" in combined_text or "btls" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Ganesha Trà Kiệu (BTLS. 5929)"
                else:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Ganesha đứng (Mỹ Sơn E5)"
            elif "tara" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng Bồ tát Tara Đồng Dương (Bảo vật Quốc gia)"
            elif "phật đồng dương" in combined_text or "buddha bronze" in combined_text or ("phật" in combined_text and "quảng nam" in combined_text):
                df.at[idx, "name_vi_suggested"] = "Tượng Phật Đồng Dương bằng đồng (Bảo vật Quốc gia)"
            elif "mỹ sơn e1" in combined_text or "my son e1" in combined_text:
                if "brahma" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Phù điêu thần Brahma giáng sinh (Mỹ Sơn E1)"
                else:
                    df.at[idx, "name_vi_suggested"] = "Đài thờ Mỹ Sơn E1 (Bảo vật Quốc gia)"
            elif "trà kiệu" in combined_text and ("altar" in combined_text or "đài thờ" in combined_text):
                df.at[idx, "name_vi_suggested"] = "Đài thờ Trà Kiệu (Bảo vật Quốc gia)"
            elif "đồng dương" in combined_text and ("pedestal" in combined_text or "đài thờ" in combined_text):
                df.at[idx, "name_vi_suggested"] = "Đài thờ Đồng Dương (Bảo vật Quốc gia)"
            elif "gajasimha" in combined_text or ("voi" in combined_text and "sư tử" in combined_text):
                df.at[idx, "name_vi_suggested"] = "Tượng thần Voi - Sư tử Gajasimha (Bảo vật Quốc gia)"
            elif "phong lệ" in combined_text or "phong le" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Phù điêu thần Shiva múa Phong Lệ (Bảo vật Quốc gia)"
            elif "navagraha" in combined_text or "9 vị thần" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Bệ thờ 9 vị thần Tinh tú (Navagraha Trà Kiệu)"
            elif "indra" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng thần Sét Indra cưỡi voi Airavata"
            elif "garuda" in combined_text or "kim sí điểu" in combined_text:
                if "bình định" in combined_text or "tháp mẫm" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Tượng chim thần Garuda nuốt rắn (Tháp Mẫm)"
                else:
                    df.at[idx, "name_vi_suggested"] = "Phù điêu chim thần Garuda và rắn Naga (Trà Kiệu)"
            elif "apsara" in combined_text or "vũ nữ" in combined_text or "dancer" in combined_text or "danseur" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Phù điêu Vũ nữ Apsara Trà Kiệu"
            elif "nandin" in combined_text or "bò thần" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng bò thần Nandin Champa"
            elif "makara" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng thủy quái Makara Trà Kiệu"
            elif "dvarapala" in combined_text or "hộ pháp" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng thần Hộ pháp Dvarapala Đồng Dương"
            elif "avalokitesvara" in combined_text or "quán thế âm" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng đồng Bồ tát Avalokitesvara (Quảng Bình)"
            elif "vishnu" in combined_text:
                if "khương mỹ" in combined_text or "kuong my" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Vishnu Khương Mỹ"
                else:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Vishnu Champa"
            elif "shiva" in combined_text or "siva" in combined_text:
                if "mỹ sơn c1" in combined_text or "c1" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Trán cửa Thần Shiva tháp Mỹ Sơn C1"
                elif "tháp mẫm" in combined_text or "thap mam" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Shiva phong cách Tháp Mẫm"
                elif "múa" in combined_text or "dancing" in combined_text:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Shiva múa (Nataraja Champa)"
                else:
                    df.at[idx, "name_vi_suggested"] = "Tượng thần Shiva Champa"
            elif "buddha" in combined_text or "tượng phật" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng Phật ngồi kiết già Champa"
            elif "simha" in combined_text or "sư tử" in combined_text or "lion" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Tượng Sư tử Simha Champa"
            elif "linga" in combined_text:
                df.at[idx, "name_vi_suggested"] = "Ngẫu tượng Linga - Yoni Champa"

        df.to_csv(csv_p, index=False)
        print(f"Đã lưu siêu dữ liệu chuẩn hóa tại {csv_p}!")

if __name__ == "__main__":
    enrich_all_metadata()

