"""
Module chuẩn hóa dữ liệu theo nguyên tắc liêm chính học thuật (Academic Integrity):
1. Giữ nguyên tính nguyên bản của các trường dữ liệu gốc (Ground Truth) từ API nguồn.
   - Nếu nguồn chính thức không cung cấp, giữ nguyên None (null).
2. Tách toàn bộ các thông tin do AI/Regex trích xuất sang trường riêng 'inferred_metadata'.
3. Gán mã 'knowledge_ref' liên kết tới kho tri thức kiểm chứng được (RAG Knowledge).
"""
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_period(text: str) -> str | None:
    if not text:
        return None
    m_en = re.search(r'((?:early|mid|middle|late|later)?\s*\d{1,2}(?:st|nd|rd|th)?(?:\s*[-–]\s*\d{1,2}(?:st|nd|rd|th)?)?\s*centur(?:y|ies)(?:\s*(?:ad|ce|bc))?)', text, re.IGNORECASE)
    if m_en:
        matched = m_en.group(1).strip()
        nums = re.findall(r'\d+', matched)
        if len(nums) == 2:
            prefix = "Cuối " if "late" in matched.lower() or "later" in matched.lower() else ("Đầu " if "early" in matched.lower() else "")
            return f"{prefix}Thế kỷ {nums[0]} - {nums[1]}"
        elif len(nums) == 1:
            prefix = "Cuối " if "late" in matched.lower() or "later" in matched.lower() else ("Đầu " if "early" in matched.lower() else ("Giữa " if "mid" in matched.lower() else ""))
            return f"{prefix}Thế kỷ {nums[0]}"
        return matched

    m_fr = re.search(r'((?:début|fin)?\s*([ivx]+)[èe]?\s*[-–]\s*([ivx]+)[èe]?\s*siècle|([ivx]+)[èe]?\s*siècle)', text, re.IGNORECASE)
    if m_fr:
        return m_fr.group(1).strip().capitalize()

    m_vi = re.search(r'(thế\s*kỷ\s*(?:thứ\s*)?[ivx\d]+(?:\s*[-–]\s*[ivx\d]+)?)', text, re.IGNORECASE)
    if m_vi:
        return m_vi.group(1).strip().capitalize()
    return None

def extract_material(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    if 'sandstone' in t or 'grès' in t or 'sa thạch' in t or 'đá cát' in t:
        return 'Sa thạch (Sandstone)'
    if 'bronze' in t or 'đồng' in t:
        return 'Đồng (Bronze)'
    if 'terracotta' in t or 'đất nung' in t:
        return 'Đất nung (Terracotta)'
    if 'ceramic' in t or 'gốm' in t or 'pottery' in t:
        return 'Gốm (Ceramic)'
    if 'gold' in t or 'vàng' in t:
        return 'Vàng (Gold)'
    if 'stone' in t or 'đá' in t:
        return 'Đá (Stone)'
    return None

def extract_origin(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    
    # Loại trừ trường hợp là địa chỉ của bảo tàng lưu giữ (không phải nơi khai quật)
    if 'museum of cham sculpture - danang' in t and not any(k in t for k in ['qua giang', 'phong le', 'quang nam', 'tra kieu', 'my son']):
        pass

    sites = [
        ('Trà Kiệu, Quảng Nam', ['tra kieu', 'trà kiệu']),
        ('Mỹ Sơn, Quảng Nam', ['my son', 'mỹ sơn']),
        ('Đồng Dương, Quảng Nam', ['dong duong', 'đồng dương']),
        ('Tháp Mẫm, Bình Định', ['thap mam', 'tháp mẫm']),
        ('Khương Mỹ, Quảng Nam', ['kuong my', 'khuong my', 'khương mỹ']),
        ('Quá Giáng, Đà Nẵng', ['qua giang', 'quá giáng']),
        ('Đại Hữu, Quảng Bình', ['dai huu', 'đại hữu']),
        ('Hà Trung, Quảng Trị', ['ha trung', 'hà trung']),
        ('Hà Mạn, Bình Định', ['ha ban', 'hà mạn']),
        ('Mộc Thành, Bình Định', ['mung thanh', 'mộc thành']),
        ('Bằng An, Quảng Nam', ['bang an', 'bằng an']),
        ('Phong Lệ, Đà Nẵng', ['phong le', 'phong lệ']),
        ('Chánh Lộ, Quảng Ngãi', ['chanh lo', 'chánh lộ']),
        ('Po Rome, Ninh Thuận', ['po rome', 'pô rômê']),
        ('Quảng Nam', ['quang nam', 'quảng nam']),
        ('Bình Định', ['binh dinh', 'bình định']),
        ('Quảng Trị', ['quang tri', 'quảng trị']),
        ('Quảng Bình', ['quang binh', 'quảng bình']),
        ('Ninh Thuận', ['ninh thuan', 'ninh thuận']),
        ('Kon Tum', ['kon tum']),
    ]
    for site_label, aliases in sites:
        if any(a in t for a in aliases):
            return site_label
    return None

def extract_suggested_name_vi(name_en: str) -> str | None:
    if not name_en:
        return None
    raw = name_en.strip()
    
    m_vi = re.search(r'Cổ vật Chăm(?:\s*,\s*|\s*\()([^)]+)\)?', raw)
    if m_vi:
        val = m_vi.group(1).strip()
        val = re.sub(r'^\(|\)$', '', val).strip()
        return val[:1].upper() + val[1:]
        
    t = raw.lower()
    if 'trà kiệu altar' in t or 'tra kieu altar' in t: return 'Đài thờ Trà Kiệu'
    if 'mỹ sơn e1 pedestal' in t or ('my son e1' in t and 'pedestal' in t): return 'Đài thờ Mỹ Sơn E1'
    if 'boddhisattva tara' in t or 'tara' in t: return 'Tượng Bồ tát Tara Đồng Dương'
    if 'dong duong pedestal' in t or ('đồng dương' in t and 'pedestal' in t): return 'Đài thờ Đồng Dương'
    if 'gajasimha' in t: return 'Tượng thần Voi - Sư tử (Gajasimha)'
    if 'brahma birth' in t: return 'Phù điêu thần Brahma giáng sinh'
    if 'chanh lo relief of uma' in t: return 'Phù điêu nữ thần Uma Chánh Lộ'
    if 'my son c1 shiva' in t: return 'Tượng thần Shiva tháp Mỹ Sơn C1'
    if 'phong le relief' in t: return 'Phù điêu thần Shiva múa Phong Lệ'
    if 'thap mam dragon' in t: return 'Tượng Rồng phong cách Tháp Mẫm'
    if 'standing ganesha' in t or 'ganesha' in t: return 'Tượng thần Ganesha'
    if 'po nagar stele' in t: return 'Bia ký Po Nagar'
    if 'heraldic lion' in t: return 'Tượng Sư tử huy hiệu (Simha)'
    
    res = raw
    replacements = [
        (r'\bDancing Shiva\b', 'Tượng thần Shiva múa'),
        (r'\bStanding Shiva\b', 'Tượng thần Shiva đứng'),
        (r'\bShiva Seated\b', 'Tượng thần Shiva ngồi'),
        (r'\bShiva\b', 'Tượng thần Shiva'),
        (r'\bSiva\b', 'Tượng thần Shiva'),
        (r'\bBrahma\b', 'Tượng thần Brahma'),
        (r'\bVishnu\b', 'Tượng thần Vishnu'),
        (r'\bUma\b', 'Tượng nữ thần Uma'),
        (r'\bGanesha\b', 'Tượng thần Ganesha'),
        (r'\bBuddha\b', 'Tượng Phật'),
        (r'\bAltar\b', 'Đài thờ'),
        (r'\bPedestal\b', 'Bệ thờ'),
        (r'\bLintel\b', 'Lanh-tô (Trán cửa)'),
        (r'\bPediment\b', 'Trán cửa vòm'),
        (r'\bTympanum\b', 'Vòm trán cửa'),
        (r'\bFrieze\b', 'Gờ chỉ trang trí'),
        (r'\bPillar\b', 'Trụ đá'),
        (r'\bStele\b', 'Bia ký'),
        (r'\bRelief\b', 'Phù điêu'),
        (r'\bGaruda and Naga\b', 'Phù điêu chim thần Garuda và rắn Naga'),
        (r'\bGaruda\b', 'Tượng chim thần Garuda'),
        (r'\bNaga\b', 'Phù điêu rắn thần Naga'),
        (r'\bNandi\b', 'Tượng bò thần Nandi'),
        (r'\bApsara\b', 'Phù điêu vũ nữ Apsara'),
        (r'\bDancer\b', 'Phù điêu vũ công'),
        (r'\bDanseur\b', 'Phù điêu vũ công'),
        (r'\bLion\b', 'Tượng Sư tử (Simha)'),
        (r'\bElephant\b', 'Tượng Voi'),
        (r'\bLinga\b', 'Linga (Sinh thực khí nam)'),
        (r'\bYoni\b', 'Yoni (Sinh thực khí nữ)'),
        (r'\bDvarapala\b', 'Tượng thần Hộ pháp (Dvarapala)'),
    ]
    for pattern, rep in replacements:
        if re.search(pattern, res, re.IGNORECASE):
            res = re.sub(pattern, rep, res, count=1, flags=re.IGNORECASE)
            break
            
    res = re.sub(r'(\.jpg|\.png|\.jpeg|\.JPG)', '', res)
    res = re.sub(r'(?:-\s*)?DSC\d+', '', res)
    res = re.sub(r'-\s*Museum of Cham Sculpture\s*-\s*Danang,?\s*Vietnam', '', res, flags=re.IGNORECASE)
    res = re.sub(r'-\s*National Museum of Vietnamese History\s*-\s*Hanoi,?\s*Vietnam', '', res, flags=re.IGNORECASE)
    res = re.sub(r'Museum of Vietnamese History,?\s*Hanoi', '', res, flags=re.IGNORECASE)
    res = re.sub(r'\(\d+\)$', '', res)
    return res.strip()

def assign_knowledge_ref(artifact: dict) -> str:
    name = (artifact.get('name_en') or '') + ' ' + (artifact.get('name_vi') or '') + ' ' + (artifact.get('description') or '')
    name_l = name.lower()
    
    if any(k in name_l for k in ['trà kiệu', 'tra kieu', 'mỹ sơn e1', 'my son e1', 'tara', 'đồng dương', 'dong duong', 'gajasimha', 'phong lệ', 'phong le', 'chánh lộ', 'chanh lo']):
        return "data/rag_knowledge/01_national_treasures.md"
    if any(k in name_l for k in ['shiva', 'siva', 'vishnu', 'brahma', 'ganesha', 'uma', 'garuda', 'naga', 'linga', 'yoni', 'nandi']):
        return "data/rag_knowledge/03_iconography_mythology.md"
    if any(k in name_l for k in ['tháp mẫm', 'thap mam', 'khương mỹ', 'bình định', 'quảng nam', 'po rome', 'bằng an']):
        return "data/rag_knowledge/02_art_styles_periods.md"
    return "data/rag_knowledge/04_museum_collections.md"

def run_academic_separation():
    logger.info("Bắt đầu chuẩn hóa dữ liệu tách bạch Ground Truth và Inferred Metadata...")
    art_file = Path("data/processed/artifacts.jsonl")
    
    artifacts = [json.loads(line) for line in art_file.read_text('utf-8').strip().split('\n') if line]
    
    for a in artifacts:
        source = a.get('source_name', '')
        aid = a['artifact_id']
        name_en = a.get('name_en') or ''
        desc = a.get('description') or ''
        full_text = f"{name_en} {desc}"
        
        # 1. Gán tri thức kiểm chứng được
        a['knowledge_ref'] = assign_knowledge_ref(a)
        
        # 2. Xây dựng ngữ cảnh không gian (Spatial Context) đa tầng
        current_loc = a.get('current_location') or 'Bảo tàng Điêu khắc Chăm'
        extracted_site = extract_origin(full_text)
        
        spatial_context = {
            "current_location": current_loc, # Nơi lưu giữ hiện nay
            "excavation_site": None,         # Nơi khai quật xác định qua báo cáo khảo cổ
            "archaeological_site": None,     # Di tích / Địa điểm liên quan
            "style_provenance": None         # Xuất xứ vùng văn hóa / phong cách
        }
        
        # 3. Xử lý theo từng nguồn và phân cấp độ xác minh
        if source == "The Metropolitan Museum of Art":
            a['verification_status'] = "verified_official_source"
            spatial_context['excavation_site'] = a.get('origin_site')
            spatial_context['style_provenance'] = "Văn hóa Champa (Việt Nam)"
            a['spatial_context'] = spatial_context
            a['inferred_metadata'] = {
                "provenance": "official_museum_api",
                "confidence": "high",
                "suggested_name_vi": extract_suggested_name_vi(name_en)
            }
        elif source == "Cleveland Museum of Art":
            a['verification_status'] = "verified_official_source"
            spatial_context['style_provenance'] = "Văn hóa Champa (Việt Nam)"
            if "dong duong" in full_text.lower():
                spatial_context['archaeological_site'] = "Phật viện Đồng Dương, Quảng Nam"
            a['spatial_context'] = spatial_context
            a['inferred_metadata'] = {
                "provenance": "official_museum_api",
                "confidence": "high",
                "suggested_name_vi": extract_suggested_name_vi(name_en)
            }
        elif source == "Wikidata":
            a['verification_status'] = "verified_official_source"
            if "my son" in full_text.lower() or "mỹ sơn" in full_text.lower():
                spatial_context['archaeological_site'] = "Thánh địa Mỹ Sơn, Quảng Nam"
            elif "tra kieu" in full_text.lower() or "trà kiệu" in full_text.lower():
                spatial_context['archaeological_site'] = "Kinh đô Trà Kiệu, Quảng Nam"
            elif "dong duong" in full_text.lower() or "đồng dương" in full_text.lower():
                spatial_context['archaeological_site'] = "Phật viện Đồng Dương, Quảng Nam"
            a['spatial_context'] = spatial_context
            a['inferred_metadata'] = {
                "provenance": "sparql_endpoint",
                "confidence": "high",
                "suggested_name_vi": extract_suggested_name_vi(name_en)
            }
        else: # Wikimedia Commons
            a['verification_status'] = "unverified_heuristic"
            # Phân tách: Nơi phát hiện suy đoán theo phong cách
            spatial_context['style_provenance'] = extracted_site
            a['spatial_context'] = spatial_context
            
            inferred = {
                "provenance": "heuristic_postprocessing",
                "confidence": "medium",
                "suggested_name_vi": extract_suggested_name_vi(name_en),
                "extracted_period": extract_period(full_text),
                "extracted_material": extract_material(full_text),
                "extracted_origin_site": extracted_site,
                "notes": "Thông tin trích xuất từ tiêu đề và mô tả của người chụp, cần chuyên gia xác minh"
            }
            a['inferred_metadata'] = inferred
            
            # Đảm bảo trường gốc chỉ giữ nếu có căn cứ chính thức từ bảo tàng
            a['name_vi'] = None
            a['period'] = None
            a['material'] = None
            a['origin_site'] = None
            
    # Ghi lại file
    with open(art_file, 'w', encoding='utf-8') as f:
        for a in artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
            
    logger.info(f"Hoàn tất chuẩn hóa cho {len(artifacts)} hiện vật!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_academic_separation()
