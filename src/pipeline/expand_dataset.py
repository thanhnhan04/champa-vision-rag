"""
Script Mở Rộng Dữ Liệu Hiện Vật Champa & Khử Trùng Lặp Đa Tầng (Multi-tier Deduplication):
1. Quét mở rộng các nguồn giấy phép mở:
   - Cleveland Museum of Art (CMA Open Access CC0)
   - The Metropolitan Museum of Art (The Met Open Access CC0)
   - Wikimedia Commons (Các danh mục chuyên sâu về hiện vật bảo tàng Đà Nẵng, BTLS, Mỹ Sơn, Trà Kiệu, Đồng Dương, Guimet, AIC)
2. Tải và kiểm tra tính toàn vẹn của tệp ảnh (Pillow, min edge >= 256px).
3. Khử trùng lặp đa tầng nghiêm ngặt:
   - Trùng tuyệt đối (Exact match): Mã băm SHA-256.
   - Trùng tri giác (Perceptual match): Mã băm pHash với khoảng cách Hamming <= 4.
   - Khử trùng nội bộ lô mới (Intra-batch dedup) giữ ảnh có độ phân giải cao hơn.
4. Cập nhật đồng bộ vào metadata.csv và metadata_cleaned.csv.
"""
import os
import sys
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import httpx
from PIL import Image
import imagehash
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USER_AGENT = "ChampaBot/0.2 (Educational research project on Champa sculpture; contact: student@edu.vn)"

class RateLimitError(Exception):
    pass

class DatasetExpander:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.images_dir = self.root / "data" / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_cleaned_path = self.root / "data" / "metadata_cleaned.csv"
        self.meta_path = self.root / "data" / "metadata.csv"
        
        self.client = httpx.Client(
            headers={'User-Agent': USER_AGENT},
            timeout=30.0,
            follow_redirects=True
        )
        
        # Nạp dữ liệu hiện có để chống trùng
        self.existing_df = pd.read_csv(self.meta_cleaned_path)
        self.existing_sha256 = set(self.existing_df['sha256'].dropna().astype(str).str.lower())
        self.existing_phashes = []
        for ph in self.existing_df['phash'].dropna().astype(str):
            try:
                self.existing_phashes.append(imagehash.hex_to_hash(ph))
            except Exception:
                pass
                
        self.existing_image_ids = set(self.existing_df['image_id'].dropna().astype(str))
        self.existing_urls = set(self.existing_df['image_path'].dropna().astype(str).str.lower())
        
        logger.info(f"Đã nạp {len(self.existing_df)} bản ghi hiện có:")
        logger.info(f" - Số mã băm SHA-256 độc nhất: {len(self.existing_sha256)}")
        logger.info(f" - Số mã băm pHash chuẩn bị kiểm tra: {len(self.existing_phashes)}")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def fetch_json(self, url: str, params: dict = None) -> dict:
        time.sleep(1.0)
        resp = self.client.get(url, params=params)
        if resp.status_code in (429, 403):
            raise RateLimitError(f"Rate limited: {resp.status_code}")
        resp.raise_for_status()
        return resp.json()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def download_image_bytes(self, url: str) -> bytes:
        time.sleep(1.0)
        resp = self.client.get(url)
        if resp.status_code in (429, 403):
            raise RateLimitError(f"Rate limited: {resp.status_code}")
        resp.raise_for_status()
        return resp.content

    def check_duplicate(self, sha256_val: str, phash_val: imagehash.ImageHash) -> Tuple[bool, str]:
        """Kiểm tra xem ảnh có bị trùng lặp với dataset hiện có không."""
        if sha256_val.lower() in self.existing_sha256:
            return True, "Trùng tuyệt đối (Exact SHA-256)"
            
        for existing_ph in self.existing_phashes:
            dist = phash_val - existing_ph
            if dist <= 4:
                return True, f"Trùng tri giác (pHash Hamming distance={dist} <= 4)"
                
        return False, ""

    def collect_cma_candidates(self) -> List[Dict[str, Any]]:
        """Thu thập các hiện vật Champa từ Cleveland Museum of Art (CC0)."""
        logger.info("--- [CMA] Bắt đầu quét Cleveland Museum of Art ---")
        candidates = []
        queries = ["champa", "tra-kieu", "my-son", "dong-duong"]
        seen_cma_ids = set()
        
        for q in queries:
            try:
                data = self.fetch_json("https://openaccess-api.clevelandart.org/api/artworks/", params={'q': q, 'has_image': '1', 'limit': 30})
                items = data.get('data', [])
                for it in items:
                    cma_id = str(it.get('id'))
                    if cma_id in seen_cma_ids:
                        continue
                    seen_cma_ids.add(cma_id)
                    
                    culture = str(it.get('culture', '')).lower()
                    title = str(it.get('title', '')).lower()
                    if not any(k in culture or k in title for k in ['cham', 'vietnam', 'dong duong', 'tra-kieu']):
                        continue
                    if any(bad in culture or bad in title for bad in ['france', 'french', 'paris', 'japan', 'china']):
                        continue
                        
                    images = it.get('images', {})
                    web_img = images.get('web', {}).get('url')
                    if web_img:
                        candidates.append({
                            'source': 'Cleveland Museum of Art',
                            'artifact_id': f"CMA_{cma_id}",
                            'image_id': f"IMG_CMA_{cma_id}",
                            'title': it.get('title', 'CMA Champa Artifact'),
                            'culture': it.get('culture'),
                            'date': it.get('creation_date'),
                            'material': it.get('technique', 'Sandstone / Bronze'),
                            'current_location': 'Cleveland Museum of Art',
                            'image_url': web_img,
                            'license': 'CC0-1.0'
                        })
            except Exception as e:
                logger.error(f"Lỗi khi quét CMA query={q}: {e}")
                
        logger.info(f"[CMA] Tìm thấy {len(candidates)} ứng viên hợp lệ.")
        return candidates

    def collect_met_candidates(self) -> List[Dict[str, Any]]:
        """Thu thập các hiện vật Champa từ The Metropolitan Museum of Art (CC0)."""
        logger.info("--- [The Met] Bắt đầu quét The Met Open Access ---")
        candidates = []
        try:
            data = self.fetch_json("https://collectionapi.metmuseum.org/public/collection/v1.1/search", params={'q': 'champa', 'hasImages': 'true'})
            object_ids = data.get('objectIDs', []) or []
            logger.info(f"[The Met] Tổng số objectIDs trả về: {len(object_ids)}")
            
            for obj_id in object_ids:
                try:
                    obj = self.fetch_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}")
                    if not obj or not obj.get('isPublicDomain'):
                        continue
                    culture = str(obj.get('culture', '')).lower()
                    title = str(obj.get('title', '')).lower()
                    classification = str(obj.get('classification', '')).lower()
                    
                    # Bộ lọc khắt khe loại bỏ tranh Pháp (Champs Elysées, Le Retour des Champs...)
                    if any(bad in culture or bad in title or bad in classification for bad in ['textiles', 'prints', 'drawings', 'french', 'france', 'paris', 'german']):
                        continue
                    if not any(good in culture or good in title for good in ['cham', 'vietnam']):
                        continue
                        
                    primary_img = obj.get('primaryImage')
                    if primary_img:
                        candidates.append({
                            'source': 'The Metropolitan Museum of Art',
                            'artifact_id': f"MET_{obj_id}",
                            'image_id': f"IMG_MET_{obj_id}",
                            'title': obj.get('title', 'The Met Champa Sculpture'),
                            'culture': obj.get('culture'),
                            'date': obj.get('objectDate'),
                            'material': obj.get('medium', 'Sandstone'),
                            'current_location': 'The Metropolitan Museum of Art',
                            'image_url': primary_img,
                            'license': 'CC0-1.0'
                        })
                    # Ảnh các góc phụ nếu có
                    for add_idx, add_url in enumerate(obj.get('additionalImages', [])):
                        candidates.append({
                            'source': 'The Metropolitan Museum of Art',
                            'artifact_id': f"MET_{obj_id}",
                            'image_id': f"IMG_MET_{obj_id}_v{add_idx+1}",
                            'title': f"{obj.get('title')} (View {add_idx+2})",
                            'culture': obj.get('culture'),
                            'date': obj.get('objectDate'),
                            'material': obj.get('medium', 'Sandstone'),
                            'current_location': 'The Metropolitan Museum of Art',
                            'image_url': add_url,
                            'license': 'CC0-1.0'
                        })
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Lỗi khi quét The Met: {e}")
            
        logger.info(f"[The Met] Tìm thấy {len(candidates)} ứng viên hợp lệ.")
        return candidates

    def collect_commons_candidates(self) -> List[Dict[str, Any]]:
        """Thu thập các hiện vật Champa từ các danh mục chuyên biệt của Wikimedia Commons."""
        logger.info("--- [Wikimedia Commons] Bắt đầu quét các Category ---")
        categories = [
            "Category:Sculptures at the Museum of Cham Sculpture",
            "Category:Tra Kieu sculptures at the Museum of Cham Sculpture",
            "Category:Tháp Mắm sculptures at the Museum of Cham Sculpture",
            "Category:Đồng Dương sculptures at the Museum of Cham Sculpture",
            "Category:Hindu sculptures at the Museum of Cham Sculpture",
            "Category:Khương Mỹ sculptures at the Museum of Cham Sculpture",
            "Category:Mỹ Sơn sculptures at the Museum of Cham Sculpture",
            "Category:Phong Lệ sculptures at the Museum of Cham Sculpture",
            "Category:Chiên Đàn sculptures at the Museum of Cham Sculpture",
            "Category:Chanh Lo sculptures at the Museum of Cham Sculpture",
            "Category:Tra Kieu altar with Apsaras",
            "Category:Cham statues of dancers",
            "Category:Devi Goddess (Huong Que statue)",
            "Category:Cham sculptures in the Ho Chi Minh City Museum of History",
            "Category:Collection of Cham sculptures in the National Museum of Vietnamese History",
            "Category:Cham sculptures in the Vietnam National Museum of Fine Arts",
            "Category:Collection of Cham sculptures in the Musée Guimet",
            "Category:Art of Vietnam in the Art Institute of Chicago",
            "Category:Cham art in the British Museum"
        ]
        
        candidates = []
        seen_page_ids = set()
        
        for cat in categories:
            try:
                data = self.fetch_json("https://commons.wikimedia.org/w/api.php", params={
                    'action': 'query',
                    'format': 'json',
                    'generator': 'categorymembers',
                    'gcmtitle': cat,
                    'gcmtype': 'file',
                    'gcmlimit': 50,
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|mime|extmetadata'
                })
                pages = data.get('query', {}).get('pages', {})
                for pid, p in pages.items():
                    if pid in seen_page_ids:
                        continue
                    seen_page_ids.add(pid)
                    
                    img_info = (p.get('imageinfo') or [{}])[0]
                    img_url = img_info.get('url')
                    if not img_url:
                        continue
                        
                    mime = img_info.get('mime', '')
                    if 'image' not in mime or 'svg' in mime or 'gif' in mime:
                        continue
                        
                    ext = img_info.get('extmetadata', {})
                    lic = ext.get('LicenseShortName', {}).get('value', 'CC-BY-SA')
                    desc = ext.get('ImageDescription', {}).get('value', '')
                    title = p.get('title', '').replace('File:', '').strip()
                    
                    # Bỏ qua nếu đã có sẵn trong danh sách image_ids
                    img_id = f"IMG_{pid}"
                    if img_id in self.existing_image_ids:
                        continue
                        
                    candidates.append({
                        'source': 'Wikimedia Commons',
                        'artifact_id': f"COMMONS_{pid}",
                        'image_id': img_id,
                        'title': title,
                        'culture': 'Champa',
                        'date': ext.get('DateTimeOriginal', {}).get('value', 'Thời kỳ Champa cổ'),
                        'material': 'Đá sa thạch (Sandstone) / Kim loại',
                        'current_location': cat.replace('Category:', ''),
                        'image_url': img_url,
                        'license': lic,
                        'description': desc
                    })
            except Exception as e:
                logger.error(f"Lỗi khi quét Commons cat={cat}: {e}")
                
        logger.info(f"[Wikimedia Commons] Tìm thấy {len(candidates)} ứng viên hợp lệ.")
        return candidates

    def run_pipeline(self):
        """Thực thi toàn bộ luồng tải, kiểm tra mã băm, khử trùng lặp và ghi dữ liệu."""
        all_candidates = []
        all_candidates.extend(self.collect_cma_candidates())
        all_candidates.extend(self.collect_met_candidates())
        all_candidates.extend(self.collect_commons_candidates())
        
        logger.info(f"===> Tổng số ứng viên thu thập được từ mọi nguồn: {len(all_candidates)}")
        
        # Khử trùng nội bộ ứng viên theo image_url
        unique_candidates = []
        seen_cand_urls = set()
        for c in all_candidates:
            u = c['image_url'].strip().lower()
            if u not in seen_cand_urls and u not in self.existing_urls:
                seen_cand_urls.add(u)
                unique_candidates.append(c)
                
        logger.info(f"===> Số ứng viên sau khi lọc trùng URL: {len(unique_candidates)}")
        
        accepted_records = []
        skipped_duplicates = []
        skipped_small_or_corrupt = []
        
        # Bộ đệm để khử trùng nội bộ lô mới (Intra-batch)
        batch_sha256 = set()
        batch_phashes = []
        
        for idx, item in enumerate(unique_candidates):
            img_id = item['image_id']
            img_url = item['image_url']
            logger.info(f"[{idx+1}/{len(unique_candidates)}] Đang xử lý: {item['title'][:50]}...")
            
            try:
                img_bytes = self.download_image_bytes(img_url)
                if len(img_bytes) < 5000:
                    skipped_small_or_corrupt.append((img_id, "File quá nhỏ (<5KB)"))
                    continue
                    
                # Tính SHA-256
                sha256_val = hashlib.sha256(img_bytes).hexdigest()
                
                # Kiểm tra trùng SHA-256 với dataset cũ hoặc lô mới
                if sha256_val in self.existing_sha256 or sha256_val in batch_sha256:
                    skipped_duplicates.append((img_id, "Trùng tuyệt đối SHA-256"))
                    continue
                    
                # Mở bằng Pillow để kiểm tra ảnh và tính pHash
                from io import BytesIO
                with Image.open(BytesIO(img_bytes)) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    w, h = pil_img.size
                    if min(w, h) < 256:
                        skipped_small_or_corrupt.append((img_id, f"Kích thước quá bé ({w}x{h} < 256px)"))
                        continue
                        
                    phash_obj = imagehash.phash(pil_img)
                    phash_str = str(phash_obj)
                    
                # Kiểm tra trùng pHash với dataset cũ (Hamming <= 4)
                is_dup, reason = self.check_duplicate(sha256_val, phash_obj)
                if is_dup:
                    skipped_duplicates.append((img_id, reason))
                    continue
                    
                # Kiểm tra trùng pHash với lô mới
                intra_dup = False
                for b_ph in batch_phashes:
                    if (phash_obj - b_ph) <= 4:
                        intra_dup = True
                        break
                if intra_dup:
                    skipped_duplicates.append((img_id, "Trùng tri giác pHash trong nội bộ lô mới"))
                    continue
                    
                # Lưu file ảnh thực tế xuống data/images/
                ext = ".jpg"
                if ".png" in img_url.lower():
                    ext = ".png"
                elif ".webp" in img_url.lower():
                    ext = ".webp"
                    
                file_name = f"{img_id}{ext}"
                target_path = self.images_dir / file_name
                target_path.write_bytes(img_bytes)
                
                # Cập nhật thông tin bản ghi đạt chuẩn
                item['image_path'] = str(target_path.as_posix())
                item['sha256'] = sha256_val
                item['phash'] = phash_str
                item['width'] = w
                item['height'] = h
                item['name_vi_suggested'] = self.generate_vietnamese_name(item['title'])
                item['era'] = item.get('date', 'Thế kỷ VII – XIII (Vương quốc Champa cổ)')
                item['origin_site'] = self.infer_origin_site(item['title'], item['current_location'])
                item['artifact_id_raw'] = item['artifact_id']
                
                accepted_records.append(item)
                batch_sha256.add(sha256_val)
                batch_phashes.append(phash_obj)
                
                logger.info(f"   -> ĐÃ THU NHẬN HỢP LỆ: {file_name} ({w}x{h})")
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {img_id}: {e}")
                skipped_small_or_corrupt.append((img_id, str(e)))
                
        logger.info("\n" + "="*50)
        logger.info("KẾT QUẢ THỰC HIỆN BỔ SUNG DỮ LIỆU:")
        logger.info(f" • Ứng viên ban đầu: {len(unique_candidates)}")
        logger.info(f" • Đã thu nhận thành công: {len(accepted_records)}")
        logger.info(f" • Đã loại bỏ do trùng lặp (SHA-256 / pHash): {len(skipped_duplicates)}")
        logger.info(f" • Bị loại do lỗi hoặc kích thước < 256px: {len(skipped_small_or_corrupt)}")
        logger.info("="*50)
        
        if accepted_records:
            self.save_and_merge(accepted_records)
            
    def generate_vietnamese_name(self, title: str) -> str:
        """Tự động chuẩn hóa tên tiếng Việt cho hiện vật."""
        t_low = title.lower()
        if "tara" in t_low:
            return "Tượng Bồ tát Tara Đồng Dương"
        elif "ganesha" in t_low:
            return "Tượng thần Ganesha Champa"
        elif "gajasimha" in t_low:
            return "Tượng thần Voi - Sư tử Gajasimha"
        elif "shiva" in t_low or "siva" in t_low:
            if "múa" in t_low or "dancing" in t_low:
                return "Tượng thần Shiva múa (Nataraja Champa)"
            return "Tượng thần Shiva Champa"
        elif "brahma" in t_low:
            return "Phù điêu thần Brahma giáng sinh"
        elif "apsara" in t_low or "danseuse" in t_low or "dancer" in t_low:
            return "Phù điêu Vũ nữ Apsara Champa"
        elif "garuda" in t_low:
            return "Tượng chim thần Garuda Champa"
        elif "devi" in t_low or "uma" in t_low or "huong que" in t_low:
            return "Tượng Nữ thần Devi / Uma (Hương Quế)"
        elif "dvarapala" in t_low or "guardian" in t_low or "hộ pháp" in t_low:
            return "Tượng thần Hộ pháp Dvarapala Champa"
        elif "lion" in t_low or "sư tử" in t_low:
            return "Tượng Sư tử Champa (Simha)"
        elif "elephant" in t_low or "voi" in t_low:
            return "Tượng Voi Champa (Trà Kiệu)"
        elif "altar" in t_low or "đài thờ" in t_low:
            return "Đài thờ điêu khắc Champa"
        else:
            clean_title = title.replace('_', ' ').replace('.jpg', '').replace('.JPG', '').replace('.png', '')
            return f"Hiện vật Điêu khắc Champa ({clean_title})"

    def infer_origin_site(self, title: str, location: str) -> str:
        s = f"{title} {location}".lower()
        if "mỹ sơn" in s or "my son" in s:
            return "Thánh địa Mỹ Sơn, Duy Xuyên, Quảng Nam"
        elif "trà kiệu" in s or "tra kieu" in s:
            return "Kinh đô Trà Kiệu, Duy Xuyên, Quảng Nam"
        elif "đồng dương" in s or "dong duong" in s:
            return "Phật viện Đồng Dương, Thăng Bình, Quảng Nam"
        elif "tháp mẫm" in s or "thap mam" in s:
            return "Tháp Mẫm, An Nhơn, Bình Định"
        elif "khương mỹ" in s or "khuong my" in s:
            return "Tháp Khương Mỹ, Tam Kỳ, Quảng Nam"
        elif "phong lệ" in s or "phong le" in s:
            return "Phong Lệ, Cẩm Lệ, Đà Nẵng"
        elif "chiên đàn" in s:
            return "Tháp Chiên Đàn, Phú Ninh, Quảng Nam"
        elif "chánh lộ" in s or "chanh lo" in s:
            return "Chánh Lộ, Quảng Ngãi"
        elif "hương quế" in s or "huong que" in s:
            return "Hương Quế, Quế Sơn, Quảng Nam"
        else:
            return "Miền Trung Việt Nam (Vương quốc Champa cổ)"

    def save_and_merge(self, new_records: List[Dict[str, Any]]):
        """Ghi nối và chuẩn hóa metadata vào file CSV."""
        new_df = pd.DataFrame(new_records)
        
        # Đảm bảo đủ các cột như metadata_cleaned.csv
        cols = list(self.existing_df.columns)
        for c in cols:
            if c not in new_df.columns:
                new_df[c] = ""
        new_df = new_df[cols]
        
        merged_df = pd.concat([self.existing_df, new_df], ignore_index=True)
        merged_df.to_csv(self.meta_cleaned_path, index=False, encoding='utf-8')
        merged_df.to_csv(self.meta_path, index=False, encoding='utf-8')
        
        logger.info(f"Đã cập nhật metadata thành công! Tổng số ảnh mới trong dataset: {len(merged_df)} ảnh.")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    expander = DatasetExpander(project_root)
    expander.run_pipeline()

