import os
import json
import logging
import hashlib
from pathlib import Path
import httpx
from PIL import Image
import imagehash
from tenacity import retry, wait_exponential, stop_after_attempt
import time

logger = logging.getLogger(__name__)

class ImageDownloader:
    def __init__(self, config):
        self.delay = config.get('global', {}).get('default_delay', 1.5)
        self.user_agent = config.get('global', {}).get('user_agent', 'ChampaBot/0.1')
        self.images_dir = Path("data/images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(headers={'User-Agent': self.user_agent}, timeout=60.0, follow_redirects=True)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _download_file(self, url: str, target_path: Path):
        time.sleep(self.delay)
        with self.client.stream("GET", url) as response:
            if response.status_code in (403, 429):
                raise Exception(f"Rate limited or forbidden: {response.status_code}")
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    def process_image(self, record: dict) -> dict:
        image_url = record.get('image_url')
        image_id = record.get('image_id')
        
        if not image_url or not image_id:
            return record

        # Lấy đuôi file từ mime hoặc URL
        ext = ".jpg"
        mime = record.get('mime') or ''
        
        # Skip video/audio
        if 'video' in mime or 'audio' in mime or image_url.lower().endswith(('.webm', '.mp4', '.ogv', '.ogg')):
            logger.info(f"Bỏ qua định dạng không phải ảnh: {image_url}")
            return record
            
        if 'png' in mime:
            ext = ".png"
        elif 'webp' in mime:
            ext = ".webp"
            
        file_name = f"{image_id}{ext}"
        target_path = self.images_dir / file_name

        # Bỏ qua nếu đã tải và file hợp lệ
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info(f"Đã tồn tại ảnh: {file_name}")
        else:
            logger.info(f"Đang tải {file_name} từ {image_url}")
            try:
                self._download_file(image_url, target_path)
            except Exception as e:
                logger.error(f"Lỗi tải {image_url}: {e}")
                return record

        # Kiểm tra bằng Pillow và tính mã băm
        try:
            with Image.open(target_path) as img:
                # Kiểm tra kích thước tối thiểu (ví dụ 256px cạnh ngắn)
                min_edge = min(img.width, img.height)
                if min_edge < 256:
                    logger.warning(f"Ảnh {file_name} quá nhỏ ({img.width}x{img.height}). Đánh dấu lỗi.")
                    target_path.unlink() # Xóa file vì không đạt chuẩn
                    return record
                    
                # Tính perceptual hash (phash)
                phash_val = str(imagehash.phash(img))
                
            # Tính sha256 cho bản thân nội dung file
            hasher = hashlib.sha256()
            with open(target_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            sha256_val = hasher.hexdigest()
            
            # Cập nhật record
            record['file_path'] = str(target_path.as_posix())
            record['sha256'] = sha256_val
            record['phash'] = phash_val
            record['width'] = img.width
            record['height'] = img.height
            
        except Exception as e:
            logger.error(f"File ảnh {file_name} bị hỏng hoặc không hợp lệ: {e}")
            if target_path.exists():
                try:
                    time.sleep(0.5) # Wait for file lock to release
                    target_path.unlink()
                except Exception as del_e:
                    logger.warning(f"Không thể xóa {file_name}: {del_e}. Sẽ bỏ qua.")
                
        return record

def run_download_pipeline(config):
    images_file = Path("data/processed/images.jsonl")
    if not images_file.exists():
        logger.error("Không tìm thấy tệp images.jsonl")
        return

    downloader = ImageDownloader(config)
    updated_records = []
    
    with open(images_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # Chạy hàm tải và cập nhật
            updated_record = downloader.process_image(record)
            updated_records.append(updated_record)
            
    # Ghi đè lại với thông tin mới
    with open(images_file, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logger.info("Hoàn tất tải ảnh và cập nhật metadata!")

