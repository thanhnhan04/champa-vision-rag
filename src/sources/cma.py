"""
Module thu thập dữ liệu từ Cleveland Museum of Art (CMA).
Nguồn này cung cấp API Open Access (CC0) rất chi tiết.
"""
import time
import json
import logging
from typing import Iterator, Dict, Any, List
from datetime import datetime
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.models import Artifact, ImageInfo

logger = logging.getLogger(__name__)

API_URL = "https://openaccess-api.clevelandart.org/api/artworks/"

class RateLimitError(Exception):
    pass

class CMAClient:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.client = httpx.Client(timeout=30.0)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError))
    )
    def fetch_page(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay)
        response = self.client.get(url, params=params)
        
        if response.status_code in (429, 403):
            logger.warning(f"Bị giới hạn tốc độ (HTTP {response.status_code}). Đang thử lại...")
            raise RateLimitError(f"HTTP {response.status_code}")
            
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.json()

    def search_champa(self) -> Iterator[Dict[str, Any]]:
        params = {
            'q': 'champa',
            'has_image': '1',
            'limit': 50,
            'skip': 0
        }
        
        while True:
            data = self.fetch_page(API_URL, params)
            items = data.get('data', [])
            
            if not items:
                break
                
            for item in items:
                yield item
                
            params['skip'] += len(items)

def parse_cma_item(item_data: Dict[str, Any]) -> tuple[Artifact | None, List[ImageInfo]]:
    """Phân tích dữ liệu JSON trả về từ CMA API thành Pydantic models."""
    
    # Chỉ lấy dữ liệu mở (CC0)
    if item_data.get('share_license_status') != 'CC0':
        logger.info(f"Bỏ qua {item_data.get('title')} do không phải CC0")
        return None, []

    item_id = item_data.get('id')
    artifact_id = f"CMA_{item_id}"
    
    description = item_data.get('description', '')
    if description:
        import re
        description = re.sub(r'<[^>]+>', '', description).strip()
        
    retrieved_at = datetime.now()

    artifact = Artifact(
        artifact_id=artifact_id,
        name_en=item_data.get('title'),
        period=item_data.get('creation_date'),
        material=item_data.get('technique'),
        dimensions=item_data.get('measurements'),
        origin_site=item_data.get('find_spot'),
        current_location="Cleveland Museum of Art",
        description=description,
        source_url=item_data.get('url'),
        source_name="Cleveland Museum of Art",
        license="cc0",
        retrieved_at=retrieved_at,
        notes=item_data.get('culture', [])[0] if item_data.get('culture') else None
    )

    images = []
    
    # Lấy ảnh chính
    primary_images = item_data.get('images', {})
    if primary_images and 'web' in primary_images:
        web_img = primary_images['web']
        images.append(ImageInfo(
            image_id=f"IMG_{item_id}_main",
            artifact_id=artifact_id,
            file_path="",
            image_url=web_img.get('url'),
            author=None,
            license="cc0",
            width=int(web_img.get('width', 0)) if web_img.get('width') else None,
            height=int(web_img.get('height', 0)) if web_img.get('height') else None,
            mime="image/jpeg",
            view_type="main",
            retrieved_at=retrieved_at
        ))
        
    # Lấy ảnh bổ sung (nếu có)
    alt_images = item_data.get('alternate_images', [])
    for idx, alt in enumerate(alt_images):
        if 'web' in alt:
            alt_web = alt['web']
            images.append(ImageInfo(
                image_id=f"IMG_{item_id}_alt{idx}",
                artifact_id=artifact_id,
                file_path="",
                image_url=alt_web.get('url'),
                author=None,
                license="cc0",
                width=int(alt_web.get('width', 0)) if alt_web.get('width') else None,
                height=int(alt_web.get('height', 0)) if alt_web.get('height') else None,
                mime="image/jpeg",
                view_type=f"alternate_{idx}",
                retrieved_at=retrieved_at
            ))
            
    return artifact, images

