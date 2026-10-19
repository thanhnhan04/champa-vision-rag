"""
Module thu thập dữ liệu từ Wikimedia Commons thông qua MediaWiki API.
Nguồn này được chọn vì API ổn định, giấy phép mở rõ ràng (CC, PD) và dữ liệu phong phú.
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

API_URL = "https://commons.wikimedia.org/w/api.php"

class RateLimitError(Exception):
    pass

class CommonsClient:
    def __init__(self, user_agent: str, delay: float = 1.5):
        self.headers = {'User-Agent': user_agent}
        self.delay = delay
        self.client = httpx.Client(headers=self.headers, timeout=30.0)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError))
    )
    def fetch_page(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gọi API với cơ chế thử lại lũy tiến và tôn trọng giới hạn tốc độ."""
        time.sleep(self.delay)  # Tôn trọng rate limit
        response = self.client.get(API_URL, params=params)
        
        if response.status_code in (429, 403):
            logger.warning(f"Bị giới hạn tốc độ (HTTP {response.status_code}). Đang thử lại...")
            raise RateLimitError(f"HTTP {response.status_code}")
            
        response.raise_for_status()
        response.encoding = 'utf-8' # Bắt buộc dùng utf-8
        return response.json()

    def iter_category_files(self, category: str) -> Iterator[Dict[str, Any]]:
        """Duyệt qua tất cả tệp trong một category cụ thể."""
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'categorymembers',
            'gcmtitle': category,
            'gcmtype': 'file',
            'gcmlimit': 50,
            'prop': 'imageinfo',
            'iiprop': 'url|size|mime|extmetadata',
        }
        
        while True:
            data = self.fetch_page(params)
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'imageinfo' in page_data:
                    yield page_data
                    
            if 'continue' not in data:
                break
            params.update(data['continue'])

def parse_commons_file(page_data: Dict[str, Any], allowed_licenses: List[str]) -> tuple[Artifact | None, ImageInfo | None]:
    """Phân tích dữ liệu JSON trả về từ API thành Pydantic models."""
    title = page_data.get('title', '')
    imageinfo = page_data['imageinfo'][0]
    ext = imageinfo.get('extmetadata', {})
    
    # Kiểm tra giấy phép
    license_short = ext.get('LicenseShortName', {}).get('value', '').lower()
    license_clean = license_short.replace(' ', '-')
    
    # Chấp nhận một số định dạng phổ biến của giấy phép
    is_allowed = False
    for allowed in allowed_licenses:
        if allowed.lower() in license_clean or license_clean in allowed.lower():
            is_allowed = True
            break
            
    if not is_allowed:
        logger.info(f"Bỏ qua {title} do giấy phép không hợp lệ: {license_short}")
        return None, None

    # Lấy thông tin
    artifact_id = f"COMMONS_{page_data.get('pageid')}"
    image_id = f"IMG_{page_data.get('pageid')}"
    
    description = ext.get('ImageDescription', {}).get('value')
    if description:
        import re
        description = re.sub(r'<[^>]+>', '', description).strip()
        
    author = ext.get('Artist', {}).get('value')
    if author:
        # Làm sạch thẻ HTML cơ bản nếu có
        import re
        author = re.sub(r'<[^>]+>', '', author).strip()

    retrieved_at = datetime.now()

    artifact = Artifact(
        artifact_id=artifact_id,
        name_en=title.replace('File:', '').replace('.jpg', '').replace('.png', '').strip(),
        description=description,
        source_url=imageinfo.get('descriptionurl'),
        source_name="Wikimedia Commons",
        license=license_short,
        retrieved_at=retrieved_at,
        current_location=None # Hardcoding removed because we scrape multiple museums
    )

    image = ImageInfo(
        image_id=image_id,
        artifact_id=artifact_id,
        file_path="", # Sẽ được cập nhật sau khi tải xong ảnh
        image_url=imageinfo.get('url'),
        author=author,
        license=license_short,
        width=imageinfo.get('width'),
        height=imageinfo.get('height'),
        mime=imageinfo.get('mime'),
        retrieved_at=retrieved_at
    )
    
    return artifact, image

