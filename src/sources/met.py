import time
import logging
from typing import Iterator, Dict, Any, List
from datetime import datetime
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.models import Artifact, ImageInfo

logger = logging.getLogger(__name__)

class MetClient:
    def __init__(self, user_agent: str, delay: float = 1.0):
        # API của Met chặn yêu cầu nếu không có User-Agent chuẩn
        self.headers = {'User-Agent': user_agent}
        self.delay = delay
        self.client = httpx.Client(headers=self.headers, timeout=30.0, follow_redirects=True)
        self.search_url = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
        self.object_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(httpx.RequestError)
    )
    def fetch_json(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        time.sleep(self.delay)
        response = self.client.get(url, params=params)
        
        if response.status_code in (429, 403):
            logger.warning(f"Bị giới hạn tốc độ hoặc chặn từ Met (HTTP {response.status_code}).")
            response.raise_for_status()
            
        if response.status_code == 404:
            return {}
            
        response.raise_for_status()
        return response.json()

    def iter_champa_objects(self) -> Iterator[Dict[str, Any]]:
        params = {
            'q': 'champa',
            'hasImages': 'true'
        }
        data = self.fetch_json(self.search_url, params=params)
        object_ids = data.get('objectIDs', [])
        
        if not object_ids:
            logger.warning("Không tìm thấy objectIDs nào từ The Met.")
            return
            
        logger.info(f"The Met trả về {len(object_ids)} kết quả tiềm năng.")
        
        for obj_id in object_ids:
            obj_data = self.fetch_json(self.object_url.format(obj_id))
            if obj_data:
                # Lọc sơ bộ: đảm bảo thuộc văn hóa Champa hoặc liên quan
                culture = obj_data.get('culture', '').lower()
                title = obj_data.get('title', '').lower()
                if 'cham' in culture or 'cham' in title or 'vietnam' in culture:
                    yield obj_data

def parse_met_object(obj_data: Dict[str, Any]) -> tuple[Artifact | None, List[ImageInfo]]:
    """Phân tích dữ liệu JSON của Met."""
    # Chỉ lấy dữ liệu Public Domain
    if not obj_data.get('isPublicDomain'):
        logger.info(f"Bỏ qua {obj_data.get('objectID')} do không phải Public Domain")
        return None, []
        
    obj_id = obj_data.get('objectID')
    artifact_id = f"MET_{obj_id}"
    
    retrieved_at = datetime.now()
    
    dimensions = obj_data.get('dimensions')
    culture = obj_data.get('culture')
    period = obj_data.get('period')
    
    # Kết hợp culture và period nếu có
    full_period = period
    if culture and period:
        full_period = f"{culture}, {period}"
    elif culture:
        full_period = culture
        
    artifact = Artifact(
        artifact_id=artifact_id,
        name_en=obj_data.get('title', 'Unknown Title'),
        period=full_period,
        material=obj_data.get('medium'),
        dimensions=dimensions,
        origin_site=obj_data.get('excavation'),
        current_location=obj_data.get('repository', 'The Metropolitan Museum of Art'),
        description=obj_data.get('creditLine'),
        source_url=obj_data.get('objectURL'),
        source_name="The Metropolitan Museum of Art",
        license="cc0-1.0", # Met Open Access là CC0
        retrieved_at=retrieved_at
    )
    
    images = []
    
    # Primary image
    primary_url = obj_data.get('primaryImage')
    if primary_url:
        images.append(ImageInfo(
            image_id=f"IMG_MET_{obj_id}_primary",
            artifact_id=artifact_id,
            file_path="",
            image_url=primary_url,
            license="cc0-1.0",
            retrieved_at=retrieved_at
        ))
        
    # Additional images
    for idx, img_url in enumerate(obj_data.get('additionalImages', [])):
        images.append(ImageInfo(
            image_id=f"IMG_MET_{obj_id}_{idx}",
            artifact_id=artifact_id,
            file_path="",
            image_url=img_url,
            license="cc0-1.0",
            retrieved_at=retrieved_at
        ))
        
    return artifact, images

