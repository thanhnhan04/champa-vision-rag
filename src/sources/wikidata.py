"""
Module thu thập dữ liệu từ Wikidata qua SPARQL Endpoint.
Nguồn này cung cấp metadata có cấu trúc rất cao nhưng số lượng có thể hạn chế (chủ yếu là các bảo vật quốc gia hoặc hiện vật nổi tiếng).
"""
import time
import json
import logging
import urllib.parse
from typing import Iterator, Dict, Any, List
from datetime import datetime
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.models import Artifact, ImageInfo

logger = logging.getLogger(__name__)

SPARQL_URL = "https://query.wikidata.org/sparql"

class RateLimitError(Exception):
    pass

class WikidataClient:
    def __init__(self, user_agent: str, delay: float = 2.0):
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'application/sparql-results+json'
        }
        self.delay = delay
        self.client = httpx.Client(headers=self.headers, timeout=30.0)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError))
    )
    def fetch_sparql(self, query: str) -> Dict[str, Any]:
        time.sleep(self.delay)
        response = self.client.get(SPARQL_URL, params={'query': query})
        
        if response.status_code in (429, 403):
            logger.warning(f"Bị giới hạn tốc độ (HTTP {response.status_code}). Đang thử lại...")
            raise RateLimitError(f"HTTP {response.status_code}")
            
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.json()

    def search_champa(self) -> Iterator[Dict[str, Any]]:
        # Truy vấn các hiện vật thuộc văn hóa Champa hoặc nằm trong Bảo tàng Điêu khắc Chăm
        query = """
        SELECT ?item ?itemLabel ?image ?materialLabel ?collectionLabel ?date WHERE {
          { ?item wdt:P2596 wd:Q216788. } # Văn hóa: Champa
          UNION
          { ?item wdt:P195 wd:Q6940790. } # Bộ sưu tập: Museum of Cham Sculpture
          
          OPTIONAL { ?item wdt:P18 ?image. }
          OPTIONAL { ?item wdt:P186 ?material. }
          OPTIONAL { ?item wdt:P195 ?collection. }
          OPTIONAL { ?item wdt:P571 ?date. }
          
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """
        data = self.fetch_sparql(query)
        bindings = data.get('results', {}).get('bindings', [])
        
        for item in bindings:
            yield item

def parse_wikidata_item(item_data: Dict[str, Any]) -> tuple[Artifact | None, List[ImageInfo]]:
    """Phân tích dữ liệu JSON trả về từ Wikidata thành Pydantic models."""
    
    item_uri = item_data.get('item', {}).get('value', '')
    item_id = item_uri.split('/')[-1]
    artifact_id = f"WIKIDATA_{item_id}"
    
    # Lấy thông tin cơ bản
    name_en = item_data.get('itemLabel', {}).get('value')
    if not name_en or name_en.startswith("Q"): # Bỏ qua nếu không có nhãn tiếng Anh
        logger.info(f"Bỏ qua {item_id} vì không có tên")
        return None, []
        
    retrieved_at = datetime.now()
    
    # Phân tích niên đại (nếu có)
    period = None
    date_val = item_data.get('date', {}).get('value')
    if date_val:
        period = date_val.split('T')[0] # Lấy năm/tháng/ngày cơ bản

    artifact = Artifact(
        artifact_id=artifact_id,
        name_en=name_en,
        period=period,
        material=item_data.get('materialLabel', {}).get('value'),
        current_location=item_data.get('collectionLabel', {}).get('value'),
        source_url=item_uri,
        source_name="Wikidata",
        license="cc0", # Metadata trên Wikidata luôn là CC0
        retrieved_at=retrieved_at
    )

    images = []
    
    image_url = item_data.get('image', {}).get('value')
    if image_url:
        # Xử lý URL ảnh từ Wiki (cần decode %20 và tương tự)
        image_url = urllib.parse.unquote(image_url)
        images.append(ImageInfo(
            image_id=f"IMG_{item_id}",
            artifact_id=artifact_id,
            file_path="",
            image_url=image_url,
            author=None,
            license="cc-by-sa", # Ảnh trên Wiki Commons thường là CC-BY-SA, sẽ lấy mặc định
            view_type="main",
            retrieved_at=retrieved_at
        ))
            
    return artifact, images

