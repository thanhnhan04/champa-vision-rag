import json
import time
import logging
from pathlib import Path
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

class WikiTextCrawler:
    def __init__(self, user_agent: str, delay: float = 2.0):
        self.headers = {'User-Agent': user_agent}
        self.delay = delay
        self.client = httpx.Client(headers=self.headers, timeout=30.0)
        self.search_url = "https://vi.wikipedia.org/w/api.php"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(httpx.RequestError)
    )
    def fetch_json(self, params: dict) -> dict:
        time.sleep(self.delay)
        response = self.client.get(self.search_url, params=params)
        
        if response.status_code in (429, 403):
            logger.warning(f"Bị giới hạn tốc độ (HTTP {response.status_code}).")
            response.raise_for_status()
            
        response.raise_for_status()
        return response.json()

    def search_article(self, keyword: str) -> str:
        """Tìm bài viết phù hợp nhất trên Wikipedia Tiếng Việt"""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': f"{keyword} Chăm Pa", # Thêm Chăm Pa để giới hạn phạm vi
            'utf8': '',
            'format': 'json'
        }
        data = self.fetch_json(params)
        results = data.get('query', {}).get('search', [])
        
        if results:
            # Lấy bài viết đầu tiên
            return results[0]['title']
        return ""

    def get_article_extract(self, title: str) -> str:
        """Lấy phần mở đầu (abstract) của bài viết"""
        params = {
            'action': 'query',
            'prop': 'extracts',
            'exintro': True, # Chỉ lấy đoạn mở đầu
            'explaintext': True, # Dạng text thuần (không có HTML)
            'titles': title,
            'format': 'json'
        }
        data = self.fetch_json(params)
        pages = data.get('query', {}).get('pages', {})
        
        for page_id, page_info in pages.items():
            if 'extract' in page_info:
                return page_info['extract'].strip()
        return ""

def run_enrichment():
    logger.info("Bắt đầu cào thêm văn bản (Thuyết minh) từ Wikipedia...")
    artifacts_file = Path("data/processed/artifacts.jsonl")
    
    if not artifacts_file.exists():
        logger.error("Không tìm thấy tệp artifacts.jsonl")
        return
        
    artifacts = []
    with open(artifacts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                artifacts.append(json.loads(line))
                
    crawler = WikiTextCrawler(user_agent="ChampaThesisBot/0.1 (contact: email@truong.edu.vn)")
    enriched_count = 0
    
    # Từ khóa thủ công cho một số hiện vật nổi tiếng nếu tên tiếng Anh khó tìm
    keyword_map = {
        "mỹ sơn e1": "Đài thờ Mỹ Sơn E1",
        "trà kiệu": "Đài thờ Trà Kiệu",
        "tara": "Tượng Bồ tát Tara Đồng Dương",
        "đồng dương": "Phật viện Đồng Dương",
        "tháp mẫm": "Tháp Mẫm"
    }

    for a in artifacts:
        desc = a.get('description', '') or ''
        
        # Bỏ qua nếu đã có mô tả chi tiết (> 150 ký tự) và không phải là text vô nghĩa
        if len(desc) > 150 and "is in the public domain" not in desc.lower():
            continue
            
        name = a.get('name_vi') or a.get('name_en') or ""
        
        # Tìm từ khóa tối ưu
        search_kw = name
        for key, val in keyword_map.items():
            if key in name.lower():
                search_kw = val
                break
                
        # Dọn dẹp tên (loại bỏ DSC, view, .jpg)
        import re
        search_kw = re.sub(r'(\.jpg|\.png|dsc\d+|view\s*\d+)', '', search_kw, flags=re.IGNORECASE).strip()
        
        if len(search_kw) < 4:
            continue
            
        logger.info(f"Đang tìm thông tin cho: {search_kw}")
        title = crawler.search_article(search_kw)
        
        if title:
            logger.info(f" -> Tìm thấy bài viết: {title}")
            extract = crawler.get_article_extract(title)
            
            if extract and len(extract) > 50:
                # Ghi thêm vào mô tả hiện tại
                new_desc = extract
                if desc:
                    new_desc = f"{desc}\n\n[Wikipedia] {extract}"
                
                a['description'] = new_desc
                a['notes'] = (a.get('notes') or '') + f" Nguồn text bổ sung: Wikipedia ({title})."
                enriched_count += 1
                logger.info(f" -> Cập nhật thành công ({len(extract)} ký tự).")
            else:
                logger.info(" -> Bài viết quá ngắn hoặc không có nội dung.")
        else:
            logger.info(" -> Không tìm thấy bài viết phù hợp.")
            
    # Lưu lại
    if enriched_count > 0:
        with open(artifacts_file, "w", encoding="utf-8") as f:
            for a in artifacts:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                
    logger.info(f"Hoàn tất. Đã làm giàu dữ liệu cho {enriched_count} hiện vật.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_enrichment()

