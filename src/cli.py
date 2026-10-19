import argparse
import yaml
import json
import logging
from pathlib import Path
from src.sources.commons import CommonsClient, parse_commons_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config/sources.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_commons(config):
    logger.info("Bắt đầu thu thập từ Wikimedia Commons...")
    global_cfg = config.get('global', {})
    commons_cfg = config.get('sources', {}).get('commons', {})
    allowed_licenses = config.get('allowed_licenses', [])
    
    client = CommonsClient(
        user_agent=global_cfg.get('user_agent', 'ChampaBot/0.1'),
        delay=commons_cfg.get('delay', 1.5)
    )
    
    categories = commons_cfg.get('categories', [])
    
    artifacts = []
    images = []
    
    # Tạo thư mục nếu chưa có
    Path("data/raw/commons").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    for category in categories:
        logger.info(f"Đang duyệt danh mục: {category}")
        for idx, page_data in enumerate(client.iter_category_files(category)):
            # Lưu raw
            raw_path = f"data/raw/commons/{page_data['pageid']}.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)
                
            artifact, image = parse_commons_file(page_data, allowed_licenses)
            if artifact and image:
                artifacts.append(artifact.model_dump(mode='json'))
                images.append(image.model_dump(mode='json'))
                logger.info(f"Đã thu thập: {artifact.name_en}")
                
    # Mở file mode 'a' để ghi nối (append)
    with open("data/processed/artifacts.jsonl", "a", encoding="utf-8") as f:
        for a in artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open("data/processed/images.jsonl", "a", encoding="utf-8") as f:
        for i in images:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành! Thu thập được {len(artifacts)} hiện vật.")

def run_cma(config):
    logger.info("Bắt đầu thu thập từ Cleveland Museum of Art...")
    from src.sources.cma import CMAClient, parse_cma_item
    
    cma_cfg = config.get('sources', {}).get('cma', {})
    client = CMAClient(delay=cma_cfg.get('delay', 1.0))
    
    artifacts = []
    images = []
    
    Path("data/raw/cma").mkdir(parents=True, exist_ok=True)
    
    for item_data in client.search_champa():
        item_id = item_data.get('id')
        raw_path = f"data/raw/cma/{item_id}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, ensure_ascii=False, indent=2)
            
        artifact, imgs = parse_cma_item(item_data)
        if artifact and imgs:
            artifacts.append(artifact.model_dump(mode='json'))
            for img in imgs:
                images.append(img.model_dump(mode='json'))
            logger.info(f"Đã thu thập: {artifact.name_en} ({len(imgs)} ảnh)")
            
    # Mở file mode 'a' để ghi nối (append) thay vì ghi đè nếu đã chạy commons
    with open("data/processed/artifacts.jsonl", "a", encoding="utf-8") as f:
        for a in artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open("data/processed/images.jsonl", "a", encoding="utf-8") as f:
        for i in images:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành CMA! Thu thập thêm {len(artifacts)} hiện vật.")

def run_wikidata(config):
    logger.info("Bắt đầu thu thập từ Wikidata...")
    from src.sources.wikidata import WikidataClient, parse_wikidata_item
    
    global_cfg = config.get('global', {})
    wiki_cfg = config.get('sources', {}).get('wikidata', {})
    client = WikidataClient(
        user_agent=global_cfg.get('user_agent', 'ChampaBot/0.1'),
        delay=wiki_cfg.get('delay', 2.0)
    )
    
    artifacts = []
    images = []
    
    Path("data/raw/wikidata").mkdir(parents=True, exist_ok=True)
    
    for item_data in client.search_champa():
        item_id = item_data.get('item', {}).get('value', '').split('/')[-1]
        raw_path = f"data/raw/wikidata/{item_id}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, ensure_ascii=False, indent=2)
            
        artifact, imgs = parse_wikidata_item(item_data)
        if artifact and imgs:
            artifacts.append(artifact.model_dump(mode='json'))
            for img in imgs:
                images.append(img.model_dump(mode='json'))
            logger.info(f"Đã thu thập: {artifact.name_en}")
            
    with open("data/processed/artifacts.jsonl", "a", encoding="utf-8") as f:
        for a in artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open("data/processed/images.jsonl", "a", encoding="utf-8") as f:
        for i in images:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành Wikidata! Thu thập thêm {len(artifacts)} hiện vật.")

def run_met(config):
    logger.info("Bắt đầu thu thập từ The Metropolitan Museum of Art...")
    from src.sources.met import MetClient, parse_met_object
    
    global_cfg = config.get('global', {})
    met_cfg = config.get('sources', {}).get('met', {})
    client = MetClient(
        user_agent=global_cfg.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'),
        delay=met_cfg.get('delay', 1.0)
    )
    
    artifacts = []
    images = []
    
    Path("data/raw/met").mkdir(parents=True, exist_ok=True)
    
    for obj_data in client.iter_champa_objects():
        obj_id = obj_data.get('objectID')
        raw_path = f"data/raw/met/{obj_id}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(obj_data, f, ensure_ascii=False, indent=2)
            
        artifact, imgs = parse_met_object(obj_data)
        if artifact and imgs:
            artifacts.append(artifact.model_dump(mode='json'))
            for img in imgs:
                images.append(img.model_dump(mode='json'))
            logger.info(f"Đã thu thập từ Met: {artifact.name_en}")
            
    with open("data/processed/artifacts.jsonl", "a", encoding="utf-8") as f:
        for a in artifacts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    with open("data/processed/images.jsonl", "a", encoding="utf-8") as f:
        for i in images:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành The Met! Thu thập thêm {len(artifacts)} hiện vật.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Champa Artifacts Crawler")
    parser.add_argument("--source", choices=["commons", "cma", "wikidata", "met", "all"], default="commons", help="Nguồn dữ liệu")
    parser.add_argument("--action", choices=["crawl", "download"], default="crawl", help="Hành động (thu thập hoặc tải ảnh)")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.action == "crawl":
        if args.source in ("commons", "all"):
            run_commons(config)
        if args.source in ("cma", "all"):
            run_cma(config)
        if args.source in ("wikidata", "all"):
            run_wikidata(config)
        if args.source in ("met", "all"):
            run_met(config)
    elif args.action == "download":
        from src.pipeline.downloader import run_download_pipeline
        run_download_pipeline(config)

