"""
Phân tích hiện tượng phân mảnh định danh (Identity Fragmentation & Cross-source Duplicates)
giữa Wikimedia Commons, The Met, Cleveland Museum of Art và Wikidata.
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_cross_source_duplicates(threshold: float = 0.88):
    emb_file = Path("embeddings/emb_hybrid_dino_siglip.npy")
    ids_file = Path("embeddings/ids.json")
    meta_file = Path("data/metadata.csv")
    
    if not emb_file.exists() or not ids_file.exists():
        logger.error("Thiếu embeddings hoặc ids.json")
        return
        
    E = np.load(emb_file)
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    meta_df = pd.read_csv(meta_file).set_index("image_id")
    
    S = E @ E.T
    n = len(ids)
    
    duplicate_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            art_i = ids[i]["artifact_id"]
            art_j = ids[j]["artifact_id"]
            img_i = ids[i]["image_id"]
            img_j = ids[j]["image_id"]
            
            if art_i != art_j and S[i, j] >= threshold:
                title_i = meta_df.loc[img_i, "title"] if img_i in meta_df.index else ""
                title_j = meta_df.loc[img_j, "title"] if img_j in meta_df.index else ""
                duplicate_pairs.append({
                    "similarity": float(S[i, j]),
                    "image_id_A": img_i,
                    "artifact_id_A": art_i,
                    "title_A": str(title_i)[:60],
                    "image_id_B": img_j,
                    "artifact_id_B": art_j,
                    "title_B": str(title_j)[:60]
                })
                
    duplicate_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "cross_source_duplicates.csv"
    pd.DataFrame(duplicate_pairs).to_csv(out_csv, index=False, encoding="utf-8")
    
    logger.info(f"Phát hiện {len(duplicate_pairs)} cặp ảnh trùng lặp thực tế giữa các artifact_id khác nhau (ngưỡng >= {threshold}).")
    logger.info(f"Đã lưu danh sách vào {out_csv}")
    return duplicate_pairs

if __name__ == "__main__":
    analyze_cross_source_duplicates()

