"""
Kiểm thử ảnh người dùng tải lên với các mô hình truy xuất (CLIP, SigLIP, DINOv2, Weighted):
1. Đọc ảnh người dùng tải lên.
2. Trích xuất embedding bằng các mô hình.
3. Tìm kiếm Top-5 hiện vật tương đồng nhất trong cơ sở dữ liệu đã chuẩn hóa (428 ảnh).
4. Phân tích chi tiết kết quả đúng/sai và nguyên nhân.
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
try:
    from transformers import AutoProcessor, AutoModel
except (ImportError, AttributeError):
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.modeling_auto import AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_image(image_path: str):
    img_path = Path(image_path)
    if not img_path.exists():
        logger.error(f"Không tìm thấy file ảnh: {img_path}")
        return

    logger.info(f"Đang mở ảnh: {img_path}")
    raw_img = Image.open(img_path).convert("RGB")
    
    # 1. Tải metadata và danh sách IDs đã làm sạch
    meta_df = pd.read_csv("data/metadata_cleaned.csv")
    ids_cleaned = json.load(open("embeddings/ids_cleaned.json", "r", encoding="utf-8"))
    valid_idx = np.load("embeddings/valid_cleaned_indices.npy")
    
    # 2. Tải embeddings thư viện (Gallery)
    emb_dir = Path("embeddings")
    gallery_clip = np.load(emb_dir / "emb_clip.npy")[valid_idx]
    gallery_siglip = np.load(emb_dir / "emb_siglip.npy")[valid_idx]
    gallery_dino = np.load(emb_dir / "emb_dino.npy")[valid_idx]
    gallery_weighted = np.hstack([np.sqrt(0.7) * gallery_clip, np.sqrt(0.3) * gallery_dino]).astype('float32')

    models_info = [
        {"name": "CLIP (ViT-B/32)", "hf_id": "openai/clip-vit-base-patch32", "type": "clip", "gallery": gallery_clip},
        {"name": "SigLIP (vit-base)", "hf_id": "google/siglip-base-patch16-224", "type": "siglip", "gallery": gallery_siglip},
        {"name": "DINOv2 (dinov2-base)", "hf_id": "facebook/dinov2-base", "type": "dino", "gallery": gallery_dino},
    ]

    query_embeddings = {}
    
    device = "cpu"
    logger.info(f"Thiết bị chạy: {device}")

    # 3. Trích xuất đặc trưng ảnh truy vấn
    for m in models_info:
        logger.info(f"Đang trích xuất vector bằng {m['name']}...")
        processor = AutoProcessor.from_pretrained(m["hf_id"])
        model = AutoModel.from_pretrained(m["hf_id"]).to(device)
        model.eval()

        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        with torch.no_grad():
            if m["type"] in ["clip", "siglip"]:
                feat = model.get_image_features(**inputs)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    vec = feat.pooler_output
                else:
                    vec = feat
            elif m["type"] == "dino":
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :]
                
        vec = vec.cpu().numpy()
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        vec = (vec / np.maximum(norm, 1e-12)).astype("float32")
        query_embeddings[m["type"]] = vec

    # Vector Weighted: [sqrt(0.7) * CLIP, sqrt(0.3) * DINO]
    query_weighted = np.hstack([np.sqrt(0.7) * query_embeddings["clip"], np.sqrt(0.3) * query_embeddings["dino"]]).astype("float32")

    test_configs = [
        {"name": "CLIP (ViT-B/32)", "query_vec": query_embeddings["clip"], "gallery": gallery_clip},
        {"name": "SigLIP (vit-base)", "query_vec": query_embeddings["siglip"], "gallery": gallery_siglip},
        {"name": "DINOv2 (dinov2-base)", "query_vec": query_embeddings["dino"], "gallery": gallery_dino},
        {"name": "Weighted (0.7 CLIP + 0.3 DINOv2)", "query_vec": query_weighted, "gallery": gallery_weighted}
    ]

    all_results = {}

    for cfg in test_configs:
        q_vec = cfg["query_vec"]
        gallery = cfg["gallery"]
        # Tính tương đồng Cosine
        sims = (q_vec @ gallery.T)[0]
        top_k_indices = np.argsort(-sims)[:5]
        
        top_matches = []
        for rank, idx in enumerate(top_k_indices):
            cand_item = ids_cleaned[idx]
            cand_img_id = cand_item["image_id"]
            cand_art_id = cand_item["artifact_id"]
            sim_score = float(sims[idx])
            
            meta_row = meta_df[meta_df["image_id"] == cand_img_id]
            title = meta_row.iloc[0]["title"] if len(meta_row) > 0 else "Unknown"
            era = meta_row.iloc[0]["era"] if len(meta_row) > 0 else "Unknown"
            mat = meta_row.iloc[0]["material"] if len(meta_row) > 0 else "Unknown"
            loc = meta_row.iloc[0]["current_location"] if len(meta_row) > 0 else "Unknown"
            
            top_matches.append({
                "rank": rank + 1,
                "similarity": sim_score,
                "image_id": cand_img_id,
                "artifact_id": cand_art_id,
                "title": title,
                "era": era,
                "material": mat,
                "location": loc
            })
            
        all_results[cfg["name"]] = top_matches

    # 4. Xuất kết quả
    print("\n" + "="*80)
    print("KẾT QUẢ TRUY XUẤT CHO ẢNH TẢI LÊN:")
    print("="*80)
    for model_name, matches in all_results.items():
        print(f"\n--- Mô hình: {model_name} ---")
        for m in matches:
            print(f" Top {m['rank']} (Sim: {m['similarity']:.4f}) | {m['image_id']} [{m['artifact_id']}]")
            print(f"        Tên: {m['title'][:70]}")
            print(f"        Niên đại: {m['era']} | Chất liệu: {m['material']} | Nơi giữ: {m['location']}")
            
    # Lưu file JSON
    out_json = Path("results/user_image_test_results.json")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Đã lưu kết quả chi tiết vào {out_json}")

if __name__ == "__main__":
    img_target = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/PC/.gemini/antigravity/brain/e52126b9-18a3-4dde-9fb3-051d5d11b08d/.user_uploaded/media_1792054543586_a292cf25.png"
    test_single_image(img_target)

