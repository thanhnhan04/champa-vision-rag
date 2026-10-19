"""
Trích xuất embeddings hình ảnh bằng CLIP, SigLIP, DINOv2 theo hướng dẫn mục 4.1.
Cố định seed và lưu thứ tự ids.json để chống lệch chỉ số.
"""
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
try:
    from transformers import AutoModel, AutoProcessor, AutoImageProcessor
except (ImportError, AttributeError):
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cố định seed
torch.manual_seed(42)
np.random.seed(42)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Sử dụng thiết bị tính toán: {dev.upper()}")

def load_model_processor(name: str, kind: str):
    logger.info(f"Đang tải mô hình: {name} (loại: {kind})...")
    if kind == 'dino':
        proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(dev).eval()
    elif kind == 'siglip':
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(dev).eval()
    else: # clip
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(dev).eval()
    return model, proc

@torch.no_grad()
def extract_embeddings_for_model(paths: list, name: str, kind: str, bs: int = 16):
    model, proc = load_model_processor(name, kind)
    out = []
    
    start_time = time.time()
    for i in range(0, len(paths), bs):
        batch_paths = paths[i:i+bs]
        imgs = [Image.open(p).convert('RGB') for p in batch_paths]
        
        if kind == 'dino':
            x = proc(images=imgs, return_tensors='pt').to(dev)
            outputs = model(**x)
            # DINOv2 CLS token
            h = outputs.last_hidden_state[:, 0]
        else: # clip, siglip
            x = proc(images=imgs, return_tensors='pt').to(dev)
            feat = model.get_image_features(**x)
            if hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
                h = feat.pooler_output
            elif isinstance(feat, torch.Tensor):
                h = feat
            else:
                h = feat[0]
            
        # Chuẩn hóa L2
        h_norm = torch.nn.functional.normalize(h, dim=-1)
        out.append(h_norm.cpu().numpy())
        
        if (i // bs) % 5 == 0:
            logger.info(f"[{kind.upper()}] Đã xử lý {min(i+bs, len(paths))}/{len(paths)} ảnh...")
            
    total_time = time.time() - start_time
    avg_latency = (total_time / len(paths)) * 1000 # ms / image
    
    E = np.vstack(out).astype('float32')
    logger.info(f"Hoàn thành {kind.upper()}: Shape {E.shape}, Tốc độ: {avg_latency:.1f} ms/ảnh.")
    return E, avg_latency

def run_extraction():
    meta_file = Path("data/metadata.csv")
    if not meta_file.exists():
        logger.error("Không tìm thấy metadata.csv")
        return
        
    df = pd.read_csv(meta_file)
    paths = df.image_path.tolist()
    
    # 1. Lưu thứ tự ID tương ứng
    emb_dir = Path("embeddings")
    emb_dir.mkdir(exist_ok=True)
    
    ids_data = [
        {
            'index': idx,
            'artifact_id': row['artifact_id'],
            'image_id': row['image_id'],
            'image_path': row['image_path']
        }
        for idx, row in df.iterrows()
    ]
    with open(emb_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids_data, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu embeddings/ids.json khóa chặt chỉ số hiện vật.")

    MODELS = {
        'clip': ('openai/clip-vit-base-patch32', 'clip'),
        'siglip': ('google/siglip-base-patch16-224', 'siglip'),
        'dino': ('facebook/dinov2-base', 'dino'),
    }
    
    stats = {}
    extracted = {}
    
    for key, (name, kind) in MODELS.items():
        E, lat = extract_embeddings_for_model(paths, name, kind, bs=16)
        extracted[key] = E
        stats[key] = {'latency_ms': lat, 'dim': E.shape[1], 'model_name': name}
        np.save(emb_dir / f"emb_{key}.npy", E)
        logger.info(f"Đã lưu embeddings/emb_{key}.npy")
        
    # Mở rộng ghép embedding: DINOv2 + SigLIP (ghép và re-normalize L2)
    logger.info("Tạo biểu diễn ghép (Hybrid Ensemble: DINOv2 + SigLIP)...")
    E_dino = extracted['dino']
    E_siglip = extracted['siglip']
    E_fused = np.hstack([E_dino, E_siglip])
    # Chuẩn hóa L2 lại
    norm = np.linalg.norm(E_fused, axis=-1, keepdims=True)
    E_fused = (E_fused / np.maximum(norm, 1e-12)).astype('float32')
    np.save(emb_dir / "emb_hybrid_dino_siglip.npy", E_fused)
    stats['hybrid_dino_siglip'] = {'latency_ms': stats['dino']['latency_ms'] + stats['siglip']['latency_ms'], 'dim': E_fused.shape[1], 'model_name': 'Fused (DINOv2 + SigLIP)'}
    logger.info(f"Đã lưu embeddings/emb_hybrid_dino_siglip.npy với chiều: {E_fused.shape}")
    
    with open(emb_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Hoàn thành trích xuất toàn bộ embeddings!")

if __name__ == "__main__":
    run_extraction()
