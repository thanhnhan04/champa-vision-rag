"""
Tạo các biến thể Ensemble bổ sung:
- CLIP + DINOv2 (Concatenation: 512 + 768 = 1280 chiều)
- CLIP + SigLIP (Concatenation: 512 + 768 = 1280 chiều)
- Weighted Score Combination (sẽ tính trực tiếp trong quá trình đánh giá)
"""
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_ensembles():
    emb_dir = Path("embeddings")
    E_clip = np.load(emb_dir / "emb_clip.npy")
    E_dino = np.load(emb_dir / "emb_dino.npy")
    E_siglip = np.load(emb_dir / "emb_siglip.npy")
    
    # 1. Ensemble CLIP + DINOv2
    E_clip_dino = np.hstack([E_clip, E_dino])
    norm = np.linalg.norm(E_clip_dino, axis=-1, keepdims=True)
    E_clip_dino = (E_clip_dino / np.maximum(norm, 1e-12)).astype('float32')
    np.save(emb_dir / "emb_ensemble_clip_dino.npy", E_clip_dino)
    logger.info(f"Đã lưu emb_ensemble_clip_dino.npy: {E_clip_dino.shape}")
    
    # 2. Ensemble CLIP + SigLIP
    E_clip_siglip = np.hstack([E_clip, E_siglip])
    norm = np.linalg.norm(E_clip_siglip, axis=-1, keepdims=True)
    E_clip_siglip = (E_clip_siglip / np.maximum(norm, 1e-12)).astype('float32')
    np.save(emb_dir / "emb_ensemble_clip_siglip.npy", E_clip_siglip)
    logger.info(f"Đã lưu emb_ensemble_clip_siglip.npy: {E_clip_siglip.shape}")

if __name__ == "__main__":
    generate_ensembles()

