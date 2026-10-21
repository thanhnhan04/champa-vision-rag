"""
Script Trích xuất Đặc trưng & Cập nhật Thư viện Embeddings cho Toàn bộ 669 Ảnh Hiện vật:
- Tái sử dụng 428 vector embeddings đã tính trước.
- Trích xuất đặc trưng gia tăng cho 241 ảnh mới bổ sung.
- Cập nhật đồng bộ các mô hình: CLIP Enhanced, DINOv2 Enhanced, SigLIP Enhanced và Baseline.
- Cập nhật ids_cleaned.json và valid_cleaned_indices.npy.
"""
import sys
import json
import time
import logging
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from PIL import Image
import torch

try:
    from transformers import AutoProcessor, AutoModel, AutoImageProcessor
except (ImportError, AttributeError):
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def letterbox_pad(img: Image.Image, target_size=(224, 224), pad_color=(128, 128, 128)) -> Image.Image:
    w, h = img.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, pad_color)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded

def update_all_embeddings(project_root: Path):
    emb_dir = project_root / "embeddings"
    meta_df = pd.read_csv(project_root / "data" / "metadata_cleaned.csv")
    total_imgs = len(meta_df)
    logger.info(f"Tổng số ảnh trong dataset hiện tại: {total_imgs}")
    
    # Tạo danh sách ids_cleaned mới cho 669 ảnh
    new_ids_cleaned = []
    for _, row in meta_df.iterrows():
        new_ids_cleaned.append({
            "image_id": row["image_id"],
            "artifact_id": row["artifact_id"],
            "artifact_id_raw": row.get("artifact_id_raw", row["artifact_id"])
        })
        
    (emb_dir / "ids_cleaned.json").write_text(json.dumps(new_ids_cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(emb_dir / "valid_cleaned_indices.npy", np.arange(total_imgs))
    logger.info(f"Đã cập nhật ids_cleaned.json ({total_imgs} mục) và valid_cleaned_indices.npy.")
    
    # 241 ảnh mới từ chỉ số 428 trở đi
    new_rows = meta_df.iloc[428:].copy()
    logger.info(f"Số ảnh mới cần trích xuất đặc trưng: {len(new_rows)}")
    
    if len(new_rows) == 0:
        logger.info("Không có ảnh mới cần trích xuất.")
        return

    # Nạp ảnh mới vào bộ nhớ
    new_raw_imgs = []
    new_pad_imgs = []
    for _, row in new_rows.iterrows():
        p = Path(row["image_path"])
        if not p.exists():
            p = project_root / "data" / "images" / f"{row['image_id']}.jpg"
        if not p.exists():
            p = project_root / "data" / "images" / f"{row['image_id']}.png"
        img = Image.open(p).convert("RGB")
        new_raw_imgs.append(img)
        new_pad_imgs.append(letterbox_pad(img))
        
    bs = 16
    device = "cpu"
    
    # -------------------------------------------------------------
    # 1. CLIP: Enhanced và Baseline
    # -------------------------------------------------------------
    logger.info("Trích xuất đặc trưng CLIP (openai/clip-vit-base-patch32)...")
    proc_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_clip = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    clip_enh_new, clip_base_new = [], []
    for i in range(0, len(new_rows), bs):
        # Enhanced
        batch_pad = new_pad_imgs[i:i+bs]
        inp_enh = proc_clip(images=batch_pad, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model_clip.get_image_features(**inp_enh)
            v = feat.pooler_output if hasattr(feat, "pooler_output") and feat.pooler_output is not None else feat
            v = v.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            clip_enh_new.append(v)
            
        # Baseline
        batch_raw = new_raw_imgs[i:i+bs]
        inp_base = proc_clip(images=batch_raw, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model_clip.get_image_features(**inp_base)
            v = feat.pooler_output if hasattr(feat, "pooler_output") and feat.pooler_output is not None else feat
            v = v.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            clip_base_new.append(v)
            
    v_clip_enh_new = np.vstack(clip_enh_new).astype("float32")
    v_clip_base_new = np.vstack(clip_base_new).astype("float32")
    
    # Cập nhật emb_clip_enhanced.npy
    old_clip_enh = np.load(emb_dir / "emb_clip_enhanced.npy")[:428]
    full_clip_enh = np.vstack([old_clip_enh, v_clip_enh_new])
    np.save(emb_dir / "emb_clip_enhanced.npy", full_clip_enh)
    
    old_clip_base = np.load(emb_dir / "emb_clip.npy")[:428]
    full_clip_base = np.vstack([old_clip_base, v_clip_base_new])
    np.save(emb_dir / "emb_clip.npy", full_clip_base)
    logger.info(f"Đã lưu CLIP Embeddings: {full_clip_enh.shape}")

    # -------------------------------------------------------------
    # 2. DINOv2: Enhanced và Baseline
    # -------------------------------------------------------------
    logger.info("Trích xuất đặc trưng DINOv2 (facebook/dinov2-base)...")
    proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    
    dino_enh_new, dino_base_new = [], []
    for i in range(0, len(new_rows), bs):
        # Enhanced (Hybrid 0.6 CLS + 0.4 Patch Mean)
        batch_pad = new_pad_imgs[i:i+bs]
        inp_enh = proc_dino(images=batch_pad, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_dino(**inp_enh)
            cls_t = out.last_hidden_state[:, 0, :]
            pt_t = out.last_hidden_state[:, 1:, :].mean(dim=1)
            v = 0.6 * cls_t + 0.4 * pt_t
            v = v.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            dino_enh_new.append(v)
            
        # Baseline (CLS token gốc)
        batch_raw = new_raw_imgs[i:i+bs]
        inp_base = proc_dino(images=batch_raw, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_dino(**inp_base)
            cls_t = out.last_hidden_state[:, 0, :]
            v = cls_t.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            dino_base_new.append(v)
            
    v_dino_enh_new = np.vstack(dino_enh_new).astype("float32")
    v_dino_base_new = np.vstack(dino_base_new).astype("float32")
    
    old_dino_enh = np.load(emb_dir / "emb_dino_enhanced.npy")[:428]
    full_dino_enh = np.vstack([old_dino_enh, v_dino_enh_new])
    np.save(emb_dir / "emb_dino_enhanced.npy", full_dino_enh)
    
    old_dino_base = np.load(emb_dir / "emb_dino.npy")[:428]
    full_dino_base = np.vstack([old_dino_base, v_dino_base_new])
    np.save(emb_dir / "emb_dino.npy", full_dino_base)
    logger.info(f"Đã lưu DINOv2 Embeddings: {full_dino_enh.shape}")

    # -------------------------------------------------------------
    # 3. SigLIP: Enhanced và Baseline
    # -------------------------------------------------------------
    logger.info("Trích xuất đặc trưng SigLIP (google/siglip-base-patch16-224)...")
    proc_sig = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model_sig = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
    
    sig_enh_new, sig_base_new = [], []
    for i in range(0, len(new_rows), bs):
        # Enhanced
        batch_pad = new_pad_imgs[i:i+bs]
        inp_enh = proc_sig(images=batch_pad, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model_sig.get_image_features(**inp_enh)
            v = feat.pooler_output if hasattr(feat, "pooler_output") and feat.pooler_output is not None else feat
            v = v.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            sig_enh_new.append(v)
            
        # Baseline
        batch_raw = new_raw_imgs[i:i+bs]
        inp_base = proc_sig(images=batch_raw, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model_sig.get_image_features(**inp_base)
            v = feat.pooler_output if hasattr(feat, "pooler_output") and feat.pooler_output is not None else feat
            v = v.cpu().numpy()
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            sig_base_new.append(v)
            
    v_sig_enh_new = np.vstack(sig_enh_new).astype("float32")
    v_sig_base_new = np.vstack(sig_base_new).astype("float32")
    
    old_sig_enh = np.load(emb_dir / "emb_siglip_enhanced.npy")[:428]
    full_sig_enh = np.vstack([old_sig_enh, v_sig_enh_new])
    np.save(emb_dir / "emb_siglip_enhanced.npy", full_sig_enh)
    
    old_sig_base = np.load(emb_dir / "emb_siglip.npy")[:428]
    full_sig_base = np.vstack([old_sig_base, v_sig_base_new])
    np.save(emb_dir / "emb_siglip.npy", full_sig_base)
    logger.info(f"Đã lưu SigLIP Embeddings: {full_sig_enh.shape}")

    logger.info("=== HOÀN TẤT TOÀN DIỆN CẬP NHẬT EMBEDDINGS CHO 669 ẢNH! ===")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    update_all_embeddings(project_root)

