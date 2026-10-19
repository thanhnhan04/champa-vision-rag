"""
Tạo bộ Embeddings Tăng Cường (Letterbox Padding) cho toàn bộ 428 ảnh thư viện:
- CLIP Enhanced (bảo toàn tỷ lệ 100%, chống CenterCrop)
- SigLIP Enhanced (bảo toàn tỷ lệ 100%)
- DINOv2 Enhanced (Letterbox + Hybrid CLS + Patch Mean)
- Weighted Enhanced (0.5 DINOv2 + 0.5 CLIP)
"""
import sys
import json
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from PIL import Image
import torch
try:
    from transformers import AutoProcessor, AutoModel
except (ImportError, AttributeError):
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.modeling_auto import AutoModel

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

def main():
    print("=== TẠO TOÀN BỘ EMBEDDINGS TĂNG CƯỜNG (ENHANCED GALLERY) ===")
    meta_df = pd.read_csv("data/metadata_cleaned.csv")
    ids = json.load(open("embeddings/ids_cleaned.json", "r", encoding="utf-8"))
    img_id_to_path = dict(zip(meta_df["image_id"], meta_df["image_path"]))
    
    # 1. Chuẩn bị ảnh letterboxed
    print("Đang nạp và tiền xử lý Letterbox cho 428 ảnh...")
    letterboxed_imgs = []
    for item in ids:
        img_id = item["image_id"]
        p = Path(img_id_to_path.get(img_id, f"data/images/{img_id}.jpg"))
        if not p.exists():
            p = Path("data/images") / f"{img_id}.jpg"
        img = Image.open(p).convert("RGB")
        letterboxed_imgs.append(letterbox_pad(img))
        
    emb_dir = Path("embeddings")
    bs = 16
    
    # CLIP Enhanced
    clip_file = emb_dir / "emb_clip_enhanced.npy"
    if not clip_file.exists():
        print("Trích xuất CLIP Enhanced...")
        proc_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model_clip = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        clip_list = []
        for i in range(0, len(letterboxed_imgs), bs):
            batch = letterboxed_imgs[i:i+bs]
            inp = proc_clip(images=batch, return_tensors="pt")
            with torch.no_grad():
                feat = model_clip.get_image_features(**inp)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    v = feat.pooler_output
                else:
                    v = feat
                v = v.cpu().numpy()
                v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                clip_list.append(v)
        E_clip = np.vstack(clip_list).astype("float32")
        np.save(clip_file, E_clip)
        print("Đã lưu emb_clip_enhanced.npy:", E_clip.shape)
    else:
        print("Đã có emb_clip_enhanced.npy.")

    # SigLIP Enhanced
    siglip_file = emb_dir / "emb_siglip_enhanced.npy"
    if not siglip_file.exists():
        print("Trích xuất SigLIP Enhanced...")
        proc_sig = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model_sig = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
        sig_list = []
        for i in range(0, len(letterboxed_imgs), bs):
            batch = letterboxed_imgs[i:i+bs]
            inp = proc_sig(images=batch, return_tensors="pt")
            with torch.no_grad():
                feat = model_sig.get_image_features(**inp)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    v = feat.pooler_output
                else:
                    v = feat
                v = v.cpu().numpy()
                v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                sig_list.append(v)
        E_sig = np.vstack(sig_list).astype("float32")
        np.save(siglip_file, E_sig)
        print("Đã lưu emb_siglip_enhanced.npy:", E_sig.shape)
    else:
        print("Đã có emb_siglip_enhanced.npy.")

    print("Hoàn tất tạo Gallery Enhanced!")

if __name__ == "__main__":
    main()

