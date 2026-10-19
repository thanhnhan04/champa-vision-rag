"""
Thử nghiệm & Đánh giá giải pháp Khắc phục Độ nhạy Góc chụp và Kích thước (Viewpoint & Scale Robustness)
So sánh:
1. Baseline: CenterCrop mặc định của ViT
2. Enhanced: Letterbox Padding (bảo toàn 100% hình thể) + Multi-Scale TTA + DINOv2 Hybrid Pooling (CLS + Patch)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# 1. Hàm tiền xử lý Letterbox Padding (giữ nguyên tỷ lệ khung hình, chống mất đầu/chân tượng)
def letterbox_pad(img: Image.Image, target_size=(224, 224), pad_color=(128, 128, 128)) -> Image.Image:
    w, h = img.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", target_size, pad_color)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded

# 2. Trích xuất đặc trưng với DINOv2
def extract_features_dino(img: Image.Image, model, proc, use_hybrid=True):
    # Padding bảo toàn tỷ lệ
    padded = letterbox_pad(img)
    inputs = proc(images=padded, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        cls_tok = out.last_hidden_state[:, 0, :]
        if use_hybrid:
            patch_toks = out.last_hidden_state[:, 1:, :].mean(dim=1)
            v = 0.6 * cls_tok + 0.4 * patch_toks
        else:
            v = cls_tok
    v = v.cpu().numpy()
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

# 3. Trích xuất đặc trưng Multi-scale TTA (Test-Time Augmentation)
def extract_robust_query_features(img: Image.Image, model, proc, model_type="dino"):
    # View 1: Full image letterboxed (tỷ lệ 1.0)
    v_full = extract_features_dino(img, model, proc, use_hybrid=True)
    
    # View 2: Center zoom crop 85% (người chụp đứng gần hoặc phóng to)
    w, h = img.size
    crop_w, crop_h = int(w * 0.85), int(h * 0.85)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    img_crop = img.crop((left, top, left + crop_w, top + crop_h))
    v_crop = extract_features_dino(img_crop, model, proc, use_hybrid=True)
    
    # View 3: Lật ngang nhẹ (đối xứng góc nhìn)
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    v_flip = extract_features_dino(img_flip, model, proc, use_hybrid=True)
    
    # Tổng hợp đặc trưng bất biến
    v_combined = 1.0 * v_full + 0.4 * v_crop + 0.2 * v_flip
    return v_combined / np.maximum(np.linalg.norm(v_combined, axis=-1, keepdims=True), 1e-12)

print("Đã nạp xong hàm kỹ thuật!")

