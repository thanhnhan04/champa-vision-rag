"""
Benchmark đánh giá hiệu quả giải pháp:
- Thử nghiệm trên các ảnh hiện vật đại diện dưới các biến đổi:
  + Biến đổi 1: Phóng to / Cắt cúp khung hình (Scale Shift: Crop 80%, 70%)
  + Biến đổi 2: Góc nghiêng / Xoay nhẹ (Angle Shift: Rotation 8°, -8°)
  + Biến đổi 3: Thay đổi tỷ lệ khung hình (Aspect ratio distortion)
- So sánh Top-1 Accuracy và Cosine Similarity giữa:
  1. Baseline (AutoProcessor CenterCrop + Single CLS)
  2. Robust Enhanced (Letterbox Padding + Multi-Scale TTA + DINOv2 Hybrid Pooling)
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

# 1. Letterbox Padding
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
    print("=== BẮT ĐẦU BENCHMARK ĐỘ BỀN VỮNG GÓC NHÌN & TỶ LỆ (VIEWPOINT & SCALE) ===")
    
    # Load dataset
    meta_df = pd.read_csv("data/metadata_cleaned.csv")
    ids = json.load(open("embeddings/ids_cleaned.json", "r", encoding="utf-8"))
    valid_idx = np.load("embeddings/valid_cleaned_indices.npy")
    
    # Gallery baseline
    gallery_dino_base = np.load("embeddings/emb_dino.npy")[valid_idx]
    
    # Model
    print("Đang nạp mô hình DINOv2...")
    proc = AutoProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    
    enhanced_emb_file = Path("embeddings/emb_dino_enhanced.npy")
    if enhanced_emb_file.exists():
        print("Đã tìm thấy embeddings/emb_dino_enhanced.npy sẵn có. Nạp trực tiếp...")
        gallery_dino_enhanced = np.load(enhanced_emb_file)
    else:
        print("Đang tạo Gallery Enhanced cho 428 ảnh thư viện (theo lô 16)...")
        img_id_to_path = dict(zip(meta_df["image_id"], meta_df["image_path"]))
        enhanced_gallery_list = []
        bs = 16
        for i in range(0, len(ids), bs):
            batch_items = ids[i:i+bs]
            batch_imgs = []
            for item in batch_items:
                img_id = item["image_id"]
                img_p_str = img_id_to_path.get(img_id, f"data/images/{img_id}.jpg")
                p = Path(img_p_str)
                if not p.exists():
                    p = Path("data/images") / f"{img_id}.jpg"
                img = Image.open(p).convert("RGB")
                batch_imgs.append(letterbox_pad(img))
            inputs = proc(images=batch_imgs, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
                cls_t = out.last_hidden_state[:, 0, :]
                pt_t = out.last_hidden_state[:, 1:, :].mean(dim=1)
                v = 0.6 * cls_t + 0.4 * pt_t
                v = v.cpu().numpy()
                v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                enhanced_gallery_list.append(v)
            if (i // bs) % 5 == 0:
                print(f"Đã xử lý {min(i+bs, len(ids))}/{len(ids)} ảnh...")
        gallery_dino_enhanced = np.vstack(enhanced_gallery_list).astype("float32")
        print(f"Gallery Enhanced hoàn tất. Shape: {gallery_dino_enhanced.shape}")
        np.save(enhanced_emb_file, gallery_dino_enhanced)
        print("Đã lưu embeddings/emb_dino_enhanced.npy")
    
    # Danh sách các hiện vật kiểm thử đại diện
    test_samples = [
        {"name": "Bồ tát Tara", "path": "data/demo_samples/sample_tara.png", "target_art": "WIKIDATA_Q106654903"},
        {"name": "Thần Ganesha", "path": "data/demo_samples/sample_ganesha.jpg", "target_art": "COMMONS_132522171"},
        {"name": "Phật Đồng Dương", "path": "data/demo_samples/sample_phat_dong_duong.png", "target_art": "COMMONS_95035950"},
        {"name": "Vũ nữ Trà Kiệu", "path": "data/demo_samples/sample_vu_nu_tra_kieu.jpg", "target_art": "COMMONS_40764319"},
        {"name": "Đài thờ Mỹ Sơn E1", "path": "data/demo_samples/sample_dai_tho_my_son_e1.jpg", "target_art": "WIKIDATA_Q106653085"}
    ]
    
    # Biến đổi thử thách
    perturbations = [
        ("Ảnh gốc", lambda img: img),
        ("Cắt cúp cận cảnh (Zoom 80%)", lambda img: img.crop((int(img.width*0.1), int(img.height*0.1), int(img.width*0.9), int(img.height*0.9)))),
        ("Cắt cúp góc trên (Upper 70%)", lambda img: img.crop((0, 0, img.width, int(img.height*0.7)))),
        ("Góc xoay nghiêng +10°", lambda img: img.rotate(10, expand=True, fillcolor=(128,128,128))),
        ("Góc xoay nghiêng -10°", lambda img: img.rotate(-10, expand=True, fillcolor=(128,128,128))),
        ("Khung hình ngang (Letterbox/Squash)", lambda img: img.resize((img.width, int(img.height*0.65))))
    ]
    
    results = []
    
    def get_baseline_vec(img):
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            v = out.last_hidden_state[:, 0, :].numpy()
            return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            
    def get_enhanced_vec(img):
        # 1. Full Letterbox
        pad_full = letterbox_pad(img)
        inp1 = proc(images=pad_full, return_tensors="pt")
        with torch.no_grad():
            out1 = model(**inp1)
            v1 = (0.6 * out1.last_hidden_state[:, 0, :] + 0.4 * out1.last_hidden_state[:, 1:, :].mean(dim=1)).numpy()
            v1 = v1 / np.maximum(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-12)
            
        # 2. Multi-scale zoom (85%)
        w, h = img.size
        cw, ch = int(w * 0.85), int(h * 0.85)
        pad_zoom = letterbox_pad(img.crop(((w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2)))
        inp2 = proc(images=pad_zoom, return_tensors="pt")
        with torch.no_grad():
            out2 = model(**inp2)
            v2 = (0.6 * out2.last_hidden_state[:, 0, :] + 0.4 * out2.last_hidden_state[:, 1:, :].mean(dim=1)).numpy()
            v2 = v2 / np.maximum(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-12)
            
        # 3. Aggregation
        v_comb = 1.0 * v1 + 0.4 * v2
        return v_comb / np.maximum(np.linalg.norm(v_comb, axis=-1, keepdims=True), 1e-12)

    for item in test_samples:
        p = Path(item["path"])
        if not p.exists():
            continue
        raw_img = Image.open(p).convert("RGB")
        target_art = item["target_art"]
        
        for p_name, p_func in perturbations:
            t_img = p_func(raw_img)
            
            # Baseline
            v_b = get_baseline_vec(t_img)
            sims_b = (v_b @ gallery_dino_base.T)[0]
            top1_b_idx = np.argsort(-sims_b)[0]
            top1_b_art = ids[top1_b_idx]["artifact_id"]
            top1_b_sim = float(sims_b[top1_b_idx])
            b_correct = (top1_b_art == target_art)
            
            # Enhanced
            v_e = get_enhanced_vec(t_img)
            sims_e = (v_e @ gallery_dino_enhanced.T)[0]
            top1_e_idx = np.argsort(-sims_e)[0]
            top1_e_art = ids[top1_e_idx]["artifact_id"]
            top1_e_sim = float(sims_e[top1_e_idx])
            e_correct = (top1_e_art == target_art)
            
            results.append({
                "artifact": item["name"],
                "perturbation": p_name,
                "baseline_correct": b_correct,
                "baseline_sim": top1_b_sim,
                "enhanced_correct": e_correct,
                "enhanced_sim": top1_e_sim,
                "target_art": target_art,
                "top1_b_art": top1_b_art,
                "top1_e_art": top1_e_art
            })

    res_df = pd.DataFrame(results)
    print("\n=== KẾT QUẢ SO SÁNH CHI TIẾT ===")
    print(res_df[["artifact", "perturbation", "baseline_correct", "baseline_sim", "enhanced_correct", "enhanced_sim"]].to_string())
    
    b_acc = res_df["baseline_correct"].mean() * 100
    e_acc = res_df["enhanced_correct"].mean() * 100
    b_avg_sim = res_df["baseline_sim"].mean()
    e_avg_sim = res_df["enhanced_sim"].mean()
    
    print("\n=== TỔNG KẾT ĐỘ CHÍNH XÁC ===")
    print(f"Baseline Top-1 Accuracy: {b_acc:.1f}% | Avg Sim: {b_avg_sim:.4f}")
    print(f"Enhanced Top-1 Accuracy: {e_acc:.1f}% | Avg Sim: {e_avg_sim:.4f}")
    print(f"Mức độ cải thiện độ chính xác: +{e_acc - b_acc:.1f}%")

if __name__ == "__main__":
    main()
