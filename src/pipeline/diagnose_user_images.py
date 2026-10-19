import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
try:
    from transformers import AutoProcessor, AutoModel
except (ImportError, AttributeError):
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.modeling_auto import AutoModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

p1 = r'C:\Users\PC\.gemini\antigravity\brain\e52126b9-18a3-4dde-9fb3-051d5d11b08d\.user_uploaded\media_1792055235517_b03c5adf.png'
p2 = r'C:\Users\PC\.gemini\antigravity\brain\e52126b9-18a3-4dde-9fb3-051d5d11b08d\.user_uploaded\media_1792054543586_a292cf25.png'

ids = json.load(open('embeddings/ids_cleaned.json', encoding='utf-8'))
valid_idx = np.load('embeddings/valid_cleaned_indices.npy')
gallery_dino = np.load('embeddings/emb_dino.npy')[valid_idx]
gallery_clip = np.load('embeddings/emb_clip.npy')[valid_idx]
gallery_weighted = np.hstack([np.sqrt(0.7)*gallery_clip, np.sqrt(0.3)*gallery_dino])
gallery_weighted = gallery_weighted / np.linalg.norm(gallery_weighted, axis=-1, keepdims=True)

df = pd.read_csv('data/metadata_cleaned.csv')
img_map = dict(zip(df['image_id'], df['name_vi_suggested']))

print("Đang tải các mô hình...")
proc_dino = AutoProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').eval()

proc_clip = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
model_clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32').eval()

for idx, p in enumerate([p1, p2]):
    img = Image.open(p).convert('RGB')
    print(f"\n=======================================================")
    print(f"=== USER IMAGE {idx+1} ({img.size}) ===")
    print(f"=======================================================")
    
    # DINO
    inp_d = proc_dino(images=img, return_tensors='pt')
    with torch.no_grad():
        v_d = model_dino(**inp_d).last_hidden_state[:, 0, :].numpy()
        v_d = v_d / np.linalg.norm(v_d)
    
    # CLIP
    inp_c = proc_clip(images=img, return_tensors='pt')
    with torch.no_grad():
        feat = model_clip.get_image_features(**inp_c)
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            v_c = feat.pooler_output
        else:
            v_c = feat
        v_c = v_c.cpu().numpy()
        v_c = v_c / np.linalg.norm(v_c)
        
    v_w = np.hstack([np.sqrt(0.7)*v_c, np.sqrt(0.3)*v_d])
    v_w = v_w / np.linalg.norm(v_w)
    
    for m_name, v, gal in [('DINOv2', v_d, gallery_dino), ('CLIP', v_c, gallery_clip), ('Weighted', v_w, gallery_weighted)]:
        sims = (v @ gal.T)[0]
        top3 = np.argsort(-sims)[:3]
        print(f"\n-- {m_name} Top 3 Matches --")
        for rank, k in enumerate(top3):
            item = ids[k]
            img_id = item['image_id']
            art_id = item['artifact_id']
            name = img_map.get(img_id, 'Chưa xác định')
            print(f"  {rank+1}. Sim={sims[k]:.4f} | ID={img_id} | Art={art_id} | Tên: {name}")
