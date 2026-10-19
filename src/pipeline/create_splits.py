"""
Chia tập dữ liệu theo nhóm hiện vật (GroupKFold) chống rò rỉ dữ liệu (Data Leakage)
theo hướng dẫn mục 3.2.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupKFold
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_grouped_folds(n_splits: int = 5):
    meta_file = Path("data/metadata.csv")
    if not meta_file.exists():
        logger.error("Không tìm thấy data/metadata.csv")
        return
        
    df = pd.read_csv(meta_file)
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    
    for k, (tr, te) in enumerate(gkf.split(df, groups=df.artifact_id)):
        train_ids = sorted(df.iloc[tr].artifact_id.unique().tolist())
        test_ids = sorted(df.iloc[te].artifact_id.unique().tolist())
        train_imgs = len(df[df.artifact_id.isin(train_ids)])
        test_imgs = len(df[df.artifact_id.isin(test_ids)])
        
        folds.append({
            'fold': k,
            'train_artifacts_count': len(train_ids),
            'test_artifacts_count': len(test_ids),
            'train_images_count': train_imgs,
            'test_images_count': test_imgs,
            'train_ids': train_ids,
            'test_ids': test_ids
        })
        logger.info(f"Fold {k}: Train {len(train_ids)} hiện vật ({train_imgs} ảnh) | Test {len(test_ids)} hiện vật ({test_imgs} ảnh)")
        
    out_file = splits_dir / "folds.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Đã lưu các fold chia vào {out_file}")

if __name__ == "__main__":
    create_grouped_folds()

