"""
Quy trình Đánh giá Chuẩn mực & Khách quan (Rigorous & Objective Evaluation Pipeline):
1. Hỗ trợ 3 cấu hình dữ liệu:
   - Kịch bản 1 (Raw Data): Dữ liệu thô ban đầu (430 ảnh, 224 hiện vật).
   - Kịch bản 2 (Cleaned Labels): Đã làm sạch nhãn qua kiểm toán ngoại sinh (428 ảnh, 247 hiện vật, 244 queries đa góc).
   - Kịch bản 3 (Cleaned Strict Cross-View): Sử dụng tiêu chí lọc độc lập khách quan qua Perceptual Hash (pHash Hamming dist <= 10).
     Gallery cố định chung cho mọi model. Loại bỏ các query không còn ảnh đối chứng trong gallery (242 queries hợp lệ).
2. Cluster Bootstrap 95% CI: Lấy mẫu theo HIỆN VẬT (Artifact ID) n=2000 lần, seed=42.
3. Paired Cluster Bootstrap Test cho mọi cặp mô hình kèm p-value thực nghiệm trực tiếp.
4. Báo cáo đầy đủ mọi mô hình ở mọi kịch bản.
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CPU_HARDWARE_INFO = "CPU (Intel/AMD x86_64, Single-node Local)"

def hamming_dist(h1, h2):
    try:
        n1 = int(str(h1), 16)
        n2 = int(str(h2), 16)
        return bin(n1 ^ n2).count('1')
    except:
        return 64

def cluster_bootstrap_ci(query_art_ids: list, per_query_scores: list, n_boot: int = 2000, seed: int = 42):
    """Cluster Bootstrap CI: Lấy mẫu có hoàn lại theo HIỆN VẬT (artifact_id)."""
    rng = np.random.default_rng(seed)
    df_scores = pd.DataFrame({"art_id": query_art_ids, "score": per_query_scores})
    unique_arts = df_scores["art_id"].unique()
    n_arts = len(unique_arts)
    if n_arts == 0:
        return (0.0, 0.0)
    grouped = {art: group["score"].values for art, group in df_scores.groupby("art_id")}
    means = []
    for _ in range(n_boot):
        sampled_arts = rng.choice(unique_arts, size=n_arts, replace=True)
        sampled_scores = np.concatenate([grouped[a] for a in sampled_arts])
        means.append(float(sampled_scores.mean()))
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return float(ci_low), float(ci_high)

def paired_cluster_bootstrap_test(query_art_ids: list, scores_a: list, scores_b: list, n_boot: int = 2000, seed: int = 42):
    """Kiểm định ghép cặp Cluster Bootstrap so sánh 2 mô hình theo Hiện vật."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    df_diff = pd.DataFrame({"art_id": query_art_ids, "diff": diff})
    unique_arts = df_diff["art_id"].unique()
    n_arts = len(unique_arts)
    grouped = {art: group["diff"].values for art, group in df_diff.groupby("art_id")}
    means = []
    for _ in range(n_boot):
        sampled_arts = rng.choice(unique_arts, size=n_arts, replace=True)
        sampled_diffs = np.concatenate([grouped[a] for a in sampled_arts])
        means.append(float(sampled_diffs.mean()))
    means = np.asarray(means)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    p_val = 2.0 * min(np.mean(means <= 0), np.mean(means >= 0))
    p_val = min(p_val, 1.0)
    is_sig = (p_val < 0.05)
    return float(ci_low), float(ci_high), float(p_val), bool(is_sig)

def evaluate_retrieval(E_dict: dict, ids_list: list, hashes: list = None, use_strict_phash: bool = False):
    """
    Đánh giá Leave-One-Out.
    Nếu use_strict_phash = True:
      - Loại các ảnh có d_hamming(hash_q, hash_j) <= 10 khỏi gallery chung cố định.
      - Nếu query không còn ảnh ground truth nào, loại query khỏi tập đánh giá.
    """
    art_ids = np.asarray([x["artifact_id"] for x in ids_list])
    n_total = len(ids_list)
    results = {}
    ks = (1, 5, 10)
    
    # 1. Xác định trước tập query và gallery cố định chung cho MỌI mô hình
    valid_query_indices = []
    query_valid_galleries = {}
    
    for q in range(n_total):
        if not use_strict_phash:
            # Leave-One-Out thông thường: gallery là tất cả các ảnh khác q
            cands = [j for j in range(n_total) if j != q]
            rel_total = (art_ids[cands] == art_ids[q]).sum()
            if rel_total > 0:
                valid_query_indices.append(q)
                query_valid_galleries[q] = cands
        else:
            # Kịch bản Strict: loại các ảnh burst-shot d_hamming <= 10
            cands = [j for j in range(n_total) if j != q and hamming_dist(hashes[q], hashes[j]) > 10]
            rel_total = (art_ids[cands] == art_ids[q]).sum()
            # Bắt buộc: Chỉ giữ query nếu còn ít nhất 1 ảnh đúng trong gallery!
            if rel_total > 0:
                valid_query_indices.append(q)
                query_valid_galleries[q] = cands
                
    valid_query_arts = art_ids[valid_query_indices].tolist()
    
    # 2. Đánh giá từng mô hình trên chính xác tập query và gallery cố định này
    for model_name, E in E_dict.items():
        dim = E.shape[1]
        S = E @ E.T
        
        rec_art = {k: [] for k in ks}
        map_art = []
        rec_img = {k: [] for k in ks}
        map_img = []
        mrr_img = []
        
        for q in valid_query_indices:
            gallery = query_valid_galleries[q]
            # Xếp hạng gallery theo độ tương đồng giảm dần của mô hình hiện tại
            order = sorted(gallery, key=lambda j: S[q, j], reverse=True)
            
            # --- Image-level ---
            rel_i = (art_ids[order] == art_ids[q]).astype(int)
            for k in ks:
                rec_img[k].append(int(rel_i[:k].any()))
            hits = np.where(rel_i == 1)[0]
            mrr_img.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)
            prec = np.cumsum(rel_i) / (np.arange(len(rel_i)) + 1)
            actual_rel = rel_i.sum()
            map_img.append(float((prec * rel_i).sum() / actual_rel) if actual_rel > 0 else 0.0)
            
            # --- Artifact-level ---
            seen_arts = []
            rel_a = []
            for j in order:
                a_j = art_ids[j]
                if a_j not in seen_arts:
                    seen_arts.append(a_j)
                    rel_a.append(1 if a_j == art_ids[q] else 0)
            rel_a = np.asarray(rel_a)
            for k in ks:
                rec_art[k].append(int(rel_a[:k].any()))
            prec_a = np.cumsum(rel_a) / (np.arange(len(rel_a)) + 1)
            map_art.append(float((prec_a * rel_a).sum() / 1.0))
            
        ci_r1 = cluster_bootstrap_ci(valid_query_arts, rec_art[1])
        ci_map = cluster_bootstrap_ci(valid_query_arts, map_art)
        ci_img_r1 = cluster_bootstrap_ci(valid_query_arts, rec_img[1])
        ci_img_map = cluster_bootstrap_ci(valid_query_arts, map_img)
        
        results[model_name] = {
            "dim": dim,
            "n_queries": len(valid_query_indices),
            "query_arts": valid_query_arts,
            "artifact_level": {
                "R@1": float(np.mean(rec_art[1])),
                "R@1_CI": ci_r1,
                "R@5": float(np.mean(rec_art[5])),
                "R@10": float(np.mean(rec_art[10])),
                "mAP": float(np.mean(map_art)),
                "mAP_CI": ci_map,
                "scores_r1": rec_art[1],
                "scores_map": map_art
            },
            "image_level": {
                "R@1": float(np.mean(rec_img[1])),
                "R@1_CI": ci_img_r1,
                "R@5": float(np.mean(rec_img[5])),
                "R@10": float(np.mean(rec_img[10])),
                "MRR": float(np.mean(mrr_img)),
                "mAP": float(np.mean(map_img)),
                "mAP_CI": ci_img_map
            }
        }
    return results

def run_all_benchmarks():
    emb_dir = Path("embeddings")
    raw_clip = np.load(emb_dir / "emb_clip.npy")
    raw_siglip = np.load(emb_dir / "emb_siglip.npy")
    raw_dino = np.load(emb_dir / "emb_dino.npy")
    raw_hybrid = np.load(emb_dir / "emb_hybrid_dino_siglip.npy")
    raw_clip_dino = np.load(emb_dir / "emb_ensemble_clip_dino.npy")
    raw_clip_siglip = np.load(emb_dir / "emb_ensemble_clip_siglip.npy")
    raw_weighted = np.hstack([np.sqrt(0.7) * raw_clip, np.sqrt(0.3) * raw_dino]).astype('float32')
    
    E_raw_dict = {
        "CLIP": raw_clip,
        "SigLIP": raw_siglip,
        "DINOv2": raw_dino,
        "Hybrid (DINOv2 + SigLIP)": raw_hybrid,
        "Ensemble (CLIP + DINOv2)": raw_clip_dino,
        "Ensemble (CLIP + SigLIP)": raw_clip_siglip,
        "Weighted (0.7 CLIP + 0.3 DINOv2)": raw_weighted
    }
    
    ids_raw = json.load(open(emb_dir / "ids.json", "r", encoding="utf-8"))
    ids_cleaned = json.load(open(emb_dir / "ids_cleaned.json", "r", encoding="utf-8"))
    valid_idx = np.load(emb_dir / "valid_cleaned_indices.npy")
    E_cleaned_dict = {k: v[valid_idx] for k, v in E_raw_dict.items()}
    
    df_cleaned = pd.read_csv("data/metadata_cleaned.csv")
    hashes_cleaned = [df_cleaned[df_cleaned["image_id"] == x["image_id"]].iloc[0]["phash"] for x in ids_cleaned]
    
    logger.info("Chạy Kịch bản 1 (Raw Data)...")
    res_raw = evaluate_retrieval(E_raw_dict, ids_raw, use_strict_phash=False)
    
    logger.info("Chạy Kịch bản 2 (Cleaned Labels)...")
    res_cleaned = evaluate_retrieval(E_cleaned_dict, ids_cleaned, use_strict_phash=False)
    
    logger.info("Chạy Kịch bản 3 (Cleaned Strict Cross-View qua pHash độc lập)...")
    res_strict = evaluate_retrieval(E_cleaned_dict, ids_cleaned, hashes=hashes_cleaned, use_strict_phash=True)
    
    def make_table(results_dict, scenario_name):
        rows = []
        for m, r in results_dict.items():
            art = r["artifact_level"]
            img = r["image_level"]
            rows.append({
                "Kịch bản": scenario_name,
                "Model": m,
                "Dim": r["dim"],
                "Queries": r["n_queries"],
                "Artifact R@1": f"{art['R@1']*100:.1f}%",
                "Artifact R@1 [95% CI]": f"[{art['R@1_CI'][0]*100:.1f}%, {art['R@1_CI'][1]*100:.1f}%]",
                "Artifact R@5": f"{art['R@5']*100:.1f}%",
                "Artifact R@10": f"{art['R@10']*100:.1f}%",
                "Artifact mAP": f"{art['mAP']:.3f}",
                "Artifact mAP [95% CI]": f"[{art['mAP_CI'][0]:.3f}, {art['mAP_CI'][1]:.3f}]",
                "Image R@1": f"{img['R@1']*100:.1f}%",
                "Image R@5": f"{img['R@5']*100:.1f}%",
                "Image mAP": f"{img['mAP']:.3f}"
            })
        return rows
        
    all_rows = make_table(res_raw, "1. Raw") + make_table(res_cleaned, "2. Cleaned") + make_table(res_strict, "3. Cleaned Strict")
    df_compare = pd.DataFrame(all_rows)
    df_compare.to_csv("results/rigorous_benchmark_comparison.csv", index=False, encoding="utf-8")
    
    # Ma trận Paired Bootstrap cho cả Kịch bản 2 và Kịch bản 3
    def compute_paired_matrix(res_scenario, scenario_title):
        models = list(res_scenario.keys())
        q_arts = res_scenario[models[0]]["query_arts"]
        rows = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a = models[i]
                m_b = models[j]
                sc_a = res_scenario[m_a]["artifact_level"]["scores_map"]
                sc_b = res_scenario[m_b]["artifact_level"]["scores_map"]
                ci_low, ci_high, p_val, is_sig = paired_cluster_bootstrap_test(q_arts, sc_a, sc_b)
                rows.append({
                    "Kịch bản": scenario_title,
                    "Model A": m_a,
                    "Model B": m_b,
                    "Diff mAP (A - B)": f"{float(np.mean(sc_a) - np.mean(sc_b)):+.3f}",
                    "95% CI of Diff": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
                    "Empirical p-value": f"{p_val:.4f}",
                    "Ý nghĩa thống kê (p < 0.05)": "Có ý nghĩa (*)" if is_sig else "Không ý nghĩa (ns)"
                })
        return pd.DataFrame(rows)
        
    df_paired_c2 = compute_paired_matrix(res_cleaned, "2. Cleaned Labels")
    df_paired_c3 = compute_paired_matrix(res_strict, "3. Strict Cross-View")
    df_paired_all = pd.concat([df_paired_c2, df_paired_c3], ignore_index=True)
    df_paired_all.to_csv("results/paired_bootstrap_matrix.csv", index=False, encoding="utf-8")
    
    logger.info("Hoàn thành benchmark chuẩn mực! Xuất bảng kết quả và ma trận p-value.")
    return df_compare, df_paired_all

if __name__ == "__main__":
    run_all_benchmarks()
