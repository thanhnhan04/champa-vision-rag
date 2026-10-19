"""
End-to-End Demo Pipeline:
Truy xuất hình ảnh (Visual Retrieval) -> Khớp Hồ sơ hiện vật -> Truy xuất Tri thức RAG -> Sinh thuyết minh có trích dẫn -> Kiểm chứng Hallucination.
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.pipeline.rag_service import ChampaRAGService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_end_to_end_demo(query_idx: int = 332):
    if len(sys.argv) > 1:
        query_idx = int(sys.argv[1])
    emb_dir = Path("embeddings")
    if not (emb_dir / "emb_hybrid_dino_siglip.npy").exists():
        logger.error("Chưa có embedding hybrid! Vui lòng chờ extract_embeddings.py hoàn tất.")
        return

    # 1. Tải embeddings và IDs
    E = np.load(emb_dir / "emb_hybrid_dino_siglip.npy")
    ids_data = json.loads((emb_dir / "ids.json").read_text("utf-8"))
    
    # 2. Xây dựng index FAISS
    dim = E.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(E)
    
    query_item = ids_data[query_idx]
    query_img_id = query_item["image_id"]
    query_art_id = query_item["artifact_id"]
    query_vec = E[query_idx:query_idx+1]
    
    logger.info(f"=== BẮT ĐẦU TRUY XUẤT CHO ẢNH QUERY: {query_img_id} (Artifact: {query_art_id}) ===")
    
    # 3. Tìm kiếm Top-5 hình ảnh tương đồng nhất (loại bỏ chính nó)
    S, I = index.search(query_vec, 10)
    matched_candidates = []
    seen_artifacts = set()
    
    for score, idx in zip(S[0], I[0]):
        if idx == query_idx:
            continue
        c_item = ids_data[idx]
        c_art = c_item["artifact_id"]
        matched_candidates.append({
            "image_id": c_item["image_id"],
            "artifact_id": c_art,
            "similarity": float(score)
        })
        if c_art not in seen_artifacts:
            seen_artifacts.add(c_art)
            
    top_candidate = matched_candidates[0]
    predicted_art_id = top_candidate["artifact_id"]
    is_correct = (predicted_art_id == query_art_id)
    
    logger.info(f"Kết quả truy xuất hình ảnh Top-1:")
    logger.info(f" - Predicted Artifact: {predicted_art_id} (Score: {top_candidate['similarity']:.4f})")
    logger.info(f" - Ground Truth Artifact: {query_art_id}")
    logger.info(f" - Match: {'CHÍNH XÁC (HIT)' if is_correct else 'SAI (MISS)'}")
    
    # 4. RAG Retrieval & Knowledge Augmentation
    rag = ChampaRAGService()
    user_question = "Hãy thuyết minh chi tiết về nguồn gốc, phong cách điêu khắc và ý nghĩa tôn giáo của hiện vật này."
    retrieved_chunks = rag.search_chunks(user_question, artifact_id=predicted_art_id, top_k=3)
    
    logger.info(f"\n=== TRUY XUẤT TRI THỨC RAG CHO {predicted_art_id} ===")
    for c in retrieved_chunks:
        logger.info(f" + [{c['chunk_id']}] {c['section']} (Nguồn: {c['authority']})")
        
    # 5. Sinh Grounded Prompt
    prompt = rag.build_grounded_prompt(predicted_art_id, retrieved_chunks)
    
    # 6. Mẫu thuyết minh mô phỏng và Kiểm chứng Hallucination
    art_info = rag.metadata.get(predicted_art_id, {})
    title = art_info.get("title", "Hiện vật Champa")
    era = art_info.get("era", "chưa rõ niên đại")
    c_ids = [c["chunk_id"] for c in retrieved_chunks]
    
    demo_commentary = f"""I. GIỚI THIỆU TỔNG QUAN & ĐỊNH DANH
Hiện vật là {title}, có niên đại vào khoảng {era} [{c_ids[0]}]. Hiện vật được lưu giữ tại {art_info.get('current_location', 'Bảo tàng')} [{c_ids[0]}].

II. ĐẶC ĐIỂM TẠO HÌNH & Ý NGHĨA BIỂU TƯỢNG TÔN GIÁO
Tác phẩm thể hiện đỉnh cao nghệ thuật điêu khắc Champa, phản ánh tín ngưỡng tôn giáo bản địa kết hợp với ảnh hưởng Ấn Độ giáo [{c_ids[1] if len(c_ids) > 1 else c_ids[0]}]. Các đường nét chạm khắc nổi khối tinh tế, trang phục và đồ trang sức mang đậm đặc trưng thẩm mỹ của thời kỳ này.

III. GIÁ TRỊ LỊCH SỬ & HIỆN TRẠNG BẢO TỒN
Hiện vật là bằng chứng vật chất vô giá cho sự hưng thịnh của nền văn minh Champa cổ [{c_ids[-1]}]. Tài liệu tham khảo hiện tại không có dữ kiện về nhiệt độ bảo quản chính xác trong kho lưu trữ."""

    verif = rag.verify_commentary(demo_commentary, c_ids)
    
    logger.info("\n=== KẾT QUẢ KIỂM CHỨNG THUYẾT MINH (VERIFIER) ===")
    logger.info(f" - Tổng số trích dẫn: {verif['total_citations']}")
    logger.info(f" - Số trích dẫn hợp lệ: {verif['valid_citations_count']}")
    logger.info(f" - Trích dẫn không hợp lệ (Hallucinated): {verif['invalid_citations_count']}")
    logger.info(f" - Citation Precision: {verif['citation_precision']*100:.1f}%")
    logger.info(f" - Có trích dẫn sai nguồn: {verif['has_unsupported_citation']}")
    
    return {
        "query_img_id": query_img_id,
        "query_art_id": query_art_id,
        "predicted_art_id": predicted_art_id,
        "is_correct": is_correct,
        "retrieved_chunks": [c["chunk_id"] for c in retrieved_chunks],
        "verification": verif
    }

if __name__ == "__main__":
    run_end_to_end_demo()

