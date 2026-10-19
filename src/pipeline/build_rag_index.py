"""
Xây dựng chỉ mục RAG (Chunking, Metadata tagging, BM25 & Semantic Index)
cho 4 tài liệu tri thức chuẩn mực về Điêu khắc Champa trong data/rag_knowledge/.
"""
import re
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOC_METADATA = {
    "01_national_treasures.md": {
        "doc_id": "DOC01_TREASURES",
        "category": "national_treasures",
        "authority": "Thủ tướng Chính phủ / Cục Di sản văn hóa"
    },
    "02_art_styles_periods.md": {
        "doc_id": "DOC02_STYLES",
        "category": "art_styles_periods",
        "authority": "Học giả EFEO (Philippe Stern, Jean Boisselier)"
    },
    "03_iconography_mythology.md": {
        "doc_id": "DOC03_ICONOGRAPHY",
        "category": "iconography_mythology",
        "authority": "Khảo cổ học & Nghiên cứu biểu tượng Ấn Độ giáo - Phật giáo"
    },
    "04_museum_collections.md": {
        "doc_id": "DOC04_MUSEUMS",
        "category": "museum_collections",
        "authority": "Bảo tàng Điêu khắc Chăm Đà Nẵng, BTLS TP.HCM, Guimet, The Met"
    }
}

def split_markdown_into_chunks(md_path: Path, doc_info: dict) -> list:
    content = md_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    chunks = []
    current_section = "Tổng quan"
    current_lines = []
    chunk_idx = 1
    
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if len(chunk_text) > 80:  # Bỏ qua đoạn rác quá ngắn
                    chunks.append({
                        "chunk_id": f"{doc_info['doc_id']}_C{chunk_idx:02d}",
                        "doc_id": doc_info["doc_id"],
                        "category": doc_info["category"],
                        "authority": doc_info["authority"],
                        "section": current_section,
                        "text": chunk_text,
                        "file_source": md_path.name
                    })
                    chunk_idx += 1
                current_lines = []
            current_section = line.lstrip("#").strip()
            current_lines.append(line)
        else:
            current_lines.append(line)
            
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if len(chunk_text) > 80:
            chunks.append({
                "chunk_id": f"{doc_info['doc_id']}_C{chunk_idx:02d}",
                "doc_id": doc_info["doc_id"],
                "category": doc_info["category"],
                "authority": doc_info["authority"],
                "section": current_section,
                "text": chunk_text,
                "file_source": md_path.name
            })
            
    return chunks

def build_knowledge_chunks():
    kb_dir = Path("data/rag_knowledge")
    all_chunks = []
    
    for filename, info in DOC_METADATA.items():
        doc_path = kb_dir / filename
        if not doc_path.exists():
            logger.warning(f"Không tìm thấy file: {doc_path}")
            continue
        chunks = split_markdown_into_chunks(doc_path, info)
        logger.info(f"Đã trích xuất {len(chunks)} chunks từ {filename}")
        all_chunks.extend(chunks)
        
    out_json = kb_dir / "knowledge_chunks.json"
    out_jsonl = kb_dir / "knowledge_chunks.jsonl"
    
    out_json.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành! Đã lưu {len(all_chunks)} chunks vào {out_json} và {out_jsonl}")
    return all_chunks

if __name__ == "__main__":
    build_knowledge_chunks()

