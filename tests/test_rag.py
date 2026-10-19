import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.rag_service import ChampaRAGService

def test_rag():
    rag = ChampaRAGService()
    chunks = rag.search_chunks("Đài thờ Trà Kiệu vũ nữ Apsara", artifact_id="Q10800366", top_k=3)
    print(f"Top 3 chunks: {[c['chunk_id'] for c in chunks]}")
    for c in chunks:
        print(f" - [{c['chunk_id']}] ({c['authority']}): {c['section']}")
        
    prompt = rag.build_grounded_prompt("Q10800366", chunks)
    print("\nSample Prompt Generated (length:", len(prompt), ")")
    
    # Test verifier
    sample_commentary = "Đài thờ Trà Kiệu là bảo vật quốc gia [DOC01_TREASURES_C01]. Điêu khắc thể hiện phong cách Trà Kiệu [DOC02_STYLES_C03]."
    check = rag.verify_commentary(sample_commentary, [c['chunk_id'] for c in chunks])
    print("\nVerifier Test:", check)

if __name__ == "__main__":
    test_rag()
