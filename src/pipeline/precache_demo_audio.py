"""
Tạo trước bộ đệm Audio Guide cho 5 hiện vật mẫu tiêu biểu:
1. Tượng thần Ganesha (Mỹ Sơn E5 - Bảo vật Quốc gia QĐ 2283)
2. Tượng Bồ tát Tara (Đồng Dương - Bảo vật Quốc gia QĐ 1426)
3. Tượng Phật Đồng Dương (Bảo vật Quốc gia QĐ 2599)
4. Vũ nữ Trà Kiệu (Bảo vật Quốc gia)
5. Đài thờ Mỹ Sơn E1 (Bảo vật Quốc gia QĐ 1426)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from rag_service import ChampaRAGService
from tts_service import ChampaTTSService

def main():
    print("=== TẠO TRƯỚC BỘ ĐỆM AUDIO GUIDE CHO DEMO ===")
    rag = ChampaRAGService('data/rag_knowledge/knowledge_chunks.json', 'data/metadata_cleaned.csv')
    tts = ChampaTTSService()
    
    samples = [
        ("Ganesha Mỹ Sơn E5", "COMMONS_132522171"),
        ("Bồ tát Tara Đồng Dương", "WIKIDATA_Q106654903"),
        ("Tượng Phật Đồng Dương", "COMMONS_95035950"),
        ("Vũ nữ Trà Kiệu", "COMMONS_40764319"),
        ("Đài thờ Mỹ Sơn E1", "WIKIDATA_Q106653085")
    ]
    
    for name, art_id in samples:
        print(f"\nĐang tạo audio cho: {name} (ID: {art_id})...")
        chunks = rag.search_chunks(name, artifact_id=art_id, top_k=3)
        comm = rag.generate_grounded_commentary(art_id, chunks)
        text = comm['commentary']
        
        # 1. Cache Piper
        res_p = tts.synthesize(text, engine="piper", use_cache=True)
        print(f"  [Piper] Lưu cache thành công: {res_p['duration_sec']:.1f}s | Size: {len(res_p['wav_bytes'])} bytes")

    print("\nHoàn tất tạo bộ đệm âm thanh cho toàn bộ hiện vật demo!")

if __name__ == "__main__":
    main()

