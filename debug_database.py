from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

# Configurazione
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "video_collection"
SEARCH_TERM = "RNN" # Prova anche "recurrent" o "networks"

def debug_database():
    print(f"🔎 Cerco '{SEARCH_TERM}' nel database (Full-Text Match)...")
    
    try:
        # Cerchiamo nel campo 'full_ocr' (Visual) e 'text' (Audio)
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="full_ocr", match=MatchText(text=SEARCH_TERM)),
                    FieldCondition(key="text", match=MatchText(text=SEARCH_TERM))
                ]
            ),
            limit=10,
            with_payload=True
        )

        points = results[0]
        
        if not points:
            print(f"❌ Nessun riscontro per '{SEARCH_TERM}'.")
            print("Il termine non è stato letto correttamente dall'OCR o non è presente nelle lezioni.")
            return

        print(f"✅ Trovati {len(points)} punti contenenti '{SEARCH_TERM}':\n")
        for p in points:
            payload = p.payload
            print(f"--- Punto ID: {p.id} ---")
            print(f"Modality: {payload.get('modality')}")
            print(f"Source:   {payload.get('source')} @ {payload.get('timestamp')}s")
            # Mostriamo un pezzetto del testo dove ha trovato il match
            content = payload.get('full_ocr') if payload.get('modality') == 'visual' else payload.get('text')
            print(f"Snippet:  {content[:150]}...")
            print("-" * 30)

    except Exception as e:
        print(f"🚨 Errore durante il debug: {e}")

if __name__ == "__main__":
    debug_database()