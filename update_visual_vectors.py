from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "video_collection" # Assicurati che sia il nome usato in app.py

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ Collezione {COLLECTION_NAME} eliminata. Tabula rasa completata.")
else:
    print("La collezione non esiste già.")