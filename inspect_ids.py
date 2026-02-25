from qdrant_client import QdrantClient


client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "abb_video_collection_v2"

ids_to_check = ['86ab204f-771e-2542-e33f-1a37368fa5b7', '2e3d7793-2260-171d-0db9-88ad0ed14a16', 'ca67b8ff-f779-01b8-3c67-2a21a2185a09', 'c5c8462a-13ce-078a-1b75-b61548b8a39e', 'f2f2f92b-22d3-df31-98a6-1c998a7c4e20']

# Fetch massivo
records = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=ids_to_check,
    with_payload=True,
    with_vectors=False
)

# Estrazione: prendo solo il testo e scarto eventuali chunk vuoti
texts = [
    record.payload.get('text', '').strip() 
    for record in records 
    if record.payload and record.payload.get('text')
]

# Unisco tutto in un'unica stringa pronta per Claude
# Uso un separatore netto così il LLM capisce che sono frammenti diversi
final_context_string = "\n\n--- CHUNK ---\n\n".join(texts)

print("Copia da qui in giù:\n")
print(final_context_string)