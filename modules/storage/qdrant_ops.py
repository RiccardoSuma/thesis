from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, 
    TextIndexParams, TokenizerType, PayloadSchemaType
)
import subprocess
import time
import requests
import os
import hashlib

def ensure_qdrant_running():
    """Automates Qdrant startup using a waterfall of Docker commands."""
    url = "http://localhost:6333/healthz"
    
    # Step 1: Check if already running
    try:
        requests.get(url, timeout=2)
        print("✅ Qdrant is already running.")
        return
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("⚠️ Qdrant is down. Automating startup...")

    # Step 2: Define the Command Waterfall
    # We try Compose first, then fallback to raw Docker run
    cwd = os.getcwd()
    commands = [
        ["sudo", "docker", "compose", "up", "-d"],
        ["sudo", "docker-compose", "up", "-d"],
        # The 'Direct Run' fallback if Compose is missing/broken
        ["sudo", "docker", "run", "-d", "--name", "qdrant_video", 
         "-p", "6333:6333", "-p", "6334:6334", 
         "-v", f"{cwd}/qdrant_data:/qdrant/storage", "qdrant/qdrant"]
    ]

    started = False
    for cmd in commands:
        try:
            # We use capture_output=False so you can see the sudo password prompt if needed
            print(f"🔄 Trying: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                started = True
                break
        except Exception as e:
            continue

    # Step 3: Verify Health
    if started:
        print("⏳ Waiting for Qdrant health check...")
        for _ in range(10):
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    print("✅ Qdrant is ready.")
                    return
            except:
                time.sleep(1)
    
    print("❌ Automation failed. You must run 'sudo docker run...' manually in Terminus.")


class VectorDB:
    def __init__(self, collection_name="abb_video_collection", vector_size=768): 
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()
        ensure_qdrant_running()

    def _ensure_collection_exists(self):
        """Crea la collezione se non esiste."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Collezione '{self.collection_name}' creata.")

    def _generate_id(self, payload):
        """
        Genera un MD5 deterministico unico.
        Includendo 'chunk_index' evitiamo che i vari pezzi di una slide si sovrascrivano.
        """
        source = payload.get('source', '')
        ts = payload.get('timestamp', 0)
        mod = payload.get('modality', 'unknown')
        # Usiamo 0 come default se il punto non è un chunk (es. l'audio)
        idx = payload.get('chunk_index', 0) 
        
        unique_str = f"{source}_{ts}_{mod}_{idx}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def upload_batch(self, vectors, payloads):
        """L'unica funzione che deve fare upload su Qdrant."""
        if not vectors or not payloads:
            return

        points = []
        for v, p in zip(vectors, payloads):
            point_id = self._generate_id(p)
            points.append(PointStruct(
                id=point_id,
                vector=v,
                payload=p
            ))
        
        # Upsert: se l'ID esiste, lo sovrascrive (evita duplicati infiniti)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"📦 Batch caricato: {len(points)} punti indicizzati.")

    def point_exists(self, payload_or_id):
            """Verifica se un punto esiste. Accetta sia il payload (dict) che l'ID (str)."""
            # DEBUG: Vediamo cosa arriva davvero
            #print(f"DEBUG: point_exists received type: {type(payload_or_id)}") 

            if isinstance(payload_or_id, dict):
                # Se è un dict, estraiamo l'ID
                point_id = self._generate_id(payload_or_id)
            elif isinstance(payload_or_id, str):
                # Se è già una stringa (ID), la usiamo direttamente
                point_id = payload_or_id
            else:
                # Se arriva altro, evitiamo il crash e restituiamo False
                return False
                
            try:
                res = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=False,
                    with_vectors=False
                )
                return len(res) > 0
            except Exception:
                return False
            
    # modules/storage/qdrant_ops.py

    def search(self, query_vector, top_k=5, query_filter=None, with_vectors=False):
        """
        Esegue la ricerca query_points.
        Aggiunto parametro 'with_vectors' per supportare MMR.
        """
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter, 
            limit=top_k,
            with_payload=True,
            with_vectors=with_vectors # <--- CRUCIALE
        ).points

    def scroll(self, scroll_filter, limit=15):
        """
        Wrapper per lo scroll (necessario per la Sliding Window temporale).
        Restituisce (points, next_page_offset).
        """
        return self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            limit=limit
        )