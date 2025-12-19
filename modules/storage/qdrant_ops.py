from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

class VectorDB:
    def __init__(self, collection_name="video_rag", vector_size=512):
        """
        Initialize connection to local Docker Qdrant instance.
        vector_size: 512 for ViT-B/32 (CLIP), 768 for ViT-L/14, etc.
        """
        # Connect to Docker container on localhost
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Verify connection
        try:
            self.client.get_collections()
            print("✅ Connected to Qdrant Docker Service")
            self._ensure_collection_exists()
        except Exception as e:
            print(f"❌ CRITICAL: Could not connect to Qdrant. Is Docker running? \nError: {e}")

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't exist yet."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection '{self.collection_name}' with size {self.vector_size}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Collection '{self.collection_name}' is ready.")

    def upload_batch(self, vectors, payloads):
        """
        Uploads a list of vectors in one network request.
        vectors: List[List[float]]
        payloads: List[dict]
        """

        if len(vectors) != len(payloads):
            raise ValueError("Vectors and payloads must have the same length.")

        if any (len(v) != self.vector_size for v in vectors):
            raise ValueError("All vectors must have the same size as the collection.")

        # Bulk upload
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
            for v, p in zip(vectors, payloads)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ Batch uploaded: {len(points)} items")
    
    def search(self, query_vector, top_k):
        """
        Finds the most similar data to the query vector.
        """
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
        return hits
    


