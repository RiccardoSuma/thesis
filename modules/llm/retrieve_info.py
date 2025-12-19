import clip
import torch
from storage.qdrant_ops import VectorDB

class InfoRetriever:
    def __init__(self, db_client: VectorDB, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.db = db_client

    def get_answer(self, query_text: str, top_k: int = 5):
        """
        Retrieves top_k relevant information from the vector database based on the query text.
        """
        # 1. Encode the query text using CLIP
        text_input = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        query_vector = text_features.cpu().numpy()[0].tolist()
        
        # 2. Search in the vector database
        hits = self.db.search(query_vector=query_vector, top_k=top_k)

        # 3. Extract and return relevant information
        context = []
        for hit in hits:
            payload = hit.payload
            context.append(payload)
        return context