import torch
from sentence_transformers import SentenceTransformer

class NomicEmbedder:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Uso il modello v1.5 che supporta Matryoshka (opzionale) e long context
        print(f"🧠 Loading Nomic-Embed-Text v1.5 on {self.device}...")
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, local_files_only=False, device=self.device)
        self.model.eval()

    def embed_document(self, text):
        """
        Aggiunge il prefisso obbligatorio 'search_document: ' 
        Serve per i dati che vanno nel DB.
        """
        # Nomic richiede questo prefisso per l'ingestion
        prefixed_text = f"search_document: {text}"
        return self.model.encode(prefixed_text, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """
        Aggiunge il prefisso obbligatorio 'search_query: '
        Serve per la domanda dell'utente.
        """
        prefixed_text = f"search_query: {text}"
        return self.model.encode(prefixed_text, convert_to_numpy=True).tolist()
    
    def free_vram(self):
        """Libera la memoria GPU."""
        self.model.cpu()
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()