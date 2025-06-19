from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingHandler:
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or (
            'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = SentenceTransformer(
                self.model_name, device=self.device)
            print(f"Embedding model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading embedding model on {self.device}: {e}")
            self.model = SentenceTransformer(self.model_name, device='cpu')

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Embed documents (use instruction format if needed)"""
        texts = [
            f"Represent this document for retrieval: {text}" for text in texts]
        return self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed query with instruction"""
        query = f"Represent this query for retrieval: {query}"
        return self.model.encode([query], convert_to_numpy=True)[0]
