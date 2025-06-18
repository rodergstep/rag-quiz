from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch


class EmbeddingHandler:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the embedding model"""
        try:
            # Use MPS (Metal Performance Shaders) for Apple Silicon
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"Embedding model loaded on {device}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to CPU
            self.model = SentenceTransformer(self.model_name, device='cpu')

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise Exception("Embedding model not loaded")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode([query])[0]
