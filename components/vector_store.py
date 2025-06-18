import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import os


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.setup_chromadb()

    def setup_chromadb(self):
        """Initialize ChromaDB client"""
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def create_collection(self, collection_name: str = "medical_guidelines"):
        """Create or get collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")

    def add_documents(self, texts: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]] = None):
        """Add documents to the collection"""
        if not self.collection:
            raise Exception("Collection not initialized")

        if not texts or not embeddings:
            raise Exception("No documents or embeddings provided")

        if len(texts) != len(embeddings):
            raise Exception("Number of texts and embeddings must match")

        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in texts]

        # Ensure metadata is a list of non-empty dictionaries
        if metadatas is None:
            metadatas = [{"source": "pdf", "chunk_id": str(
                i)} for i in range(len(texts))]
        else:
            # Validate and fix metadata
            valid_metadatas = []
            for i, meta in enumerate(metadatas):
                if not isinstance(meta, dict):
                    meta = {}
                if not meta:
                    meta = {"source": "pdf", "chunk_id": str(i)}
                valid_metadatas.append(meta)
            metadatas = valid_metadatas

        # Ensure metadata length matches texts length
        if len(metadatas) != len(texts):
            metadatas = [{"source": "pdf", "chunk_id": str(
                i)} for i in range(len(texts))]

        # Add to collection
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} documents to collection")
        except Exception as e:
            raise Exception(f"Error adding documents to collection: {str(e)}")

    def similarity_search(self, query_embedding: List[float],
                          k: int = 10) -> Dict[str, Any]:
        """Search for similar documents"""
        if not self.collection:
            raise Exception("Collection not initialized")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'distances', 'metadatas']
        )

        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }

    def get_collection_info(self):
        """Get information about the collection"""
        if self.collection:
            return self.collection.count()
        return 0
