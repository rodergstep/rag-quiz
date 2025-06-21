import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Union
import os
import logging

# Configure logging for VectorStore
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages a ChromaDB vector store for storing and retrieving document embeddings.
    Supports persistent storage, adding documents with rich metadata, and similarity search
    with diversity filtering.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initializes the VectorStore.

        Args:
            persist_directory (str): The directory where ChromaDB will store its data.
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.setup_chromadb()  # Call setup on initialization

    def setup_chromadb(self):
        """
        Initializes the ChromaDB client with persistence.
        Ensures the persistence directory exists.
        """
        try:
            # Ensure the directory for persistent storage exists
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(
                f"ChromaDB persistence directory ensured: {self.persist_directory}")

            # Initialize PersistentClient
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry for privacy
                    # Allow resetting the database if needed (e.g., for development)
                    allow_reset=True
                )
            )
            logger.info("ChromaDB PersistentClient initialized.")
        except Exception as e:
            logger.critical(
                f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise  # Re-raise to ensure the application knows initialization failed

    def create_collection(self, collection_name: str = "medical_guidelines"):
        """
        Creates or gets an existing ChromaDB collection.

        Args:
            collection_name (str): The name of the collection to create or retrieve.
        """
        if not self.client:
            logger.error(
                "ChromaDB client is not initialized. Cannot create or get collection.")
            raise Exception("ChromaDB client not initialized.")

        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(
                f"Loaded existing collection: '{collection_name}' with {self.collection.count()} documents.")
        except Exception:  # ChromaDB raises a generic Exception if collection doesn't exist
            # This catch is broad. If debugging persistence issues, you might want a more specific
            # catch for CollectionNotFoundException and allow other errors to bubble up.
            self.collection = self.client.create_collection(
                name=collection_name,
                # Use cosine similarity for embeddings
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: '{collection_name}'.")

    def add_documents(self, texts: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]] = None, pdf_id: str = None):
        """
        Adds documents (text, embeddings, and metadata) to the ChromaDB collection.

        Args:
            texts (List[str]): A list of text strings for the documents.
            embeddings (List[List[float]]): A list of embedding vectors,
                                            corresponding to each text.
            metadatas (List[Dict[str, Any]], optional): A list of dictionaries,
                                                        each containing metadata for a document.
                                                        Defaults to None, in which case basic metadata is generated.
            pdf_id (str, optional): A unique identifier for the PDF document these chunks originate from.
                                    If provided, it will be added to each chunk's metadata.

        Raises:
            Exception: If the collection is not initialized, or if there's a mismatch
                       in input lengths, or an error during the add operation.
        """
        if not self.collection:
            logger.error(
                "Collection not initialized. Call create_collection() first.")
            raise Exception("Collection not initialized")

        if not texts or not embeddings:
            logger.warning("No documents or embeddings provided to add.")
            return  # Do nothing if no data

        if len(texts) != len(embeddings):
            logger.error(
                f"Mismatch: {len(texts)} texts vs {len(embeddings)} embeddings.")
            raise Exception("Number of texts and embeddings must match")

        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()) for _ in texts]

        # Validate and ensure metadata exists for each document
        processed_metadatas = []
        if metadatas == None:
            # Create empty dicts if None provided
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            logger.warning(
                f"Metadata list length ({len(metadatas)}) does not match texts length ({len(texts)}). Generating default metadata.")
            metadatas = [{}] * len(texts)

        for i, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                logger.warning(
                    f"Metadata for document {i} is not a dictionary ({type(meta)}). Converting to empty dict.")
                meta = {}

            # Ensure essential metadata fields, especially for RAG context
            # Use .get() safely to check if keys exist and provide defaults
            clean_meta = {
                "source": meta.get("source", "pdf"),
                "chunk_id": meta.get("chunk_id", str(i)),
                # Default page 0 if not provided
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "heading": meta.get("heading", "No Heading"),
                "chunk_type": meta.get("chunk_type", "text"),
                "length": meta.get("length", len(texts[i])),
                # original_element_ids can be a list, ensure it's JSON-serializable if not primitive
                # Convert list to string for ChromaDB metadata if it contains non-primitive types or is a complex list
                "original_element_ids": str(meta.get("original_element_ids", []))
            }
            # Add pdf_id to metadata if provided
            if pdf_id:
                clean_meta["pdf_id"] = pdf_id

            processed_metadatas.append(clean_meta)

        try:
            logger.debug(
                f"Attempting to add {len(texts)} documents to collection.")
            # Only log large data if debug is on
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"First 3 metadata entries for add: {processed_metadatas[:3]}")
                logger.debug(f"First 3 IDs for add: {ids[:3]}")

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=processed_metadatas,
                ids=ids
            )
            logger.info(
                f"Successfully added {len(texts)} documents to collection (pdf_id: {pdf_id}). Current total documents: {self.collection.count()}")
        except Exception as e:
            logger.error(
                f"Error adding documents to collection (pdf_id: {pdf_id}): {e}", exc_info=True)
            raise Exception(f"Error adding documents to collection: {str(e)}")

    def similarity_search(self, query_embedding: List[float],
                          k: int = 10, diversity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Searches for similar documents in the collection and applies diversity filtering.
        Diversity is promoted by preferring chunks from different pages or distinct headings.

        Args:
            query_embedding (List[float]): The embedding vector of the query.
            k (int): The desired number of diverse top similar documents to return.
            diversity_threshold (float): A threshold (0.0-1.0) to control how "different"
                                         subsequent chunks should be from already selected ones
                                         if exact page/heading duplication is not the only filter.
                                         (Currently primarily uses page/heading for diversity).

        Returns:
            Dict[str, Any]: A dictionary containing 'documents', 'distances', and 'metadatas'
                            of the selected diverse results.

        Raises:
            Exception: If the collection is not initialized.
        """
        if not self.collection:
            logger.error(
                "Collection not initialized. Cannot perform similarity search.")
            raise Exception("Collection not initialized")

        # Retrieve more results than needed to allow for effective diversity filtering
        # A common heuristic is 2x or 3x k, or k + a buffer.
        retrieve_k = max(2 * k, k + 5)

        try:
            logger.debug(
                f"Performing similarity search for top {retrieve_k} results.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=retrieve_k,
                include=['documents', 'distances', 'metadatas']
            )

            # Extract results, handling potential empty returns gracefully
            docs = results['documents'][0] if results['documents'] else []
            dists = results['distances'][0] if results['distances'] else []
            metas = results['metadatas'][0] if results['metadatas'] else []

            if not docs:
                logger.info("No documents found for the given query.")
                return {'documents': [], 'distances': [], 'metadatas': []}

            selected_docs = []
            selected_distances = []
            selected_metadatas = []

            # Sets to track uniqueness based on key metadata
            seen_page_heading_combos = set()
            seen_chunk_ids = set()  # Fallback uniqueness

            # Iterate through retrieved results, prioritizing diversity
            for i in range(len(docs)):
                doc = docs[i]
                dist = dists[i]
                meta = metas[i]

                # Extract key metadata for diversity check
                page_start = meta.get('page_start')
                heading = meta.get('heading')
                chunk_id = meta.get('chunk_id')  # Always available

                # Create a unique identifier for this chunk's "section"
                section_identifier = (
                    page_start, heading) if page_start is not None else None

                # Apply diversity filtering:
                # 1. Prefer chunks from distinct page+heading combinations
                # 2. As a fallback, ensure chunk_id is unique (basic uniqueness)
                is_unique = True
                if section_identifier and section_identifier in seen_page_heading_combos:
                    is_unique = False
                elif chunk_id and chunk_id in seen_chunk_ids:
                    is_unique = False

                # Further refine diversity: for text content, avoid very similar chunks
                # This check is more computationally intensive and can be tuned or removed
                # if page/heading is sufficient for diversity.
                # For now, rely heavily on metadata based diversity.

                if is_unique:
                    selected_docs.append(doc)
                    selected_distances.append(dist)
                    selected_metadatas.append(meta)

                    if section_identifier:
                        seen_page_heading_combos.add(section_identifier)
                    if chunk_id:
                        seen_chunk_ids.add(chunk_id)

                # Stop when we have enough diverse documents
                if len(selected_docs) >= k:
                    break

            # If not enough diverse documents were found, fill up with top remaining (less diverse)
            if len(selected_docs) < k:
                logger.warning(
                    f"Only found {len(selected_docs)} diverse documents. Filling with top remaining for total of {k}.")
                for i in range(len(docs)):
                    if len(selected_docs) >= k:
                        break
                    doc = docs[i]
                    dist = dists[i]
                    meta = metas[i]

                    # Add if not already selected, even if not strictly "diverse" by current criteria
                    if doc not in selected_docs:  # Simple check to avoid duplicates from this pass
                        selected_docs.append(doc)
                        selected_distances.append(dist)
                        selected_metadatas.append(meta)

            logger.info(
                f"Similarity search retrieved {len(docs)} documents and selected {len(selected_docs)} diverse documents.")
            return {
                'documents': selected_docs,
                'distances': selected_distances,
                'metadatas': selected_metadatas
            }
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            raise Exception(f"Error during similarity search: {str(e)}")

    # Changed return type to int as per current implementation
    def get_collection_info(self) -> int:
        """
        Gets information about the collection, specifically the number of documents.

        Returns:
            int: The number of documents in the collection. Returns 0 if the collection
                 is not initialized or an error occurs.
        """
        if self.collection:
            try:
                count = self.collection.count()
                logger.debug(
                    f"Collection '{self.collection.name}' contains {count} documents.")
                return count
            except Exception as e:
                logger.error(
                    f"Error getting collection count: {e}", exc_info=True)
                return 0  # Return 0 on error
        logger.info("Collection not initialized, returning 0 documents.")
        return 0

    def get_documents_by_pdf_id(self, pdf_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all documents (chunks) associated with a specific PDF ID.

        Args:
            pdf_id (str): The unique identifier of the PDF document.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'document' (text),
                                  'id' (ChromaDB internal ID), and 'metadata' for the retrieved chunks.
                                  Returns an empty list if no documents are found for the given PDF ID.
        """
        if not self.collection:
            logger.error(
                "Collection not initialized. Cannot retrieve documents by PDF ID.")
            return []
        try:
            logger.debug(f"Querying for documents with pdf_id: '{pdf_id}'.")
            # FIX: Provide a dummy query_embeddings to satisfy ChromaDB's requirement
            # The exact embedding doesn't matter since the 'where' clause filters by pdf_id.
            # We use a placeholder list of 0.0s, assuming a typical embedding dimension.
            # A more robust approach would be to get the actual embedding dimension from the model,
            # but for a metadata-only query, a small non-zero vector might also work.
            # For simplicity and to avoid importing the embedding model here, we assume a small dimension.
            # The actual dimension will be checked by ChromaDB during add.
            # A common embedding dimension (e.g., for BGE models)
            dummy_embedding = [0.0] * 1024

            results = self.collection.query(
                # Provide a dummy embedding
                query_embeddings=[dummy_embedding],
                n_results=10000,  # Retrieve a large number to ensure all matching documents are found
                where={"pdf_id": pdf_id},
                # FIX: Removed 'ids' from include list as it's not a valid option for 'include'
                include=['documents', 'metadatas']
            )

            retrieved_chunks = []
            # results can be None or empty lists if no matches
            # The structure of results['documents'] is a list of lists (outer list for queries, inner for documents)
            # Ensure the inner list is not empty
            if results and results.get('documents') and results['documents'][0]:
                # Check for each component, making sure they exist and have content
                # FIX: Access ids directly from results, as it's not requested in 'include' but returned by query.
                # However, for this specific metadata query, ChromaDB might not return 'ids' by default if not specified.
                # A safer approach is to ensure 'ids' is also included in the `add` operation and then access it.
                # Since the current `add` uses UUIDs, ChromaDB does return them.
                if results.get('metadatas') and results.get('ids') and \
                   results['metadatas'][0] and results['ids'][0]:
                    for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
                        retrieved_chunks.append({
                            "document": doc,
                            "metadata": meta,
                            "id": doc_id
                        })
            logger.info(
                f"Retrieved {len(retrieved_chunks)} documents for pdf_id: {pdf_id}. Raw query result sample (first doc): {results['documents'][0][0][:100] if retrieved_chunks else 'N/A'}")
            return retrieved_chunks
        except Exception as e:
            logger.error(
                f"Error retrieving documents by pdf_id '{pdf_id}': {e}", exc_info=True)
            return []  # Return empty list on error

    def delete_documents_by_pdf_id(self, pdf_id: str):
        """
        Deletes all documents (chunks) associated with a specific PDF ID from the collection.

        Args:
            pdf_id (str): The unique identifier of the PDF document whose chunks are to be deleted.
        """
        if not self.collection:
            logger.error(
                "Collection not initialized. Cannot delete documents by PDF ID.")
            return

        try:
            logger.info(
                f"Attempting to delete documents for pdf_id: {pdf_id}.")
            self.collection.delete(where={"pdf_id": pdf_id})
            logger.info(
                f"Successfully deleted documents associated with pdf_id: {pdf_id}. New total documents: {self.collection.count()}")
        except Exception as e:
            logger.error(
                f"Error deleting documents for pdf_id '{pdf_id}': {e}", exc_info=True)
            raise Exception(f"Error deleting documents by pdf_id: {str(e)}")

    def reset_collection(self):
        """
        Resets (deletes) the current collection if it exists, and then recreates it.
        Useful for starting with a clean slate in development.
        """
        if self.client and self.collection:
            try:
                collection_name = self.collection.name
                self.client.delete_collection(collection_name)
                logger.info(f"Collection '{collection_name}' deleted.")
                self.collection = None  # Clear the reference
                # Recreate the collection
                self.create_collection(collection_name)
                logger.info(f"Collection '{collection_name}' recreated.")
            except Exception as e:
                logger.error(f"Error resetting collection: {e}", exc_info=True)
                raise Exception(f"Error resetting collection: {str(e)}")
        elif self.client:
            logger.warning(
                "No active collection to reset. Creating a default one.")
            self.create_collection()  # Ensure a collection exists
        else:
            logger.warning(
                "ChromaDB client not initialized. Cannot reset collection.")
