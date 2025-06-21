import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

# Configure logging for the embedding handler
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingHandler:
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else self._get_optimal_device()
        self.model = None
        self.load_model()

        # BGE models specifically benefit from these instructions
        # Refer to HuggingFace model card for exact instructions if different model is used
        self.document_instruction = "Represent this document for retrieval: "
        self.query_instruction = "Represent this query for retrieval: "

    def _get_optimal_device(self) -> str:
        """Determines the optimal device for Torch."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_model(self):
        """Loads the SentenceTransformer model."""
        try:
            # Add a check if model is already loaded
            if self.model is not None:
                logger.info(f"Model {self.model_name} already loaded.")
                return

            self.model = SentenceTransformer(
                self.model_name, device=self.device)
            logger.info(
                f"Embedding model '{self.model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{self.model_name}' on {self.device}. Attempting fallback to CPU.", exc_info=True)
            try:
                self.model = SentenceTransformer(self.model_name, device='cpu')
                self.device = 'cpu'  # Update device to reflect fallback
                logger.info(
                    f"Embedding model '{self.model_name}' loaded successfully on CPU as fallback.")
            except Exception as cpu_e:
                logger.critical(
                    f"Failed to load embedding model '{self.model_name}' even on CPU. Please check your installation and network. Error: {cpu_e}", exc_info=True)
                raise RuntimeError(f"Could not load embedding model: {cpu_e}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of documents using the model with the document instruction.
        Ensures input is a list of strings and handles empty inputs gracefully.
        """
        if not texts:
            logger.warning(
                "No texts provided for embedding. Returning empty array.")
            return np.array([])

        # Apply the document instruction to each text
        processed_texts = [self.document_instruction + text for text in texts]

        try:
            embeddings = self.model.encode(
                processed_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Crucial for cosine similarity with BGE models
            )
            logger.info(f"Encoded {len(texts)} document texts.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding document texts: {e}", exc_info=True)
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string with the query instruction.
        Returns a 1D numpy array (vector).
        """
        if not query:
            logger.warning(
                "Empty query provided for embedding. Returning zero vector.")
            # Return a zero vector of correct dimension
            return np.zeros(self.model.get_sentence_embedding_dimension())

        # Apply the query instruction
        processed_query = self.query_instruction + query

        try:
            embedding = self.model.encode(
                [processed_query],
                convert_to_numpy=True,
                normalize_embeddings=True  # Crucial for cosine similarity with BGE models
            )[0]  # Get the first (and only) embedding
            logger.debug(f"Encoded query: '{query[:50]}...'")  # Log a snippet
            return embedding
        except Exception as e:
            logger.error(f"Error encoding query '{query}': {e}", exc_info=True)
            raise


# Optional CLI for testing (can be moved to a separate test file)
if __name__ == "__main__":
    # Assuming pdf_processor.py is in the same directory
    from pdf_processor import PDFProcessor
    import os

    # Setup logger for main block
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)

    # Create a dummy PDF file for testing (or use an existing one)
    dummy_pdf_path = "dummy_test_document.pdf"
    if not os.path.exists(dummy_pdf_path):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(dummy_pdf_path)
            c.drawString(100, 750, "This is a test document.")
            c.drawString(100, 730, "Kapitel 1: Introduction to Testing")
            c.drawString(
                100, 710, "This chapter discusses the importance of testing in software development.")
            c.drawString(
                100, 690, "Empfehlung Nr. 1.1: Always write unit tests.")
            c.save()
            main_logger.info(f"Created a dummy PDF: {dummy_pdf_path}")
        except ImportError:
            main_logger.warning(
                "ReportLab not installed. Cannot create dummy PDF. Please provide an existing PDF for testing.")
            dummy_pdf_path = None

    if dummy_pdf_path and os.path.exists(dummy_pdf_path):
        try:
            pdf_processor = PDFProcessor()
            embedding_handler = EmbeddingHandler()

            extracted_text = pdf_processor.extract_text_from_pdf(
                dummy_pdf_path, to_markdown=True)
            main_logger.info("Text extracted from PDF.")

            chunks = pdf_processor.chunk_text(
                extracted_text, with_metadata=True)
            main_logger.info(f"Generated {len(chunks)} chunks.")

            if chunks:
                # Example: embed the first chunk
                first_chunk_text = chunks[0]['text'] if isinstance(
                    chunks[0], dict) else chunks[0]
                main_logger.info(
                    f"Embedding first chunk (preview): {first_chunk_text[:100]}...")
                chunk_embedding = embedding_handler.encode_texts(
                    [first_chunk_text])
                main_logger.info(
                    f"Shape of first chunk embedding: {chunk_embedding.shape}")

                # Example: embed a query
                query_text = "What is the main recommendation?"
                query_embedding = embedding_handler.encode_query(query_text)
                main_logger.info(
                    f"Shape of query embedding: {query_embedding.shape}")

                # Calculate cosine similarity (simple example)
                if chunk_embedding.shape[0] > 0:
                    from sklearn.metrics.pairwise import cosine_similarity
                    # Reshape query_embedding for cosine_similarity if it's 1D
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1), chunk_embedding)[0][0]
                    main_logger.info(
                        f"Cosine similarity between query and first chunk: {similarity:.4f}")
                else:
                    main_logger.warning(
                        "No chunks to compare similarity with.")
            else:
                main_logger.warning(
                    "No chunks generated for embedding testing.")

        except Exception as e:
            main_logger.error(
                f"An error occurred during testing: {e}", exc_info=True)
    else:
        main_logger.info(
            "Skipping embedding test as no dummy PDF is available.")
