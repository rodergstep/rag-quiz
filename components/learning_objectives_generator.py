import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
import re  # Import re for regex in JSON parsing

# Assuming these are in your components directory
from components.llm_handler import LocalLLMHandler
from components.vector_store import VectorStore
from components.embeddings import EmbeddingHandler

# Configure logging for the Learning Objectives Generator
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LearningObjectivesGenerator:
    """
    Generates structured learning objectives for medical continuing education
    based on retrieved context from a vector store, utilizing a local LLM.
    """

    def __init__(self, llm_handler: LocalLLMHandler, vector_store: VectorStore, embedding_handler: EmbeddingHandler):
        """
        Initializes the LearningObjectivesGenerator.

        Args:
            llm_handler (LocalLLMHandler): An instance of the local LLM handler.
            vector_store (VectorStore): An instance of the vector store for document retrieval.
            embedding_handler (EmbeddingHandler): An instance of the embedding handler for query embedding.
        """
        self.llm_handler = llm_handler
        self.vector_store = vector_store
        self.embedding_handler = embedding_handler

    def generate_learning_objectives(self, pdf_id: str, target_group: Optional[str] = None) -> pd.DataFrame:
        """
        Generates learning objectives based on all available context for a given PDF ID and target group.

        Args:
            pdf_id (str): The unique ID of the PDF to retrieve all context from.
            target_group (Optional[str]): The medical target audience for the learning objectives.
                                          E.g., "General Medicine", "Internal Medicine".

        Returns:
            pd.DataFrame: A DataFrame containing the parsed learning objectives,
                          or an empty DataFrame if none could be generated or parsed.
        """
        logger.info(
            f"Generating learning objectives for PDF ID: '{pdf_id}', target_group: '{target_group}'.")

        # 1. Retrieve all relevant context chunks for the given PDF ID
        context = self._retrieve_all_context_for_pdf(pdf_id)
        if not context:
            logger.warning(
                "No context found for the specified PDF ID. Cannot generate learning objectives.")
            return pd.DataFrame()

        # Check if the context is too large before sending to LLM
        # This is a heuristic. Actual token count depends on tokenization.
        # Max context window is 8192 for the LLM.
        # The prompt itself consumes a significant number of tokens (e.g., 1000-2000 tokens).
        # We'll estimate and warn if the context is extremely large.
        # Estimate max words the LLM can handle after prompt
        if len(context.split()) > (self.llm_handler.n_ctx - 2000):
            logger.warning(
                f"The total context ({len(context.split())} words) for PDF ID '{pdf_id}' is very large and might exceed the LLM's context window ({self.llm_handler.n_ctx} tokens). This could lead to errors or truncated responses.")
            # For now, we proceed, but this is where truncation or more advanced chunking would be needed.

        # 2. Create the learning objective prompt
        prompt = self.llm_handler.create_learning_objective_prompt(
            context=context,
            target_group=target_group
        )
        logger.debug(f"Generated prompt (first 500 chars): {prompt[:500]}...")

        # 3. Generate response from LLM
        try:
            # Increased max_tokens for JSON output as it can be verbose
            # Temperature kept low for more deterministic, fact-based output
            llm_response = self.llm_handler.generate_response(
                prompt, max_tokens=2000, temperature=0.2
            )
            logger.debug(
                f"Raw LLM response (first 500 chars): {llm_response[:500]}...")
        except Exception as e:
            logger.error(
                f"Error getting response from LLM for learning objectives: {e}", exc_info=True)
            return pd.DataFrame([{"status": f"Error generating: {str(e)}",
                                  "thema": "", "kapitel": "", "nummer": "", "zielgruppe": "",
                                  "lernziel": "", "zielkompetenz": "", "kurzfassung": "",
                                  "primäre_literaturquelle": "", "leitlinien_link": "",
                                  "hinweis_fragefokus": "", "specification": "", "sub_specification": ""}])

        # 4. Parse the JSON response
        learning_objectives_data = self._parse_llm_json_response(llm_response)

        if learning_objectives_data:
            logger.info(
                f"Successfully generated and parsed {len(learning_objectives_data)} learning objectives.")
            # Add a 'status' field to each dictionary indicating success
            for lo in learning_objectives_data:
                lo['status'] = 'Generated'
            return pd.DataFrame(learning_objectives_data)
        else:
            logger.warning(
                "No learning objectives could be parsed from the LLM's response or response was empty.")
            # Return a DataFrame with a status message if no objectives were generated
            # Ensure all expected columns are present, even if empty, for DataFrame consistency
            return pd.DataFrame([{"status": "No learning objectives generated or parseable. Try a different topic or less strict target group.",
                                  "thema": "", "kapitel": "", "nummer": "", "zielgruppe": "",
                                  "lernziel": "", "zielkompetenz": "", "kurzfassung": "",
                                  "primäre_literaturquelle": "", "leitlinien_link": "",
                                  "hinweis_fragefokus": "", "specification": "", "sub_specification": ""}])

    def _retrieve_all_context_for_pdf(self, pdf_id: str) -> str:
        """
        Retrieves all documents (chunks) associated with a specific PDF ID from the vector store.

        Args:
            pdf_id (str): The unique ID of the PDF document.

        Returns:
            str: A concatenated string of all retrieved document chunks.
                 Returns an empty string if no chunks are found.
        """
        try:
            # This directly calls the VectorStore method that retrieves all chunks for the PDF_ID
            results = self.vector_store.get_documents_by_pdf_id(pdf_id)

            context_chunks = [item['document']
                              for item in results if 'document' in item]
            if not context_chunks:
                logger.warning(f"No chunks found for PDF ID: '{pdf_id}'.")
                return ""

            # Combine all chunks with a clear separator
            combined_context = "\n\n---\n\n".join(context_chunks)
            logger.info(
                f"Retrieved {len(context_chunks)} total context chunks for PDF ID: '{pdf_id}'.")
            return combined_context
        except Exception as e:
            logger.error(
                f"Error retrieving all context for PDF ID '{pdf_id}': {e}", exc_info=True)
            return ""

    def _parse_llm_json_response(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Parses the LLM's JSON string response into a list of dictionaries.
        Robustly extracts JSON array even with surrounding text.

        Args:
            llm_response (str): The raw string response from the LLM, expected to contain JSON.

        Returns:
            List[Dict[str, Any]]: A list of parsed learning objective dictionaries.
                                  Returns an empty list if parsing fails or result is empty.
        """
        # More robust regex to find exactly the JSON array.
        # It looks for a '[' followed by any characters (non-greedy), and then a ']'
        # to ensure it captures the full array and nothing more.
        # It handles cases where LLM might add text before or after the JSON.
        json_match = re.search(r'\[\s*{[^\]]*?}\s*\]', llm_response, re.DOTALL)
        if not json_match:
            # If a single object is returned instead of an array of objects
            # This regex is specifically for a single object that starts with "thema" key
            json_match = re.search(
                r'{\s*"thema":\s*".*?"(?:,\s*".*?":\s*".*?")*\s*}', llm_response, re.DOTALL)

        if json_match:
            json_string = json_match.group(0)
        else:
            logger.warning(
                "No JSON array structure or single object found in LLM response.")
            logger.debug(f"LLM Response: {llm_response}")
            return []

        try:
            parsed_data = json.loads(json_string)
            # If the LLM returned a single JSON object instead of an an array (common if it only generates one LO)
            if isinstance(parsed_data, dict):
                # Wrap it in a list to ensure consistent output
                parsed_data = [parsed_data]

            if not isinstance(parsed_data, list):
                logger.warning(
                    "Parsed JSON is not a list of objects as expected after wrapping single object.")
                return []

            logger.debug("Successfully parsed LLM's JSON response.")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response as JSON: {e}", exc_info=True)
            # Log start of bad JSON
            logger.debug(
                f"Problematic JSON string that caused error: {json_string[:500]}...")
            return []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
            return []
