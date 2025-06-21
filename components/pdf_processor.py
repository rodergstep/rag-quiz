import logging
import re
from typing import List, Union, Dict, Any
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """
    Represents a structured content element extracted from a PDF document.
    This can be a text paragraph, a table, or a figure caption.
    """
    element_id: str
    # The textual content of the element (e.g., paragraph, markdown table, caption)
    content: str
    element_type: str  # 'text', 'table', 'figure_caption', 'heading'
    page_number: int  # The page number where this element originates
    # Additional element-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFProcessor:
    """
    A class to process PDF documents using the Docling library,
    extract text, clean it, and chunk it for Retrieval-Augmented Generation (RAG) purposes.
    It supports markdown conversion, OCR, table structure recognition, and advanced text cleaning.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, backend=PyPdfiumDocumentBackend):
        """
        Initializes the PDFProcessor.

        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.
            backend: The Docling backend to use (e.g., PyPdfiumDocumentBackend or DoclingParseV4DocumentBackend).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.backend = backend
        self.converter = self._setup_converter()

    def _setup_converter(self) -> DocumentConverter:
        """
        Sets up the Docling DocumentConverter with optimized settings for PDF processing.
        Enables OCR and table structure recognition.
        """
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned documents
            pipeline_options.do_table_structure = True  # Extract table structures

            # Safely check and configure table_structure_options if it exists
            table_options = getattr(
                pipeline_options, 'table_structure_options', None)
            if table_options:
                table_options.do_cell_matching = True
                logger.debug(
                    "Configured table_structure_options.do_cell_matching = True")

            # Create a PdfFormatOption object with the defined pipeline options and chosen backend
            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=self.backend
            )

            # Initialize converter with the PdfFormatOption for PDF input
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pdf_format_option,
                }
            )
            logger.info(
                f"Docling DocumentConverter initialized with {self.backend.__name__} and advanced PDF options (OCR, Table Structure).")
            return converter

        except Exception as e:
            logger.warning(
                f"Failed to initialize Docling with advanced options for {self.backend.__name__}. Falling back to basic converter. Error: {e}")
            # Fallback to a basic DocumentConverter if advanced setup fails
            return DocumentConverter()

    def _get_element_page_number(self, docling_element) -> int:
        """Helper to reliably extract the page number from a Docling element."""
        if hasattr(docling_element, 'page'):
            # try page_number attr or direct page object
            return getattr(docling_element.page, 'page_number', None) or docling_element.page
        elif hasattr(docling_element, 'page_number'):
            return docling_element.page_number
        return 0  # Default to 0 or some indicator if page info is missing

    def _extract_content_text(self, docling_element) -> str:
        """Helper to extract text content from various Docling element types."""
        if hasattr(docling_element, 'export_to_markdown'):
            return docling_element.export_to_markdown()  # For tables, rich text
        elif hasattr(docling_element, 'export_to_text'):
            return docling_element.export_to_text()
        elif hasattr(docling_element, 'to_markdown'):
            return docling_element.to_markdown()
        elif hasattr(docling_element, 'to_text'):
            return docling_element.to_text()
        elif hasattr(docling_element, 'text'):
            return docling_element.text
        elif hasattr(docling_element, 'content'):
            return docling_element.content
        return str(docling_element)  # Last resort

    def _process_document_elements(self, docling_document) -> List[DocumentElement]:
        """
        Processes the Docling document to extract all significant content elements
        (text blocks, tables, figures) into a unified list of DocumentElement objects.
        This provides structured output with page-level metadata.
        """
        elements: List[DocumentElement] = []
        element_counter = 0

        # Process main body content (paragraphs, headings, lists)
        if hasattr(docling_document, 'body') and docling_document.body:
            for item in docling_document.body:
                content_text = self._extract_content_text(item)
                element_page = self._get_element_page_number(
                    item) or 1  # Default to page 1 if not found
                element_type = 'text'  # Default type

                # Try to infer element type from Docling's label
                if hasattr(item, 'label'):
                    label_str = str(item.label).lower()
                    if 'heading' in label_str or 'title' in label_str:
                        element_type = 'heading'
                    elif 'list' in label_str:
                        element_type = 'list'
                    elif 'table' in label_str:  # Sometimes tables might be in body
                        element_type = 'table'
                    elif 'figure' in label_str or 'image' in label_str:  # Sometimes figures might be in body
                        element_type = 'figure_caption'

                elements.append(DocumentElement(
                    element_id=f"body_{element_counter}",
                    content=self.clean_text(
                        content_text) if element_type == 'text' else self.clean_markdown(content_text),
                    element_type=element_type,
                    page_number=element_page,
                    metadata={"docling_label": str(
                        item.label) if hasattr(item, 'label') else None}
                ))
                element_counter += 1

        # Process tables explicitly
        if hasattr(docling_document, 'tables') and docling_document.tables:
            for table_idx, table_item in enumerate(docling_document.tables):
                table_markdown = ""
                try:
                    # Docling tables often have an export_to_markdown method
                    if hasattr(table_item, 'export_to_markdown'):
                        table_markdown = table_item.export_to_markdown()
                    else:
                        logger.warning(
                            f"Table at index {table_idx} does not have export_to_markdown. Falling back to text representation.")
                        table_markdown = self._extract_content_text(table_item)
                except Exception as e:
                    logger.warning(
                        f"Could not convert table {table_idx} to markdown: {e}. Using plain text.")
                    table_markdown = self._extract_content_text(table_item)

                element_page = self._get_element_page_number(table_item) or 1
                elements.append(DocumentElement(
                    element_id=f"table_{table_idx}",
                    # Clean markdown specifically for tables
                    content=self.clean_markdown(table_markdown),
                    element_type='table',
                    page_number=element_page,
                    metadata={"table_index": table_idx}
                ))
                element_counter += 1

        # Process figures explicitly
        if hasattr(docling_document, 'figures') and docling_document.figures:
            for fig_idx, fig_item in enumerate(docling_document.figures):
                caption_text = ""
                if hasattr(fig_item, 'caption') and fig_item.caption:
                    caption_text = self._extract_content_text(fig_item.caption)
                # Sometimes text might be the caption
                elif hasattr(fig_item, 'text') and fig_item.text:
                    caption_text = fig_item.text

                if caption_text:
                    element_page = self._get_element_page_number(fig_item) or 1
                    elements.append(DocumentElement(
                        element_id=f"figure_{fig_idx}_caption",
                        # Clean text for captions
                        content=self.clean_text(caption_text),
                        element_type='figure_caption',
                        page_number=element_page,
                        metadata={"figure_index": fig_idx}
                    ))
                    element_counter += 1
                else:
                    logger.debug(
                        f"Figure {fig_idx} found but no discernible caption.")

        logger.info(
            f"Processed Docling document into {len(elements)} structured elements.")
        return elements

    def extract_structured_elements_from_pdf(self, pdf_path: str) -> List[DocumentElement]:
        """
        Extracts structured content elements (text, tables, figures with captions)
        from a PDF document, including page-level metadata.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            List[DocumentElement]: A list of structured document elements.

        Raises:
            Exception: If an error occurs during PDF processing with Docling.
        """
        try:
            result = self.converter.convert(pdf_path)
            structured_elements = self._process_document_elements(
                result.document)
            return structured_elements
        except Exception as e:
            logger.error(
                f"Error extracting structured elements from PDF '{pdf_path}': {e}", exc_info=True)
            raise Exception(
                f"Error processing PDF for structured elements: {str(e)}")

    def _convert_to_markdown_fallback(self, document) -> str:
        """
        A heuristic fallback method to convert document content to a markdown-like format
        if direct markdown export is not available from Docling's document object.
        Attempts to identify headings based on simple patterns.

        Args:
            document: The Docling document object.

        Returns:
            str: A string with basic markdown-like formatting for headings.
        """
        try:
            # Ensure we get some text content to process
            text = self._extract_content_text(document)

            lines = text.split('\n')
            formatted_lines = []

            for line in lines:
                line = line.strip()
                if line:
                    # Improved heuristic for identifying potential headings:
                    # - Line is all uppercase AND reasonably short
                    # - Line is Title Case AND reasonably short
                    # - Line matches common German or English section patterns (e.g., "Kapitel 1", "Section 2")
                    if (len(line) > 0 and len(line) < 100 and (line.isupper() or line.istitle())) or \
                       re.match(r'^(Kapitel|Abschnitt|Empfehlung|Leitlinie|Appendix|Section|Chapter)\s+[\d\.]+', line, re.IGNORECASE):
                        # Add as H2 markdown
                        formatted_lines.append(f"## {line}")
                    else:
                        formatted_lines.append(line)
                else:
                    # Preserve empty lines for paragraph breaks
                    formatted_lines.append("")

            return '\n'.join(formatted_lines)

        except Exception as e:
            logger.warning(
                f"Fallback markdown conversion failed for document. Returning plain string. Error: {e}")
            return str(document)  # Return plain string as last resort

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes plain text.
        Removes excessive whitespace and unwanted characters while preserving
        common punctuation, mathematical symbols, and specific German characters.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned and normalized text.
        """
        # Replace multiple whitespace characters (including newlines, tabs) with a single space
        text = re.sub(r'\s+', ' ', text)
        # FIX: Revert from \p{L}\p{N} which is not supported by standard `re` module.
        # Use \w (word characters) which is Unicode-aware with re.UNICODE flag,
        # and explicitly add other desired symbols.
        text = re.sub(r'[^\w\s.,;:()\-%+<>°μ§\[\]/]',
                      '', text, flags=re.UNICODE)
        return text.strip()

    def clean_markdown(self, markdown_text: str) -> str:
        """
        Cleans markdown text while attempting to preserve its structural integrity.
        Removes excessive blank lines and malformed markdown links/references.

        Args:
            markdown_text (str): The input markdown text to clean.

        Returns:
            str: The cleaned markdown text.
        """
        # Replace three or more consecutive blank lines with two blank lines to reduce excessive spacing
        markdown_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_text)
        # Clean up malformed markdown links like [Text]() by replacing them with just the 'Text'
        markdown_text = re.sub(r'\[([^\]]+)\]\(\s*\)', r'\1', markdown_text)
        # Remove markdown footnote references (e.g., [^1], [^abc]) as they are often not useful in RAG chunks
        markdown_text = re.sub(r'\[\^(\w+)\]', '', markdown_text)

        return markdown_text.strip()

    def heading_chunk_elements(self, elements: List[DocumentElement], with_metadata: bool = True) -> List[Union[str, dict]]:
        """
        Chunks a list of DocumentElement objects based on detected headings.
        This method respects the element types, treating tables and figure captions
        as atomic units or embedding them appropriately within text chunks.

        Args:
            elements (List[DocumentElement]): A list of structured document elements.
            with_metadata (bool): If True, returns a list of dictionaries with 'text' and 'metadata'.

        Returns:
            List[Union[str, dict]]: A list of text chunks, optionally with rich metadata.
        """
        chunks: List[Dict[str, Any]] = []
        current_chunk_content: List[str] = []
        current_chunk_metadata: Dict[str, Any] = {}

        # Initialize metadata for the first chunk
        if elements:
            first_element = elements[0]
            current_chunk_metadata = {
                "page_start": first_element.page_number,
                "page_end": first_element.page_number,
                "chunk_type": "heading_based",
                "original_element_ids": []
            }

        for i, element in enumerate(elements):
            is_heading = element.element_type == 'heading'
            is_new_section_candidate = is_heading

            # If we encounter a heading or a large element (like a table) that should start a new chunk
            # or if the current chunk is too large.
            if (is_new_section_candidate and current_chunk_content and len("".join(current_chunk_content)) > self.chunk_size * 0.1) or \
               len("".join(current_chunk_content)) + len(element.content) > self.chunk_size:
                # Finalize current chunk
                if current_chunk_content:
                    final_content = " ".join(current_chunk_content).strip()
                    if len(final_content) > 50:  # Only add meaningful chunks
                        chunks.append({
                            "text": final_content,
                            "metadata": {
                                **current_chunk_metadata,
                                "length": len(final_content),
                                # Try to get heading
                                "heading": self._extract_heading_from_elements(current_chunk_content)
                            }
                        })

                # Start new chunk
                current_chunk_content = []
                current_chunk_metadata = {
                    "page_start": element.page_number,
                    "page_end": element.page_number,
                    "chunk_type": "heading_based",
                    "original_element_ids": []
                }
                # Add overlap from previous chunk if it existed
                if chunks and self.chunk_overlap > 0:
                    last_chunk_text = chunks[-1]["text"]
                    overlap = last_chunk_text[-self.chunk_overlap:]
                    current_chunk_content.append(overlap)

            # Add current element to current chunk
            current_chunk_content.append(element.content)
            current_chunk_metadata["original_element_ids"].append(
                element.element_id)
            # Update end page
            current_chunk_metadata["page_end"] = element.page_number

        # Add the last chunk
        if current_chunk_content:
            final_content = " ".join(current_chunk_content).strip()
            if len(final_content) > 50:
                chunks.append({
                    "text": final_content,
                    "metadata": {
                        **current_chunk_metadata,
                        "length": len(final_content),
                        "heading": self._extract_heading_from_elements(current_chunk_content)
                    }
                })

        # Post-process chunks to add unique index and potentially simplify if metadata is not requested
        final_output_chunks = []
        for idx, chunk_data in enumerate(chunks):
            chunk_data["metadata"]["index"] = idx
            if with_metadata:
                final_output_chunks.append(chunk_data)
            else:
                final_output_chunks.append(chunk_data["text"])

        return final_output_chunks

    def _extract_heading_from_elements(self, content_parts: List[str]) -> str:
        """
        Attempts to extract a heading from a list of content parts that form a chunk.
        Looks for markdown headers or common heading patterns in the first few lines.
        """
        full_content = "\n".join(content_parts)
        lines = full_content.split('\n')
        for line in lines[:5]:  # Check first few lines
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                return stripped_line.lstrip('# ').strip()
            heading_match = re.match(
                r'^(Kapitel|Abschnitt|Empfehlung(?:\s+Nr\.)?\s*\d+(?:\.\d+)*|Leitlinie\s+\d+|Anhang\s+[A-Z])',
                stripped_line,
                re.IGNORECASE
            )
            if heading_match:
                return heading_match.group(0).strip()
            # Simple heuristic for title-like lines
            if len(stripped_line) > 0 and len(stripped_line) < 100 and stripped_line.istitle():
                return stripped_line
        return ""  # No prominent heading found

    def chunk_text(self, text_or_elements: Union[str, List[DocumentElement]], with_metadata: bool = True, use_headings: bool = True) -> List[Union[str, dict]]:
        """
        Chunks the input text or list of DocumentElement objects.
        If a string is provided, it processes it into DocumentElements first (basic, without tables/figures).
        If DocumentElements are provided, it uses them directly.
        Prioritizes heading-based chunking if enabled, otherwise falls back to recursive splitting.

        Args:
            text_or_elements (Union[str, List[DocumentElement]]): The full document text or a list of DocumentElement objects.
            with_metadata (bool): If True, returns chunks as dictionaries with metadata.
            use_headings (bool): If True, attempts to chunk by headings first.

        Returns:
            List[Union[str, dict]]: A list of text chunks, optionally with metadata.
        """
        elements_to_chunk: List[DocumentElement] = []
        if isinstance(text_or_elements, str):
            # If a string is passed, convert it to a basic list of DocumentElements
            # This path won't have explicit table/figure elements unless they were converted to text already
            lines = text_or_elements.split('\n')
            for i, line in enumerate(lines):
                if line.strip():  # Only add non-empty lines as elements
                    elements_to_chunk.append(DocumentElement(
                        element_id=f"line_{i}",
                        content=line,
                        element_type='text',
                        page_number=0  # Page number unknown if only raw text is provided
                    ))
            logger.info(
                "Chunking from raw text input. No rich document element structure available.")
        # FIX: Corrected type check for the list of DocumentElement objects
        elif isinstance(text_or_elements, list) and all(isinstance(e, DocumentElement) for e in text_or_elements):
            elements_to_chunk = text_or_elements
            logger.info(
                f"Chunking from {len(elements_to_chunk)} structured document elements.")
        else:
            raise TypeError(
                "Input must be a string or a list of DocumentElement objects.")

        # Attempt heading-based chunking first if enabled
        if use_headings and elements_to_chunk:
            # Use the new heading_chunk_elements method
            chunks_from_headings = self.heading_chunk_elements(
                elements_to_chunk, with_metadata=True)
            # Check if heading chunking produced meaningful splits
            # Meaningful if there's more than one chunk, or if the single chunk isn't just the whole document content
            if len(chunks_from_headings) > 1 or \
               (len(chunks_from_headings) == 1 and len(elements_to_chunk) > 1 and len(chunks_from_headings[0]['text']) < sum(len(e.content) for e in elements_to_chunk) * 0.9):
                logger.info(
                    f"Chunked document into {len(chunks_from_headings)} chunks using heading-based strategy.")
                return chunks_from_headings if with_metadata else [c['text'] for c in chunks_from_headings]

        # Fallback to recursive character splitting if heading chunking is not used or ineffective
        logger.info(
            "Falling back to recursive character text splitting (or headings were not effective).")
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            # Combine all content into a single string for recursive splitter
            # For this fallback, we lose some element-level metadata for inner chunks,
            # but page_number can still be added to the metadata of larger chunks.
            full_text_content = "\n\n".join(
                [e.content for e in elements_to_chunk])

            # Define separators in order of preference (larger semantic breaks first)
            separators = [
                "\n\n\n",   # Very large breaks, e.g., between major sections
                "\n## ",    # Markdown H2 (often a good natural break)
                "\n### ",   # Markdown H3
                "\n#### ",  # Markdown H4
                "\n\n",     # Paragraph breaks (two newlines)
                "\n",       # Line breaks (single newline)
                ". ",       # Sentence breaks (dot followed by space)
                " ",        # Word breaks (space)
                # Character breaks (last resort, if no other split is possible)
                ""
            ]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,  # Specifies using character length for chunk_size
                is_separator_regex=False  # Treat separators as literal strings, not regex patterns
            )

            raw_chunks = splitter.split_text(full_text_content)
            # Filter out chunks that are empty or too short after stripping whitespace
            raw_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 50]

            if with_metadata:
                # For recursive chunks, metadata might be less precise.
                # We can try to infer page numbers or general document info.
                # Here, we'll assign page 0 or the first page found as a placeholder
                # This could be improved by mapping char ranges back to original elements.
                default_page = elements_to_chunk[0].page_number if elements_to_chunk else 0
                return [
                    {
                        "text": chunk,
                        "metadata": {
                            "index": i,
                            "length": len(chunk),
                            "chunk_type": "recursive_character",
                            # This is a simplification; a more complex mapping is needed for precise page numbers for recursive splits
                            "page_start": default_page,
                            "page_end": default_page
                        }
                    }
                    for i, chunk in enumerate(raw_chunks)
                ]
            else:
                return raw_chunks

        except ImportError:
            logger.warning(
                "langchain.text_splitter not found. Falling back to simple chunking. Please install `langchain` (`pip install langchain`) for more advanced splitting features.")
            raw_chunks = self._simple_chunk(text_or_elements if isinstance(
                text_or_elements, str) else "\n\n".join([e.content for e in elements_to_chunk]))
            if with_metadata:
                default_page = elements_to_chunk[0].page_number if elements_to_chunk else 0
                return [
                    {
                        "text": chunk,
                        "metadata": {
                            "index": i,
                            "length": len(chunk),
                            "chunk_type": "simple_fallback",
                            "page_start": default_page,  # Same simplification as above
                            "page_end": default_page
                        }
                    }
                    for i, chunk in enumerate(raw_chunks)
                ]
            else:
                return raw_chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """
        A simple fallback chunking method that splits text by words and applies
        fixed-size chunking with overlap. Used if `langchain` is not available.

        Args:
            text (str): The input text to chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        words = text.split()
        chunks = []
        current_chunk_words = []
        current_length = 0

        for word in words:
            word_len_with_space = len(word) + 1  # Account for space after word

            # Check if adding the current word exceeds the chunk size
            if current_length + word_len_with_space > self.chunk_size and current_chunk_words:
                chunks.append(' '.join(current_chunk_words).strip())

                # Handle overlap by taking the last `self.chunk_overlap` characters
                # from the just-completed chunk, then splitting those into words for the new chunk start.
                overlap_text = ' '.join(
                    current_chunk_words)[-self.chunk_overlap:].strip()
                current_chunk_words = overlap_text.split() if overlap_text else [
                ]  # Ensure it's not empty list
                current_length = sum(
                    len(w) + 1 for w in current_chunk_words) if current_chunk_words else 0

                # Add the current word to the new chunk
                current_chunk_words.append(word)
                current_length += word_len_with_space
            else:
                current_chunk_words.append(word)
                current_length += word_len_with_space

        # Add the last remaining chunk if it's not empty
        if current_chunk_words:
            chunks.append(' '.join(current_chunk_words).strip())

        # Filter out any chunks that are too short after the process
        return [c for c in chunks if len(c) > 50]

    def extract_document_structure(self, pdf_path: str) -> dict:
        """
        Extracts high-level document structure information using Docling,
        including title, page count, and counts of tables/figures.
        Also attempts to extract detailed heading information.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            dict: A dictionary containing structural information about the PDF.
                  Includes fallbacks for missing data and an error field if extraction fails.
        """
        try:
            result = self.converter.convert(pdf_path)
            document = result.document

            # Extract title with fallbacks
            title = getattr(document, 'title', getattr(
                document, 'name', "Unknown Document"))
            # Extract page count with fallbacks
            page_count = len(getattr(document, 'pages', [])
                             ) or getattr(document, 'page_count', 0)

            tables_count = 0
            figures_count = 0

            # Iterate through the document's body items to count tables and figures
            if hasattr(document, 'body') and document.body:
                for item in document.body:
                    if hasattr(item, 'label'):
                        label_str = str(item.label).lower()
                        if 'table' in label_str:
                            tables_count += 1
                        elif 'figure' in label_str or 'image' in label_str:
                            figures_count += 1
            # Also check dedicated tables/figures collections
            if hasattr(document, 'tables'):
                tables_count += len(document.tables)
            if hasattr(document, 'figures'):
                figures_count += len(document.figures)

            structure_info = {
                "title": title,
                "page_count": page_count,
                "tables_count": tables_count,
                "figures_count": figures_count,
                # Detailed headings
                "headings": self._extract_headings_from_document(document)
            }
            logger.info(
                f"Successfully extracted document structure for '{pdf_path}'.")
            return structure_info

        except Exception as e:
            logger.error(
                f"Error extracting document structure from '{pdf_path}': {e}", exc_info=True)
            return {
                "title": "Extraction Failed",
                "page_count": 0,
                "tables_count": 0,
                "figures_count": 0,
                "headings": [],
                "error": str(e)
            }

    def _extract_headings_from_document(self, document) -> List[dict]:
        """
        Extracts detailed heading information (text, level, page) from the
        Docling document's internal structure.

        Args:
            document: The Docling document object.

        Returns:
            List[dict]: A list of dictionaries, each representing a heading with its details.
        """
        headings = []
        try:
            if hasattr(document, 'body') and document.body:
                for i, item in enumerate(document.body):
                    # Check if the item's label indicates it's a heading or title
                    if hasattr(item, 'label') and ('heading' in str(item.label).lower() or 'title' in str(item.label).lower()):
                        text = self._extract_content_text(item)

                        # Attempt to get the page number
                        page_info = self._get_element_page_number(item)

                        headings.append({
                            # Limit text length and strip whitespace
                            'text': text.strip()[:200],
                            # Use the Docling label directly for level info (e.g., 'heading_1')
                            'level': str(item.label),
                            'page': page_info,
                            'index': i
                        })
        except Exception as e:
            logger.warning(
                f"Could not extract detailed headings from document body: {e}")
        return headings


# Optional CLI for testing purposes
if __name__ == "__main__":
    import sys
    from pprint import pprint
    import os

    # Set up a dedicated logger for the CLI part to distinguish its output
    cli_logger = logging.getLogger("PDFProcessorCLI")
    cli_logger.setLevel(logging.INFO)
    cli_handler = logging.StreamHandler(sys.stdout)
    cli_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    cli_logger.addHandler(cli_handler)
    cli_logger.propagate = False  # Prevent messages from bubbling up to root logger

    if len(sys.argv) < 2:
        cli_logger.info(
            "Usage: python pdf_processor.py <path_to_pdf> [--plain-text] [--structure]")
        cli_logger.info(
            "  --plain-text    (Not applicable with new structured output, but kept for compatibility.)")
        cli_logger.info(
            "  --structure     Show document structure information")
        cli_logger.info(
            "  --show-elements Show extracted structured elements before chunking")
        sys.exit(1)

    pdf_path = sys.argv[1]
    # use_markdown is less relevant now as output is structured, but kept for initial parsing
    use_markdown = "--plain-text" not in sys.argv
    show_structure = "--structure" in sys.argv
    show_elements = "--show-elements" in sys.argv

    if not os.path.exists(pdf_path):
        cli_logger.error(f"Error: PDF file not found at '{pdf_path}'")
        sys.exit(1)

    processor = PDFProcessor()

    try:
        if show_structure:
            cli_logger.info("=== Document Structure Extraction ===")
            structure = processor.extract_document_structure(pdf_path)
            pprint(structure)
            cli_logger.info("\n" + "="*50 + "\n")

        cli_logger.info(f"Extracting structured elements from '{pdf_path}'...")
        # New method to get structured elements
        structured_elements = processor.extract_structured_elements_from_pdf(
            pdf_path)
        cli_logger.info(
            f"Extracted {len(structured_elements)} structured elements.")

        if show_elements:
            cli_logger.info(
                "\n=== Extracted Structured Elements (Preview) ===")
            # Show first 5 elements
            for i, element in enumerate(structured_elements[:5]):
                cli_logger.info(f"--- Element {i+1} ---")
                cli_logger.info(f"ID: {element.element_id}")
                cli_logger.info(f"Type: {element.element_type}")
                cli_logger.info(f"Page: {element.page_number}")
                cli_logger.info(f"Metadata: {element.metadata}")
                cli_logger.info(f"Content Preview: {element.content[:300]}...")
            if len(structured_elements) > 5:
                cli_logger.info(
                    f"\n... and {len(structured_elements) - 5} more elements.")
            cli_logger.info("\n" + "="*50 + "\n")

        cli_logger.info("Chunking extracted structured elements...")
        # Pass the list of DocumentElement objects to chunk_text
        chunks_with_metadata = processor.chunk_text(
            structured_elements, with_metadata=True)
        cli_logger.info(
            f"Extracted {len(chunks_with_metadata)} chunks from PDF elements.")
        cli_logger.info(
            "\nShowing first 3 chunks (or fewer if less than 3 exist):")

        for i, chunk_data in enumerate(chunks_with_metadata[:3]):
            cli_logger.info(f"\n--- Chunk {i+1} ---")
            if isinstance(chunk_data, dict):
                cli_logger.info(f"Metadata: {chunk_data['metadata']}")
                text_preview = str(chunk_data['text'])
                cli_logger.info(f"Text preview: {text_preview[:400]}..." if len(
                    text_preview) > 400 else text_preview)
            # Fallback for raw string chunks (if with_metadata was False)
            else:
                text_preview = str(chunk_data)
                cli_logger.info(f"Text preview: {text_preview[:400]}..." if len(
                    text_preview) > 400 else text_preview)

        if len(chunks_with_metadata) > 3:
            cli_logger.info(
                f"\n...and {len(chunks_with_metadata) - 3} more chunks.")

    except Exception as e:
        cli_logger.exception(
            f"An unhandled error occurred during PDF processing: {e}")
        sys.exit(1)
