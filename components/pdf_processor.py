import re
from typing import List, Union
import pdfplumber  # Better PDF parsing
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber, preserving page markers."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {i + 1} ---\n{page_text}"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return self.clean_text(text)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove unwanted characters, but keep common medical symbols
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\%\+\=\<\>\/]', '', text)
        return text.strip()

    def chunk_text(self, text: str, with_metadata: bool = True) -> List[Union[str, dict]]:
        """
        Split text into chunks.
        If with_metadata is True, return chunks with metadata (index, length).
        """
        chunks = self.text_splitter.split_text(text)
        filtered_chunks = [
            chunk for chunk in chunks if len(chunk.strip()) > 50]

        if with_metadata:
            return [
                {
                    "text": chunk,
                    "metadata": {
                        "index": i,
                        "length": len(chunk)
                    }
                }
                for i, chunk in enumerate(filtered_chunks)
            ]
        else:
            return filtered_chunks


# Optional CLI for testing
if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        sys.exit(1)

    processor = PDFProcessor()
    extracted_text = processor.extract_text_from_pdf(sys.argv[1])
    chunks_with_metadata = processor.chunk_text(
        extracted_text, with_metadata=True)
    chunks = [chunk["text"] if isinstance(
        chunk, dict) else chunk for chunk in chunks_with_metadata]

    print(f"\nExtracted {len(chunks)} chunks.\nShowing first 3:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        pprint(chunk)
