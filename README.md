# RAG Quiz

A Python application for generating quizzes from PDF documents using Retrieval-Augmented Generation (RAG) techniques. The app processes PDFs, generates embeddings, stores them in a vector database, and uses a language model to create quiz questions.

## Features

- **PDF Processing:** Extracts text from uploaded PDF files.
- **Embeddings:** Generates vector embeddings for document chunks.
- **Vector Store:** Stores and retrieves embeddings using ChromaDB.
- **Quiz Generation:** Uses a local LLM (Meta-Llama-3.1-8B-Instruct) to generate quiz questions.
- **Modular Components:** Clean separation of concerns for easy maintenance and extension.

## Prerequisites

- Python 3.12 or later
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/rag_quiz.git
   cd rag_quiz
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the LLM model:**
   - Place your `.gguf` model file in the `models/` directory.

## Usage

1. **Run the application:**

   ```bash
   python app.py
   ```

   or

   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF:**

   - Place your PDF in `data/uploads/` or use the app's upload interface (if available).

3. **Generate a quiz:**
   - The app will process the PDF and generate quiz questions using the LLM.

## Project Structure

```
rag_quiz/
│
├── app.py # Main application entry point
├── components/ # Core logic modules
│ ├── embeddings.py # Embedding generation logic
│ ├── llm_handler.py # LLM interaction logic
│ ├── pdf_processor.py # PDF text extraction
│ ├── quiz_generator.py # Quiz question generation
│ └── vector_store.py # Vector database interface
│
├── chroma_db/ # ChromaDB vector store files (auto-generated)
│
├── data/
│ └── uploads/ # Uploaded PDF files
│
├── models/
│ └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf # Local LLM model file
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Components

- **PDF Processor**: Handles PDF document loading and text extraction
- **Vector Store**: Manages document embeddings and similarity search
- **Quiz Generator**: Generates quiz questions using the FLAN-T5 model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.12+
- [ChromaDB](https://www.trychroma.com/)
- [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama) (or compatible .gguf model)
- See `requirements.txt` for Python dependencies.

## Acknowledgements

- [ChromaDB](https://www.trychroma.com/)
- [Meta Llama Models](https://huggingface.co/meta-llama)
