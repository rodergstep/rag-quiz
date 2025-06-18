import streamlit as st
import os
import tempfile
from components.pdf_processor import PDFProcessor
from components.embeddings import EmbeddingHandler
from components.vector_store import VectorStore
from components.llm_handler import LocalLLMHandler
from components.quiz_generator import QuizGenerator
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Quiz Generator",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pdf_processed = False
    st.session_state.components_loaded = False


@st.cache_resource
def load_components():
    """Load all components (cached to avoid reloading)"""
    try:
        # Initialize components
        embedding_handler = EmbeddingHandler()
        vector_store = VectorStore()
        vector_store.create_collection()

        # Load LLM (adjust path as needed)
        llm_path = "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(llm_path):
            st.error(f"LLM model not found at {llm_path}")
            return None

        llm_handler = LocalLLMHandler(llm_path)
        quiz_generator = QuizGenerator(
            llm_handler, vector_store, embedding_handler)

        return {
            'pdf_processor': PDFProcessor(),
            'embedding_handler': embedding_handler,
            'vector_store': vector_store,
            'llm_handler': llm_handler,
            'quiz_generator': quiz_generator
        }
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        return None


def main():
    st.title("🏥 Medical Quiz Generator")
    st.markdown(
        "Upload a medical guideline PDF and generate quiz questions on any topic!")

    # Load components
    if not st.session_state.components_loaded:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            components = load_components()
            if components:
                st.session_state.components = components
                st.session_state.components_loaded = True
                st.success("✅ AI models loaded successfully!")
            else:
                st.error("❌ Failed to load AI models")
                return

    components = st.session_state.components

    # Sidebar for PDF upload
    st.sidebar.header("📄 PDF Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload your medical guideline PDF (up to 600 pages)"
    )

    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF... This may take several minutes for large files."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Process PDF
                    pdf_processor = components['pdf_processor']
                    text = pdf_processor.extract_text_from_pdf(tmp_path)
                    chunks = pdf_processor.chunk_text(text)

                    st.sidebar.write(f"📊 Extracted {len(chunks)} text chunks")

                    # Generate embeddings
                    embeddings = components['embedding_handler'].encode_texts(
                        chunks)

                    # Store in vector database
                    components['vector_store'].add_documents(
                        texts=chunks,
                        embeddings=embeddings.tolist()
                    )

                    # Clean up
                    os.unlink(tmp_path)

                    st.session_state.pdf_processed = True
                    st.sidebar.success("✅ PDF processed successfully!")

                except Exception as e:
                    st.sidebar.error(f"❌ Error processing PDF: {str(e)}")
                    return

    # Main interface
    if st.session_state.pdf_processed:
        st.header("🎯 Generate Quiz Questions")

        col1, col2 = st.columns([2, 1])

        with col1:
            topic = st.text_input(
                "Enter topic for quiz generation:",
                placeholder="e.g., asthma treatment, diabetes management, hypertension guidelines"
            )

        with col2:
            num_questions = st.number_input(
                "Number of questions:",
                min_value=1,
                max_value=20,
                value=5
            )

        if st.button("🚀 Generate Quiz", type="primary"):
            if topic:
                with st.spinner(f"Generating {num_questions} quiz questions about '{topic}'..."):
                    try:
                        # Add debug mode
                        debug_mode = st.sidebar.checkbox(
                            "🐛 Debug Mode", help="Show generation details")

                        quiz_df = components['quiz_generator'].generate_quiz(
                            topic=topic,
                            num_questions=num_questions
                        )

                        if not quiz_df.empty:
                            st.success(
                                f"✅ Generated {len(quiz_df)} quiz questions!")

                            # Debug information
                            if debug_mode:
                                with st.expander("🔍 Debug Information"):
                                    st.write(
                                        f"**Requested questions:** {num_questions}")
                                    st.write(
                                        f"**Generated questions:** {len(quiz_df)}")
                                    st.write(f"**Topic:** {topic}")

                                    # Show raw generation (if you want to add this feature)
                                    if st.button("Show Raw LLM Output"):
                                        # You can store the raw output in session state during generation
                                        st.text(
                                            "Enable raw output storage in quiz_generator.py")

                            # Display questions
                            st.header("📝 Generated Quiz Questions")

                            for idx, row in quiz_df.iterrows():
                                with st.expander(f"Question {idx + 1}: {row['Question'][:50]}..."):
                                    st.write(
                                        f"**Question:** {row['Question']}")
                                    st.write(f"**A)** {row['Option A']}")
                                    st.write(f"**B)** {row['Option B']}")
                                    st.write(f"**C)** {row['Option C']}")
                                    st.write(f"**D)** {row['Option D']}")
                                    st.write(
                                        f"**Correct Answer:** {row['Correct Answer']}")

                            # Download CSV
                            csv_data = quiz_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Quiz as CSV",
                                data=csv_data,
                                file_name=f"quiz_{topic.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(
                                "⚠️ No quiz questions were generated. Try a different topic.")

                    except Exception as e:
                        st.error(f"❌ Error generating quiz: {str(e)}")

                        # Enhanced error information
                        with st.expander("🔧 Troubleshooting Information"):
                            st.write("**Common solutions:**")
                            st.write(
                                "1. Try a more specific topic (e.g., 'asthma treatment guidelines' instead of 'asthma')")
                            st.write(
                                "2. Reduce the number of questions to 5-10 first")
                            st.write(
                                "3. Check if your PDF contains relevant content about the topic")
                            st.write(
                                "4. Try restarting the app if memory issues occur")
                            st.write(f"**Full error:** {str(e)}")
            else:
                st.warning("⚠️ Please enter a topic for quiz generation.")

    else:
        st.info("👆 Please upload a PDF file to get started.")

        # Show system info
        with st.expander("ℹ️ System Information"):
            collection_count = components['vector_store'].get_collection_info(
            ) if st.session_state.components_loaded else 0
            st.write(f"📊 Documents in database: {collection_count}")
            st.write("🧠 Embedding model: all-MiniLM-L6-v2")
            st.write("🤖 LLM model: llama-2-7b-chat (Quantized)")


if __name__ == "__main__":
    main()
