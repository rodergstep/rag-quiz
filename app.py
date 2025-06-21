import streamlit as st
import os
import tempfile
import hashlib  # Import hashlib for PDF content hashing
from components.pdf_processor import PDFProcessor, DocumentElement
from components.embeddings import EmbeddingHandler
from components.vector_store import VectorStore
from components.llm_handler import LocalLLMHandler
from components.quiz_generator import QuizGenerator
from components.learning_objectives_generator import LearningObjectivesGenerator
import pandas as pd
import numpy as np  # Import numpy for converting embeddings back to array

# Page configuration for Streamlit app
st.set_page_config(
    page_title="Medical Quiz Generator",
    page_icon="🏥",
    layout="wide"  # Use wide layout for more screen space
)

# Initialize session state variables to manage application flow
if 'initialized' not in st.session_state:
    st.session_state.initialized = False  # Flag for initial setup completion
    # Flag to track if PDF has been processed for current session
    st.session_state.pdf_processed = False
    # Flag to ensure AI models are loaded once
    st.session_state.components_loaded = False
    st.session_state.components = {}  # Dictionary to store initialized components
    # Store the ID of the currently loaded PDF
    st.session_state.current_pdf_id = None


@st.cache_resource
def load_components():
    """
    Loads and initializes all core components of the application.
    This function is cached by Streamlit to prevent re-running on every rerun,
    saving time and resources by loading AI models only once.
    """
    try:
        # Initialize EmbeddingHandler (for text-to-vector conversion)
        embedding_handler = EmbeddingHandler()

        # Initialize VectorStore (for storing and retrieving document embeddings)
        vector_store = VectorStore()
        vector_store.create_collection()  # Ensure the vector collection exists

        # Load LLM (Large Language Model) from a local path
        # IMPORTANT: Verify the path to your LLM model
        llm_path = "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(llm_path):
            st.error(
                f"❌ LLM model not found at {llm_path}. Please ensure the model file is in the correct location.")
            return None

        # Handles interactions with the local LLM
        llm_handler = LocalLLMHandler(llm_path)

        # Initialize QuizGenerator, which orchestrates LLM, vector store, and embeddings
        quiz_generator = QuizGenerator(
            llm_handler, vector_store, embedding_handler)

        # NEW: Initialize LearningObjectivesGenerator
        learning_objectives_generator = LearningObjectivesGenerator(
            llm_handler, vector_store, embedding_handler)

        # Return all initialized components in a dictionary
        return {
            'pdf_processor': PDFProcessor(),  # Handles PDF parsing and chunking
            'embedding_handler': embedding_handler,
            'vector_store': vector_store,
            'llm_handler': llm_handler,
            'quiz_generator': quiz_generator,
            'learning_objectives_generator': learning_objectives_generator
        }
    except Exception as e:
        st.error(f"❌ Error loading core AI components: {str(e)}")
        # Provide more detailed error information for debugging
        st.exception(e)
        return None


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("🏥 Medical Quiz and Learning Objective Generator")
    st.markdown(
        "Upload a medical guideline PDF and generate quiz questions or structured learning objectives!")

    # Load components if not already loaded
    if not st.session_state.components_loaded:
        with st.spinner("Loading AI models... This may take a few minutes on first run (especially for the LLM)."):
            components = load_components()
            if components:
                st.session_state.components = components
                st.session_state.components_loaded = True
                st.success("✅ AI models loaded successfully!")
            else:
                st.error(
                    "❌ Failed to load AI models. Please check the console for details.")
                return  # Stop execution if components fail to load

    # Access loaded components from session state
    components = st.session_state.components

    # Sidebar for PDF upload functionality
    st.sidebar.header("📄 PDF Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="pdf_uploader",  # Added a key to prevent re-uploading on minor widget changes
        help="Upload your medical guideline PDF (up to 600 pages recommended for faster processing)"
    )

    # Logic to handle PDF upload and caching
    if uploaded_file is not None:
        # Calculate a unique ID for the PDF based on its content hash
        pdf_content = uploaded_file.getvalue()
        current_pdf_id = hashlib.sha256(pdf_content).hexdigest()
        st.sidebar.info(
            f"Calculated PDF Content ID (SHA256): `{current_pdf_id}`")

        # Determine if the PDF is different from the one currently processed in session
        # or if it's the very first time this PDF is being considered in this session.
        # This prevents unnecessary re-processing if the user simply interacts with other widgets.
        # Crucially, we *always* check the DB if `pdf_processed` is False.
        if st.session_state.current_pdf_id != current_pdf_id:
            # If it's a new PDF or the session just started with a different PDF
            st.session_state.pdf_processed = False  # Reset flag for current session
            st.session_state.current_pdf_id = current_pdf_id  # Update the tracked PDF ID

        # --- Core caching logic: Check DB first before processing ---
        # This block will run if `pdf_processed` is False (meaning we need to check/process)
        if not st.session_state.pdf_processed:
            with st.spinner("Checking database for this PDF..."):
                vector_store = components['vector_store']

                # Try to retrieve documents associated with this PDF ID from the persistent DB
                existing_documents = vector_store.get_documents_by_pdf_id(
                    current_pdf_id)

                # Debugging info:
                st.sidebar.info(
                    f"Database check for PDF ID `{current_pdf_id}` returned {len(existing_documents)} existing documents.")

            if existing_documents:
                # If documents are found in the database for this pdf_id
                st.sidebar.success(
                    f"✅ PDF '{uploaded_file.name}' data found in database! Loaded {len(existing_documents)} pre-processed chunks.")
                # Mark as processed for the current session
                st.session_state.pdf_processed = True
            else:
                # If documents are NOT found, proceed with full processing
                st.sidebar.info(
                    f"PDF '{uploaded_file.name}' not found in database. Processing new PDF (this may take a while)...")
                with st.spinner("Processing PDF... This may take several minutes for large files. Please wait."):
                    try:
                        # Create a temporary file to save the uploaded PDF content
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            # Use the raw content from uploaded_file
                            tmp_file.write(pdf_content)
                            tmp_path = tmp_file.name

                        # Initialize PDFProcessor
                        pdf_processor = components['pdf_processor']

                        # Use extract_structured_elements_from_pdf for rich content extraction
                        structured_elements = pdf_processor.extract_structured_elements_from_pdf(
                            tmp_path)

                        # Chunk the structured elements, retaining rich metadata
                        chunks_with_metadata = pdf_processor.chunk_text(
                            structured_elements, use_headings=True, with_metadata=True
                        )

                        # Separate chunk texts and their metadata for vector store ingestion
                        # Ensure that each item in chunks_with_metadata is a dictionary as expected
                        chunks = [
                            chunk["text"] for chunk in chunks_with_metadata if isinstance(chunk, dict)]
                        metadatas = [
                            chunk["metadata"] for chunk in chunks_with_metadata if isinstance(chunk, dict)]

                        st.sidebar.write(
                            f"📊 Extracted and chunked into {len(chunks)} text chunks (including tables/captions).")

                        # Generate embeddings for the extracted chunks
                        embeddings = components['embedding_handler'].encode_texts(
                            chunks)

                        # Store documents (chunks) and their embeddings/metadata in the vector database
                        # Pass the pdf_id along with other metadata so we can retrieve it later
                        vector_store.add_documents(
                            texts=chunks,
                            embeddings=embeddings.tolist(),  # Convert numpy array to list for storage
                            metadatas=metadatas,
                            pdf_id=current_pdf_id  # Pass the unique PDF ID
                        )

                        # Clean up the temporary PDF file
                        # Remove the temporary file from the disk
                        os.unlink(tmp_path)

                        # Mark as processed for the current session
                        st.session_state.pdf_processed = True
                        st.sidebar.success(
                            "✅ PDF processed and indexed successfully!")

                    except Exception as e:
                        st.sidebar.error(f"❌ Error processing PDF: {str(e)}")
                        # Show full traceback for debugging
                        st.sidebar.exception(e)
                        # Reset flag on error, forcing re-process on next try
                        st.session_state.pdf_processed = False
                        # Clear current_pdf_id if processing failed for robustness
                        st.session_state.current_pdf_id = None
                        return  # Stop current execution flow due to error

    # Main interface for quiz generation and learning objective generation
    if st.session_state.pdf_processed:
        st.header("🎯 Generate Content")

        tab1, tab2 = st.tabs(
            ["Generate Quiz Questions", "Generate Learning Objectives"])

        with tab1:
            st.subheader("Quiz Questions")
            col1, col2 = st.columns([2, 1])

            with col1:
                quiz_topic = st.text_input(
                    "Enter topic for quiz generation:",
                    placeholder="e.g., asthma treatment, diabetes management, hypertension guidelines",
                    help="Provide a specific topic to ensure relevant questions are generated from the PDF.",
                    key="quiz_topic_input"  # Unique key
                )

            with col2:
                num_questions = st.number_input(
                    "Number of questions:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Choose how many multiple-choice questions you'd like to generate.",
                    key="num_questions_input"  # Unique key
                )

            # Button to trigger quiz generation
            if st.button("🚀 Generate Quiz", type="primary", key="generate_quiz_button"):
                if quiz_topic:
                    with st.spinner(f"Generating {num_questions} quiz questions about '{quiz_topic}'..."):
                        try:
                            # Debug mode checkbox in sidebar for control
                            debug_mode = st.sidebar.checkbox(
                                "🐛 Show Debug Information (Quiz)", help="Display details about LLM input and retrieved context for quiz generation.", key="quiz_debug_mode")

                            quiz_df = components['quiz_generator'].generate_quiz(
                                topic=quiz_topic,
                                num_questions=num_questions
                            )

                            if not quiz_df.empty:
                                st.success(
                                    f"✅ Generated {len(quiz_df)} quiz questions!")

                                # Display debug information if enabled
                                if debug_mode:
                                    with st.expander("🔍 Debug Information (Quiz)"):
                                        st.write(
                                            f"**Requested questions:** {num_questions}")
                                        st.write(
                                            f"**Generated questions:** {len(quiz_df)}")
                                        st.write(f"**Topic:** {quiz_topic}")
                                        st.info(
                                            "Additional debug information can be added here (e.g., retrieved chunks, LLM prompts).")

                                # Display generated questions in an expandable format for readability
                                st.header("📝 Generated Quiz Questions")
                                for idx, row in quiz_df.iterrows():
                                    # Truncate question for expander title to keep it clean
                                    question_preview = row['Question']
                                    if len(question_preview) > 70:
                                        question_preview = question_preview[:67] + "..."
                                    with st.expander(f"Question {idx + 1}: {question_preview}"):
                                        st.write(
                                            f"**Question:** {row['Question']}")
                                        st.write(f"**A)** {row['Option A']}")
                                        st.write(f"**B)** {row['Option B']}")
                                        st.write(f"**C)** {row['Option C']}")
                                        st.write(f"**D)** {row['Option D']}")
                                        st.write(
                                            f"**Correct Answer:** {row['Correct Answer']}")
                                        # Optionally display page reference if metadata allows and is passed from QuizGenerator
                                        # st.write(f"**Source Page(s):** {row.get('Source Page', 'N/A')}")

                                # Provide option to download the quiz as CSV
                                csv_data = quiz_df.to_csv(
                                    index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Quiz as CSV",
                                    data=csv_data,
                                    # Clean filename
                                    file_name=f"quiz_{quiz_topic.replace(' ', '_').lower()}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("⚠️ No quiz questions were generated. This might happen if the PDF doesn't contain enough relevant information for the topic, or the LLM struggles to formulate questions. Try a different topic or reduce the number of questions.")

                        except Exception as e:
                            st.error(f"❌ Error generating quiz: {str(e)}")
                            # Provide troubleshooting tips in case of an error
                            with st.expander("🔧 Troubleshooting Information"):
                                st.write("**Common solutions:**")
                                st.write(
                                    "1. Try a more specific topic (e.g., 'asthma treatment guidelines' instead of just 'asthma').")
                                st.write(
                                    "2. Reduce the number of questions to 5-10 initially.")
                                st.write(
                                    "3. Ensure your PDF contains relevant content about the specified topic.")
                                st.write(
                                    "4. Check the LLM model path and ensure it's correctly loaded.")
                                st.write(
                                    "5. If running locally, consider if you have enough RAM/VRAM for the LLM.")
                                st.write(
                                    f"**Full error details (for developers):** {str(e)}")
                                st.exception(e)
                else:
                    st.warning("⚠️ Please enter a topic for quiz generation.")

        with tab2:
            st.subheader("Learning Objectives")
            # Removed the third column for num_context_chunks
            col_lo1, col_lo2 = st.columns([2, 1])

            # Removed lo_topic input as per request
            st.markdown(
                "Learning objectives will be generated from the **entire uploaded PDF**.")

            with col_lo1:
                lo_target_group_options = [
                    "Internal Medicine", "General Medicine", "Emergency Medicine", "Cardiology", "Rheumatology"]
                lo_target_group = st.selectbox(
                    "Select Target Group:",
                    options=lo_target_group_options,
                    index=0,  # Default to Internal Medicine
                    help="Choose the medical specialty for whom the learning objectives are intended.",
                    key="lo_target_group_select"  # Unique key
                )

            # Removed lo_num_context_chunks input as per request
            with col_lo2:
                # Placeholder to keep column structure if needed, or remove col_lo2 entirely if not
                st.write("")

            if st.button("✨ Generate Learning Objectives", type="primary", key="generate_lo_button"):
                # The lo_topic input is removed, so we don't need to check if it's empty
                # We directly use the current_pdf_id
                if st.session_state.current_pdf_id:
                    with st.spinner(f"Generating learning objectives for '{lo_target_group}' from the entire PDF..."):
                        try:
                            debug_mode_lo = st.sidebar.checkbox(
                                "🐛 Show Debug Information (LO)", help="Display details about LLM input and retrieved context for learning objectives.", key="lo_debug_mode")

                            lo_df = components['learning_objectives_generator'].generate_learning_objectives(
                                pdf_id=st.session_state.current_pdf_id,  # Pass the PDF ID
                                target_group=lo_target_group
                            )

                            if not lo_df.empty:
                                st.success(
                                    f"✅ Generated {len(lo_df)} learning objectives!")

                                if debug_mode_lo:
                                    with st.expander("🔍 Debug Information (Learning Objectives)"):
                                        st.write(
                                            f"**Target Group:** {lo_target_group}")
                                        st.write(
                                            f"**PDF ID Used:** {st.session_state.current_pdf_id}")
                                        st.write(
                                            "**Generated DataFrame Head:**")
                                        st.dataframe(lo_df.head())
                                        st.info(
                                            "The raw LLM prompt and response for learning objectives can be logged at DEBUG level.")

                                st.header("📚 Generated Learning Objectives")
                                # Displaying as a DataFrame is often best for structured output
                                st.dataframe(lo_df)

                                csv_data_lo = lo_df.to_csv(
                                    index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Learning Objectives as CSV",
                                    data=csv_data_lo,
                                    file_name=f"learning_objectives_{lo_target_group.replace(' ', '_').lower()}_{st.session_state.current_pdf_id[:6]}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning(
                                    "⚠️ No learning objectives were generated. This might be because the PDF lacks content suitable for the specified target group and strict criteria, or the LLM could not parse the response into the required JSON format.")

                        except Exception as e:
                            st.error(
                                f"❌ Error generating learning objectives: {str(e)}")
                            with st.expander("🔧 Troubleshooting Information (Learning Objectives)"):
                                st.write("**Common solutions:**")
                                st.write(
                                    "1. Ensure the PDF contains content relevant to the target group and meets the strict prompt criteria.")
                                st.write(
                                    "2. Check the LLM's logs for parsing errors (if debug mode is enabled).")
                                st.write(
                                    "3. **If you encounter context window errors, the PDF might be too large for the LLM's current `n_ctx` setting.**")
                                st.write(f"**Full error details:** {str(e)}")
                                st.exception(e)
                else:
                    st.warning(
                        "⚠️ Please upload a PDF first to generate learning objectives.")

    else:
        st.info(
            "👆 Please upload a PDF file to get started and unlock content generation features.")

        # Display system information (models loaded, document count in vector store)
        with st.expander("ℹ️ System Information"):
            # Fetch the raw collection info. It's expected to be an int (count) from vector_store.py
            raw_collection_info = 0  # Default fallback
            if st.session_state.components_loaded:
                try:
                    raw_collection_info = components['vector_store'].get_collection_info(
                    )
                except Exception as e:
                    st.error(f"Error fetching vector store information: {e}")
                    raw_collection_info = 0

            # Since get_collection_info() in vector_store.py now returns an int directly,
            # we just display that int.
            st.write(f"📊 Documents in vector database: {raw_collection_info}")
            st.write(
                f"🧠 Embedding model: {components['embedding_handler'].model_name if st.session_state.components_loaded else 'N/A'}")
            st.write(
                f"🤖 LLM model: {components['llm_handler'].model_path.split('/')[-1] if st.session_state.components_loaded else 'N/A'} (Quantized)")

        # Add a button to reset/clear the ChromaDB for debugging
        if st.sidebar.button("⚠️ Clear Vector Database"):
            if st.session_state.components_loaded:
                try:
                    components['vector_store'].reset_collection()
                    st.sidebar.success("✅ Vector database cleared!")
                    st.session_state.pdf_processed = False  # Reset state after clearing DB
                    st.session_state.current_pdf_id = None
                except Exception as e:
                    st.sidebar.error(f"❌ Error clearing vector database: {e}")
                    st.sidebar.exception(e)


if __name__ == "__main__":
    main()
