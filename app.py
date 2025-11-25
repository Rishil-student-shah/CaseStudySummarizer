# app.py (Enhanced UI/UX - V10: File Deletion Functionality)

import os
import streamlit as st
from typing import Optional

# --- Import RAG components (assuming they are in the same environment) ---
from pdf_reader import extract_pdf_text
from embedder import chunk_text, embed_and_build_index, save_index
from rag_pipeline import answer_query
from utils import get_file_size # Assuming utils.py is available
import time

# --- Configuration ---
APP_TITLE = "🔬 AI-Powered Case Study Analyst"
SAMPLE_DIR = "sample_cases"
OUTPUT_DIR = "outputs"

# Ensure directories exist
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---

def get_available_files():
    """Fetches available PDFs and Indexes, returning names and paths."""
    existing_pdfs = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    available_indexes = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_index.pkl")]
    return existing_pdfs, available_indexes

def delete_file_and_index(filename: str):
    """Deletes the PDF and its corresponding index file."""
    
    # 1. Delete PDF file
    pdf_path = os.path.join(SAMPLE_DIR, filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        st.toast(f"🗑️ Deleted PDF: {filename}", icon="✅")
    
    # 2. Delete Vector Index file
    index_filename = filename.replace(".pdf", "_index.pkl")
    index_path = os.path.join(OUTPUT_DIR, index_filename)
    if os.path.exists(index_path):
        os.remove(index_path)
        st.toast(f"🗑️ Deleted Index: {index_filename}", icon="✅")
    
    # Reset session state if the deleted file was the one currently selected
    if st.session_state["current_pdf_name"] == filename:
        st.session_state["current_pdf_name"] = None
        st.session_state["current_index_path"] = None

    # Rerun to update the sidebar file list
    time.sleep(0.1) # Brief pause before rerunning
    st.rerun()

def handle_upload_and_indexing(uploaded_file):
    """Handles saving the file, processing text, chunking, and indexing."""
    
    # 1. Save PDF
    save_path = os.path.join(SAMPLE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.toast(f"File saved: {uploaded_file.name}", icon="💾")

    # 2. Extract and Index
    process_bar = st.progress(0, text="Starting text extraction...")
    
    try:
        raw_text = extract_pdf_text(save_path)
        process_bar.progress(33, text="Extraction complete. Generating text chunks...")
        
        chunks = chunk_text(raw_text)
        st.info(f"📄 Successfully created **{len(chunks)}** text chunks for indexing.")
        process_bar.progress(66, text="Chunks generated. Building FAISS Vector Index...")

        db = embed_and_build_index(chunks)

        index_filename = uploaded_file.name.replace(".pdf", "_index.pkl")
        index_path = os.path.join(OUTPUT_DIR, index_filename)
        save_index(db, index_path)

        st.session_state["current_index_path"] = index_path
        st.session_state["current_pdf_name"] = uploaded_file.name
        
        process_bar.progress(100, text="Indexing Complete! Ready to analyze.")
        st.success(f"🎉 Case Study **{uploaded_file.name}** is ready to be queried!")
        st.balloons()
        
    except Exception as e:
        process_bar.empty()
        st.error(f"An error occurred during processing: {e}")

# --- Streamlit UI Components ---

def setup_page():
    """Sets up the Streamlit page configuration and main header."""
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={'About': 'A RAG-based tool for case study analysis.'}
    )
    
    # Custom CSS for modern look and feel and responsiveness
    st.markdown(
        """
        <style>
        /* Ensures the main content area has some padding for desktop use */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }

        /* --- Global Header Container --- */
        .header-container {
            padding: 10px 0;
            margin-bottom: 20px;
            border-bottom: 3px solid #1f77b4; 
        }
        
        /* Main title - Adjusted size for better balance */
        .main-header {
            font-size: 4em !important; 
            font-weight: 900 !important; 
            color: #FFFFFF !important; 
            margin: 0;
            display: flex;
            align-items: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7); 
        }

        /* Style for the caption/description below the title - Increased size */
        .stCaption p {
            font-size: 1.4em !important; 
            color: #A0A0A0 !important; 
            margin-top: 5px;
        }

        /* --- TALLER CHAT INPUT / QUESTION BLOCK --- */
        /* Targets the text area inside the chat input for a fixed, taller height */
        .stChatInput > div > div > textarea {
            min-height: 80px !important; /* Increased height */
            padding-top: 15px !important; 
            padding-bottom: 15px !important;
        }
        
        /* Custom card style for answers */
        .answer-card {
            border-left: 5px solid #1f77b4;
            padding: 20px;
            border-radius: 12px;
            background-color: #f0f8ff; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            color: #333; 
            line-height: 1.6;
        }
        .answer-card h4 {
            color: #1f77b4;
            margin-top: 0;
            font-size: 1.2em;
        }

        /* Styling for the main action button */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.2s ease-in-out;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }

        /* Custom style for the small delete button in the sidebar */
        .delete-btn button {
            background-color: transparent !important;
            color: #ff4b4b !important;
            font-weight: bold;
            padding: 0px 5px !important;
            border: none !important;
            box-shadow: none !important;
            transform: none !important;
            transition: color 0.2s;
            line-height: 1;
        }
        .delete-btn button:hover {
            color: #ff7f7f !important;
        }

        /* Responsive adjustments for smaller screens (Streamlit automatically stacks columns, but this adds padding safety) */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .main-header {
                font-size: 2.5em !important; /* Smaller header on mobile */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use a container for the header to control its styling
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.markdown(f'<p class="main-header"> {APP_TITLE} </p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close custom header container

    # Updated description
    st.caption("A **Retrieval-Augmented Generation (RAG)** tool for academic and business case analysis.")
    st.markdown("---")


def show_sidebar_status(existing_pdfs, available_indexes):
    """Displays the status of local files in the sidebar with delete buttons."""
    st.sidebar.markdown("### 📂 Local Storage Status")
    
    # PDF Files Status
    st.sidebar.markdown("#### **Uploaded Case PDFs**")
    if existing_pdfs:
        for f in existing_pdfs:
            # Display file name, size, and delete button in a row
            cols = st.sidebar.columns([0.8, 0.2])
            
            try:
                size_kb = get_file_size(os.path.join(SAMPLE_DIR, f))
                cols[0].markdown(f"💾 **{f}** (`{size_kb} KB`)")
            except Exception:
                cols[0].markdown(f"💾 **{f}**")
            
            # The delete button uses a unique key and calls the delete function
            cols[1].markdown('<div class="delete-btn">', unsafe_allow_html=True)
            if cols[1].button("❌", key=f"delete_pdf_{f}"):
                delete_file_and_index(f)
            cols[1].markdown('</div>', unsafe_allow_html=True)

    else:
        st.sidebar.info("No PDF files found.")

    # Index Files Status
    st.sidebar.markdown("#### **Available Vector Indexes**")
    if available_indexes:
        for f_index in available_indexes:
            f_pdf = f_index.replace("_index.pkl", ".pdf")
            
            # Index deletion is now handled by PDF deletion, but we can show status
            cols = st.sidebar.columns([0.8, 0.2])
            cols[0].success(f"✅ Indexed: **{f_pdf}**")
            
            # Show a dummy delete button or simply rely on the PDF delete button
            # Since deleting the PDF also deletes the index, we just show the indexed status here.
            
    else:
        st.sidebar.warning("No Vector Indexes ready. Please upload and process a case.")


# --- Main App Logic ---

def main():
    setup_page()
    
    # Initialize session state variables
    if "current_index_path" not in st.session_state:
        st.session_state["current_index_path"] = None
    if "current_pdf_name" not in st.session_state:
        st.session_state["current_pdf_name"] = None

    # Get file status and display sidebar
    existing_pdfs, available_indexes = get_available_files()
    show_sidebar_status(existing_pdfs, available_indexes)


    # Main layout: 1 column for Upload/Index, 2 columns for Query/Answer
    # Streamlit columns are inherently responsive: they stack vertically on small screens
    col1, col2 = st.columns([1, 2])

    # ----------------------------------------------------
    # COLUMN 1: UPLOAD & INDEXING (Step 1)
    # ----------------------------------------------------
    with col1:
        with st.container(border=True):
            st.markdown("### **1. 📥 Upload & Index Case Study**")
            st.write("Upload a PDF to convert it into a searchable Vector Store.")
            
            uploaded = st.file_uploader(
                "Select a PDF file to analyze:", 
                type=["pdf"], 
                key="file_uploader"
            )
            
            if uploaded is not None:
                st.info(f"File **{uploaded.name}** uploaded successfully. Click below to index.")

                if st.button(
                    f"⚡ Start Indexing", 
                    key="process_btn", 
                    use_container_width=True
                ):
                    handle_upload_and_indexing(uploaded)
                    # Rerun to update the index list immediately
                    st.rerun() 

    # ----------------------------------------------------
    # COLUMN 2: QUERYING & ANSWER (Step 2 & 3)
    # ----------------------------------------------------
    with col2:
        with st.container(border=True):
            st.markdown("### **2. 🧠 Analyze & Get Answers**")

            # Index Selector
            index_path: Optional[str] = None

            if not available_indexes:
                st.error("⚠️ No vector index found. Please process a PDF in the left panel.")
                query_to_run = None # Disable query input if no index exists
            else:
                
                # Pre-select the HBR Case Study if available (retains previous behavior)
                default_index = 0
                hbr_index_name = "HBR Case Study_index.pkl"
                if hbr_index_name in available_indexes:
                    default_index = available_indexes.index(hbr_index_name)
                    
                choice = st.selectbox(
                    "Choose an indexed case study:",
                    options=available_indexes,
                    index=default_index, # Set default index
                    key="index_selector",
                    label_visibility="collapsed"
                )
                index_path = os.path.join(OUTPUT_DIR, choice)
                st.session_state["current_index_path"] = index_path
                st.session_state["current_pdf_name"] = choice.replace("_index.pkl", ".pdf")
                
                st.success(f"Case Selected: **{st.session_state['current_pdf_name']}** is ready for analysis.")
                
                # Query Input - ONLY using st.chat_input now
                query_to_run = st.chat_input(
                    "Type your factual question here (e.g., 'What was the key challenge?')",
                    key="query_input"
                )


            # --- RAG Execution ---
            # Now only runs if query_to_run is NOT None (i.e., user submitted a chat message)
            if query_to_run and index_path:
                with st.spinner(f"Analyzing case study: {st.session_state['current_pdf_name']}..."):
                    # Clear previous results before running new query
                    st.session_state['last_answer'] = None 
                    st.session_state['last_docs'] = None
                    st.session_state['last_query'] = query_to_run
                    
                    answer, docs = answer_query(index_path, query_to_run)

                    # Store result in session state
                    st.session_state['last_answer'] = answer
                    st.session_state['last_docs'] = docs
                    st.rerun() # Trigger a rerun to display results clearly below

            # --- Result Display ---
            if 'last_answer' in st.session_state and st.session_state['last_answer'] is not None:
                st.markdown("---")
                st.subheader(f"✅ **Result for:** *{st.session_state.get('last_query', 'Your Question')}*")
                
                # Display the final answer using the styled card
                st.markdown(
                    f'<div class="answer-card"><h4>AI Analyst Response</h4>{st.session_state["last_answer"]}</div>', 
                    unsafe_allow_html=True
                )
                
                # Download button below the answer
                st.download_button(
                    label="⬇ Download Answer Text",
                    data=st.session_state["last_answer"],
                    file_name=f"{st.session_state.get('current_pdf_name', 'analysis_result').replace('.pdf', '')}_analysis_result.txt",
                    mime="text/plain",
                    key="download_btn"
                )

                # Context Details Expander
                st.markdown("---")
                with st.expander("🔎 View Retrieved Source Context (RAG Chunks)", expanded=False):
                    if st.session_state['last_docs']:
                        st.markdown(f"**{len(st.session_state['last_docs'])}** relevant chunks retrieved from the source document.")
                        for i, d in enumerate(st.session_state['last_docs']):
                            st.markdown(f"##### Chunk {i+1}")
                            # Use st.code for better readability of text chunks
                            st.code(d.page_content, language='text') 
                    else:
                        st.info("No source documents were retrieved for this query.")
                        
if __name__ == "__main__":
    main()