from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pickle


# Load embedding model correctly for FAISS
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def chunk_text(text, chunk_size=1200, chunk_overlap=250):
    """
    Larger chunks keep paragraphs intact → answers more accurate.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def embed_and_build_index(chunks):
    """
    Build FAISS index using chunks.
    """
    docs = [Document(page_content=c) for c in chunks]
    db = FAISS.from_documents(docs, embedding_model)
    return db


def save_index(db, path):
    with open(path, "wb") as f:
        pickle.dump(db, f)


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)
