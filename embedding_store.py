"""
Embedding & Vector Store Module
Creates embeddings using Sentence-Transformers and stores them in FAISS.
Supports saving/loading for persistence across sessions.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Load the Sentence-Transformer embedding model.
    Uses all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
    """
    print(f" Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    print("   Embedding model loaded")
    return embeddings


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Args:
        chunks: List of LangChain Document objects
        embeddings: HuggingFaceEmbeddings model

    Returns:
        FAISS vector store
    """
    print(f" Building FAISS vector store from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"    Vector store built ({len(chunks)} vectors)")
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = VECTOR_STORE_DIR):
    """Save the FAISS vector store to disk for reuse."""
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f" Vector store saved to: {path}/")


def load_vector_store(embeddings: HuggingFaceEmbeddings, path: str = VECTOR_STORE_DIR) -> FAISS:
    """Load a previously saved FAISS vector store."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at: {path}")
    print(f" Loading vector store from: {path}/")
    vector_store = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    print("    Vector store loaded")
    return vector_store


def similarity_search(vector_store: FAISS, query: str, k: int = 5) -> list:
    """
    Search the vector store for the most relevant chunks.

    Args:
        vector_store: FAISS vector store
        query: Search query string
        k: Number of results to return

    Returns:
        List of (Document, score) tuples
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def create_or_load_vector_store(
    chunks: list = None,
    force_rebuild: bool = False,
    store_path: str = VECTOR_STORE_DIR,
) -> tuple:
    """
    Create a new vector store or load an existing one.

    Args:
        chunks: Document chunks (required if building new store)
        force_rebuild: If True, rebuild even if saved store exists
        store_path: Path to save/load the vector store

    Returns:
        Tuple of (FAISS vector_store, HuggingFaceEmbeddings)
    """
    embeddings = get_embedding_model()

    # Try loading existing store
    if not force_rebuild and os.path.exists(store_path):
        try:
            vector_store = load_vector_store(embeddings, store_path)
            return vector_store, embeddings
        except Exception as e:
            print(f"  Could not load saved store: {e}. Rebuilding...")

    # Build new store
    if chunks is None or len(chunks) == 0:
        raise ValueError("No chunks provided and no saved vector store found.")

    vector_store = build_vector_store(chunks, embeddings)
    save_vector_store(vector_store, store_path)
    return vector_store, embeddings


if __name__ == "__main__":
    from document_processor import process_pdf
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Annual-Report-FY-2023-24.pdf"
    chunks = process_pdf(pdf_path)
    if chunks:
        vs, emb = create_or_load_vector_store(chunks, force_rebuild=True)
        results = similarity_search(vs, "What is Swiggy's revenue?", k=3)
        for doc, score in results:
            print(f"\n[Score: {score:.4f}] Page {doc.metadata.get('page_number')}")
            print(f"  {doc.page_content[:200]}")
