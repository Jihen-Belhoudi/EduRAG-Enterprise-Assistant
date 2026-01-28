"""
EduRAG Assistant - Index Builder (Step 9)

This script builds a FAISS vector database index from documents stored in /data.

Pipeline:
1. Load documents (.txt)
2. Split them into chunks
3. Generate embeddings locally (FREE)
4. Store vectors inside FAISS
5. Save index to disk for later retrieval
"""

import os
from typing import List

# FAISS vector database
from langchain_community.vectorstores import FAISS

# Local open-source embeddings (no API key required)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chunking utility
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loader for reading text documents
from langchain_community.document_loaders import TextLoader

# LangChain Document object
from langchain_core.documents import Document


# Directory where the FAISS index will be saved
INDEX_DIR = "docs/faiss_index"


# ------------------------------------------------------------
# STEP 1: Load documents from /data folder
# ------------------------------------------------------------
def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Loads all .txt files inside the data directory
    and converts them into LangChain Document objects.
    """

    docs: List[Document] = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data folder not found: {data_dir}")

    # Loop over files in /data
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        # Only load .txt files for MVP
        if filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        raise ValueError("❌ No documents found in /data folder.")

    return docs


# ------------------------------------------------------------
# STEP 2: Build FAISS Index
# ------------------------------------------------------------
def build_index():
    """
    Full indexing pipeline:
    - Load documents
    - Chunk them
    - Create embeddings
    - Store vectors in FAISS
    """

    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents into chunks...")

    # Split docs into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # each chunk ~500 characters
        chunk_overlap=100   # overlap preserves context
    )

    chunks = splitter.split_documents(docs)

    print("Creating embeddings (local model)...")

    # FREE local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")

    # Create FAISS vector store from chunks
    db = FAISS.from_documents(chunks, embeddings)

    # Save index locally
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)

    print("\n✅ Index built successfully!")
    print("Chunks indexed:", len(chunks))
    print("Saved to:", INDEX_DIR)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("\n❌ ERROR while building index:")
        print(type(e).__name__, ":", e)
        raise
