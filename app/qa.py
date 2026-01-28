"""
EduRAG Assistant - Question Answering (Step 10)

This script:
1. Loads the FAISS index
2. Retrieves relevant chunks
3. Prints the sources used
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "docs/faiss_index"


def load_index():
    """Load the saved FAISS vector index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


def ask_question(question: str):
    """Retrieve the most relevant chunks for a question."""

    db = load_index()

    print("\n🔍 Question:", question)

    # Retrieve top 3 most relevant chunks
    results = db.similarity_search(question, k=3)

    print("\n📌 Retrieved Context:\n")

    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")

        if q.lower() == "exit":
            break

        ask_question(q)
