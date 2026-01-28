"""
EduRAG Assistant - Full Chat Answering

Pipeline:
1. Retrieve relevant chunks from FAISS
2. Send context + question to local LLM (Mistral via Ollama)
3. Generate final grounded answer with citations
"""

import subprocess

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "docs/faiss_index"


def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def call_llm(prompt: str) -> str:
    """
    Calls local LLM using Ollama (mistral).
    """
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout


def ask(question: str):
    db = load_index()

    # Retrieve top chunks
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are an enterprise assistant for a training center.

Answer ONLY using the context below.
If the answer is not in the context, say: "I don't know based on the documents."

Always cite sources like [Chunk 1], [Chunk 2].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    answer = call_llm(prompt)

    print("\n Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.lower() == "exit":
            break
        ask(q)
