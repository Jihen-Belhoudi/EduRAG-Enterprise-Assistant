"""
EduRAG Assistant - Streamlit Chat UI

This app provides a simple web interface for the RAG assistant.
Users can ask questions and receive grounded answers with citations.
"""

import streamlit as st
import subprocess

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "docs/faiss_index"


# -----------------------------
# Load FAISS Index
# -----------------------------
@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# -----------------------------
# Call Local LLM via Ollama
# -----------------------------
def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


# -----------------------------
# Full RAG Answer Function
# -----------------------------
def rag_answer(question: str) -> str:
    db = load_index()

    # Retrieve top chunks
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are an enterprise assistant for a training center.

Answer ONLY using the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Always cite sources like [Chunk 1], [Chunk 2].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return call_llm(prompt)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="EduRAG Assistant", layout="centered")

st.title(" EduRAG Enterprise Assistant")
st.write("Ask questions about training center policies, courses, and FAQs.")

# Chat history storage
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
question = st.chat_input("Type your question here...")

if question:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_answer(question)
            st.markdown(answer)

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})
