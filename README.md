# EduRAG Enterprise Assistant (Local RAG + Local LLM)

Enterprise-style RAG assistant that answers questions from internal policy documents with citations.

## Features
- FAISS vector index (local)
- SentenceTransformers embeddings (local)
- Mistral via Ollama for generation (local)
- Streamlit chat UI
- Scalable evaluation suite (JSON test set + reporting)

## Architecture
data/ → chunking → embeddings → FAISS → retrieval → Ollama (Mistral) → answer + citations

## Setup
### 1) Install dependencies
pip install -r requirements.txt

### 2) Install Ollama and pull model
ollama run mistral

### 3) Build the index
python app/rag.py

### 4) Run the chat UI
streamlit run app/ui.py

## Evaluation
python evaluation/eval.py
