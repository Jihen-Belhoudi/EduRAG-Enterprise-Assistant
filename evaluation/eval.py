"""
EduRAG Assistant - Scalable Evaluation Suite (Enterprise Version)

Features:
- Loads tests dynamically from evaluation/tests.json
- Supports categories, behaviors, citations
- Uses fuzzy matching for facts (robust to paraphrasing)
- Reports pass rate per category
- Regression-ready with test IDs

Author: EduRAG Portfolio Project
"""

import json
import subprocess
from collections import defaultdict

from rapidfuzz import fuzz

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
INDEX_DIR = "docs/faiss_index"
TEST_FILE = "evaluation/tests.json"


# ------------------------------------------------------------
# Load FAISS Index
# ------------------------------------------------------------
def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ------------------------------------------------------------
# Call Local LLM (Ollama Mistral)
# ------------------------------------------------------------
def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


# ------------------------------------------------------------
# Full RAG Answer Pipeline
# ------------------------------------------------------------
def rag_answer(question: str) -> str:
    db = load_index()
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are an enterprise assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"I don't know based on the provided documents."

Always cite sources like [Chunk 1].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return call_llm(prompt)


# ------------------------------------------------------------
# Load Test Dataset
# ------------------------------------------------------------
def load_tests():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Fuzzy Fact Matching (Robust Evaluation)
# ------------------------------------------------------------
def fuzzy_match(expected_fact: str, answer: str, threshold: int = 70) -> bool:
    """
    Returns True if the expected fact is approximately present
    in the answer (handles paraphrasing).

    Example:
    Expected: "Full refund within 7 days"
    Answer: "You get a full refund if you cancel within 7 days of enrollment"
    """

    score = fuzz.partial_ratio(expected_fact.lower(), answer.lower())
    return score >= threshold


# ------------------------------------------------------------
# Evaluation Runner
# ------------------------------------------------------------
def run_eval():
    print("\n==============================")
    print("   EduRAG Evaluation Report   ")
    print("==============================\n")

    tests = load_tests()

    passed = 0
    category_scores = defaultdict(lambda: {"pass": 0, "total": 0})

    for test in tests:
        test_id = test["id"]
        category = test["category"]
        question = test["question"]

        expected_facts = test["expected_facts"]
        must_cite = test["must_include_citation"]

        print(f"Test: {test_id}")
        print(f"Category: {category}")
        print(f"Q: {question}")

        # Generate answer
        answer = rag_answer(question)

        # -------------------------
        # Fact Check (Fuzzy)
        # -------------------------
        facts_ok = all(
            fuzzy_match(fact, answer)
            for fact in expected_facts
        )

        # -------------------------
        # Citation Check
        # -------------------------
        cite_ok = True
        if must_cite:
            cite_ok = "[chunk" in answer.lower()

        # -------------------------
        # Final Pass/Fail
        # -------------------------
        test_pass = facts_ok and cite_ok

        category_scores[category]["total"] += 1

        if test_pass:
            print("✅ PASS\n")
            passed += 1
            category_scores[category]["pass"] += 1
        else:
            print("❌ FAIL")
            print("Answer:\n", answer)
            print()

    # ------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------
    total = len(tests)

    print("\n==============================")
    print("        FINAL SUMMARY         ")
    print("==============================")
    print(f"Total Passed: {passed}/{total}")
    print(f"Overall Accuracy: {passed/total:.2%}\n")

    print("Category Breakdown:")
    for cat, score in category_scores.items():
        acc = score["pass"] / score["total"]
        print(f"- {cat}: {score['pass']}/{score['total']} ({acc:.2%})")

    print("\n==============================\n")


# ------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    run_eval()
