"""Hybrid search (vector + BM25) with LangChain and Dakera.

Demonstrates combining semantic vector search with keyword-based
full-text search for better retrieval.

Usage:
    export DAKERA_API_URL="http://localhost:3000"
    python hybrid_search.py
"""

import os

from langchain_dakera.vectorstore import DakeraVectorStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraVectorStore(
    api_url=api_url,
    api_key=api_key,
    namespace="docs",
)

documents = [
    "Python is a high-level programming language known for readability.",
    "Rust provides memory safety without garbage collection.",
    "TypeScript adds static types to JavaScript for better tooling.",
    "Go is designed for concurrent systems programming at Google.",
    "FastAPI is a modern Python web framework built on Starlette.",
]

print("Indexing documents...")
store.add_texts(documents)

print("\n--- Vector search: 'memory safe language' ---")
results = store.similarity_search("memory safe language", k=3)
for doc in results:
    print(f"  [{doc.metadata['score']:.3f}] {doc.page_content[:60]}")

print("\n--- Hybrid search: 'Python web' (alpha=0.5) ---")
results = store.hybrid_search("Python web", k=3, alpha=0.5)
for doc in results:
    print(f"  [{doc.metadata['score']:.3f}] {(doc.page_content or '')[:60]}")

print("\n--- Full-text BM25: 'Google concurrent' ---")
results = store.fulltext_search("Google concurrent", k=3)
for doc in results:
    print(f"  [{doc.metadata['score']:.3f}] {(doc.page_content or '')[:60]}")
