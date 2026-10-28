"""Semantic, hybrid and BM25 search with DakeraVectorStore.

The Dakera server embeds the documents (no local embedding model) and also
indexes them for BM25 keyword search. ``hybrid_search`` weights the two with
``alpha`` (1.0 = semantic only, 0.0 = keywords only).

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/hybrid_search.py
"""

import os
import uuid

from langchain_dakera import DakeraVectorStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraVectorStore(
    api_url=api_url,
    api_key=api_key,
    namespace=f"langchain-hybrid-{uuid.uuid4().hex[:8]}",
)

texts = [
    "Python is a high-level programming language known for readability.",
    "Rust provides memory safety without garbage collection.",
    "TypeScript adds static types to JavaScript for better tooling.",
    "Go is designed for concurrent systems programming at Google.",
    "FastAPI is a modern Python web framework built on Starlette.",
]
languages = ["python", "rust", "typescript", "go", "python"]
store.add_texts(texts, metadatas=[{"language": lang} for lang in languages])
print(f"Indexed {len(texts)} documents.")

print("\n--- Semantic search: 'memory safe language' ---")
for doc in store.similarity_search("memory safe language", k=2):
    print(f"  [{doc.metadata['score']:.3f}] {doc.page_content}")

print("\n--- Hybrid search: 'Python web' at different alphas ---")
for alpha in (1.0, 0.5, 0.0):
    top = store.hybrid_search("Python web", k=1, alpha=alpha)[0]
    print(
        f"  alpha={alpha}: [{top.metadata['score']:.3f} = vector {top.metadata['vector_score']:.3f}"
        f" / bm25 {top.metadata['text_score']:.3f}] {top.page_content}"
    )

print("\n--- BM25 full-text search: 'Google concurrent' ---")
for doc in store.fulltext_search("Google concurrent", k=2):
    print(f"  [{doc.metadata['score']:.3f}] {doc.page_content}")

print("\n--- Hybrid search with a metadata filter ---")
for doc in store.hybrid_search("readable", k=2, filter={"language": {"$eq": "python"}}):
    print(f"  [{doc.metadata['language']}] {doc.page_content}")
