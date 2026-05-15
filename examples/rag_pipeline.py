"""RAG pipeline using DakeraVectorStore with LangChain.

Indexes documents into Dakera's server-side vector store (no local
embeddings needed) and retrieves the most relevant chunks for a query.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    export DAKERA_API_KEY="dk-..."          # optional
    pip install langchain-dakera langchain-core
    python rag_pipeline.py
"""

import os

from langchain_core.documents import Document

from langchain_dakera import DakeraVectorStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraVectorStore(
    api_url=api_url,
    namespace="langchain-rag-demo",
    api_key=api_key,
)

docs = [
    Document(page_content="Dakera is an AI memory platform with server-side embedding.",
             metadata={"source": "overview"}),
    Document(page_content="LangChain is a framework for building LLM applications.",
             metadata={"source": "overview"}),
    Document(page_content="Vector stores enable semantic search over document collections.",
             metadata={"source": "concepts"}),
    Document(page_content="RAG combines retrieval with generation for grounded answers.",
             metadata={"source": "concepts"}),
]

print("Indexing documents...")
ids = store.add_texts(
    [d.page_content for d in docs],
    metadatas=[d.metadata for d in docs],
)
print(f"Indexed {len(ids)} documents.")

print("\nSearching for: 'What is Dakera?'")
results = store.similarity_search("What is Dakera?", k=2)
for i, doc in enumerate(results, 1):
    print(f"  {i}. [{doc.metadata.get('source', '')}] {doc.page_content}")

print("\nSearching with scores: 'semantic search'")
scored = store.similarity_search_with_score("semantic search", k=2)
for doc, score in scored:
    print(f"  [{score:.3f}] {doc.page_content}")
