"""Retrieval-augmented generation with DakeraVectorStore.

Indexes documents into Dakera (the server embeds them, no local embedding
model or embedding API key), then builds a LangChain retriever and an LCEL
pipeline that fills a RAG prompt with the retrieved context. Pipe the prompt
into any chat model (e.g. ``ChatOpenAI``) to generate the answer.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/rag_pipeline.py
"""

import os
import uuid

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_dakera import DakeraVectorStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

store = DakeraVectorStore(
    api_url=api_url,
    api_key=api_key,
    namespace=f"langchain-rag-{uuid.uuid4().hex[:8]}",
)

docs = [
    Document(
        page_content="Dakera is an AI agent memory server with server-side embedding.",
        metadata={"source": "overview"},
    ),
    Document(
        page_content="Dakera decays memory importance over time, weighted by access.",
        metadata={"source": "concepts"},
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications.",
        metadata={"source": "overview"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for grounded answers.",
        metadata={"source": "concepts"},
    ),
]
ids = store.add_documents(docs)
print(f"Indexed {len(ids)} documents.")

print("\n--- Similarity search with scores: 'semantic search over memories' ---")
for doc, score in store.similarity_search_with_score("semantic search over memories", k=2):
    print(f"  [{score:.3f}] ({doc.metadata['source']}) {doc.page_content}")

retriever = store.as_retriever(search_kwargs={"k": 2})
prompt = ChatPromptTemplate.from_template(
    "Answer from the context only.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(documents: list[Document]) -> str:
    return "\n".join(f"- {d.page_content}" for d in documents)


# prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser() would generate the answer.
rag = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt

print("\n--- RAG prompt for: 'How does Dakera handle memory decay?' ---")
print(rag.invoke("How does Dakera handle memory decay?").to_string())

store.delete(ids)
print(f"\nDeleted {len(ids)} documents.")
