# langchain-dakera

[![CI](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Drop-in LangChain components backed by [Dakera](https://github.com/Dakera-AI/dakera-deploy) — persistent agent memory and server-side RAG with no local embedding model.**

`DakeraMemory` gives your chains conversation memory that survives restarts. `DakeraVectorStore` powers RAG with Dakera's built-in embedding engine — no OpenAI embeddings needed.

---

## Quick Start

### Step 1 — Run Dakera

Dakera is a self-hosted memory server. Spin it up with Docker:

```bash
docker run -d \
  --name dakera \
  -p 3300:3300 \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  ghcr.io/dakera-ai/dakera:latest
```

For a production setup with persistent storage, use Docker Compose:

```bash
# Download and start
curl -sSfL https://raw.githubusercontent.com/Dakera-AI/dakera-deploy/main/docker-compose.yml \
  -o docker-compose.yml
DAKERA_API_KEY=dk-mykey docker compose up -d

# Verify it's running
curl http://localhost:3300/health
```

> Full deployment guide: [github.com/Dakera-AI/dakera-deploy](https://github.com/Dakera-AI/dakera-deploy)

### Step 2 — Install the integration

```bash
pip install langchain-dakera
```

### Step 3 — Use it

```python
from langchain_dakera import DakeraMemory, DakeraVectorStore

# Persistent conversation memory
memory = DakeraMemory(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    agent_id="my-agent",
)

# RAG vector store — no local embedding model needed
vectorstore = DakeraVectorStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="my-docs",
)
```

---

## Installation

```bash
pip install langchain-dakera
```

**Requirements:** Python ≥ 3.10, a running Dakera server (see Step 1 above)

---

## DakeraMemory

Persistent semantic memory for LangChain conversation chains. Stores and recalls conversation history using Dakera's hybrid search.

### Usage

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_dakera import DakeraMemory

memory = DakeraMemory(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    agent_id="chat-agent",
    top_k=5,        # memories to recall per turn
    importance=0.7, # importance score for stored memories
)

chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o"),
    memory=memory,
)

# First session
response = chain.predict(input="My name is Alice and I'm building a chatbot.")
print(response)

# Later session — memory persists across restarts
response = chain.predict(input="What was I building?")
print(response)  # "You mentioned you were building a chatbot."
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `api_key` | `str` | `""` | Dakera API key |
| `agent_id` | `str` | — | Agent identifier for memory namespacing |
| `top_k` | `int` | `5` | Memories to surface per turn |
| `min_importance` | `float` | `0.0` | Minimum importance threshold for recall |
| `importance` | `float` | `0.7` | Importance assigned to stored memories |
| `memory_key` | `str` | `"history"` | Key injected into the prompt |
| `input_key` | `str` | first key | Input key used as recall query |

---

## DakeraVectorStore

Server-side embedded vector store for RAG. Dakera handles embeddings — no OpenAI or Hugging Face API calls needed for indexing.

### Indexing documents

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_dakera import DakeraVectorStore

# Load and split documents
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Index into Dakera (server handles embedding)
vectorstore = DakeraVectorStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="my-docs",
)
vectorstore.add_documents(chunks)
```

### RAG chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_dakera import DakeraVectorStore

vectorstore = DakeraVectorStore(
    api_url="http://localhost:3300",
    api_key="dk-mykey",
    namespace="my-docs",
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)

answer = qa_chain.run("How does Dakera handle memory decay?")
print(answer)
```

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `api_key` | `str` | `""` | Dakera API key |
| `namespace` | `str` | — | Vector namespace to read/write |
| `embedding_model` | `str` | namespace default | Server-side embedding model override |

---

## Related packages

| Package | Framework | Language |
|---------|-----------|----------|
| `crewai-dakera` | CrewAI | Python |
| `llamaindex-dakera` | LlamaIndex | Python |
| `autogen-dakera` | AutoGen | Python |
| `@dakera-ai/langchain` | LangChain.js | TypeScript |

---

## Links

- [Dakera Server](https://github.com/Dakera-AI/dakera-deploy) — self-hosted memory server
- [Dakera Python SDK](https://github.com/Dakera-AI/dakera-py) — low-level API client
- [Documentation](https://docs.dakera.ai/integrations/langchain)
- [All integrations](https://github.com/Dakera-AI/dakera-integrations)

---

## License

MIT © [Dakera AI](https://dakera.ai)
