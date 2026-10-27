# langchain-dakera

[![CI](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![dakera.ai](https://img.shields.io/badge/dakera.ai-website-22c55e?style=flat-square)](https://dakera.ai)
[![Docs](https://img.shields.io/badge/docs-dakera.ai%2Fdocs-3b82f6?style=flat-square)](https://dakera.ai/docs)

**LangChain components backed by [Dakera](https://dakera.ai) — persistent agent memory and
server-side RAG with no local embedding model.**

- `DakeraMemory` — conversation memory that survives restarts: each turn is stored on the Dakera
  server and the most relevant memories are recalled into the prompt (decay-weighted by
  importance, recency and access).
- `DakeraVectorStore` — a LangChain `VectorStore` whose texts are embedded by the Dakera server
  (no OpenAI / Hugging Face embedding calls), with semantic, hybrid and BM25 search.
- `DakeraSessionManager`, `DakeraKnowledgeGraph`, `DakeraEntityExtractor`,
  `DakeraNamespaceManager`, `DakeraAgentTools` — sessions, the memory graph, entity extraction,
  namespaces and agent statistics.

## Compatibility

| langchain-dakera | Dakera server | `dakera` SDK | LangChain |
|------------------|---------------|--------------|-----------|
| **0.4.x** | v0.12.0 – **v0.12.3** | ≥ 0.14.1 | `langchain-core` ≥ 0.1 (incl. 1.x) |
| 0.3.x | v0.12.0 | ≥ 0.13.1 | `langchain-core` ≥ 0.1 |

Features marked *v0.12.3* below (event time, the long-text report, `warnings`) need a v0.12.3
server; older v0.12 servers ignore them. Python ≥ 3.10.

---

## Quick start

### 1. Run a Dakera server

```bash
docker run -d --name dakera -p 3000:3000 \
  -v dakera-data:/data \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  -e DAKERA_STORAGE=filesystem \
  ghcr.io/dakera-ai/dakera:0.12.3

curl http://localhost:3000/health   # {"service":"dakera","status":"healthy",...}
```

The server listens on port 3000 and requires an API key (here the root key `dk-mykey`). The
image ships its default embedding model, so it is ready in seconds. Docker Compose, Kubernetes
and Helm setups are in [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy).

### 2. Install

```bash
pip install langchain-dakera
```

### 3. Remember across sessions

```python
from langchain_dakera import DakeraMemory

memory = DakeraMemory(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    agent_id="my-agent",
)

memory.save_context({"input": "I'm Alice, I build chatbots."}, {"output": "Nice to meet you!"})

# Later — another process, after a restart — the memory is still there:
print(memory.load_memory_variables({"input": "What do I build?"})["history"])
```

---

## What's new in 0.4.0

- **Server v0.12.3** support with `dakera>=0.14.1`.
- Recalls return at most `recall_k` / `top_k` ranked memories (a v0.12.3 server can add a
  session's neighbouring memories as context; they are requested separately and left out), and
  a memory and the sentences the server derives from it are not recalled twice.
- `DakeraMemory.store(event_time=...)` (*v0.12.3*), `recall(rerank=...)` with a `score` per result,
  `consolidate(memory_ids)`.
- Sessions: `start(idle_timeout_secs=...)`, `touch()`, `summary` / `ended_reason`.
- `DakeraAgentTools.import_memories` / `export_memories(format=...)` (`jsonl`, `csv`, `mem0`, `zep`).
- `DakeraVectorStore`: `hybrid_search` really blends semantic and BM25 scores (`alpha`),
  `fulltext_search` returns the documents' text, `delete(ids)`, and `embedding_model` is applied.
- Works with LangChain 1.x (`BaseMemory` from `langchain-classic`).

See the [CHANGELOG](CHANGELOG.md) for details.

---

## DakeraMemory

Persistent memory for conversations. `save_context()` stores each turn as one memory
(`"Human: ...\nAI: ..."`); `load_memory_variables()` recalls the memories most relevant to the
new input and returns them, one per line, under `memory_key` (`"history"`).

### With a chat model (LangChain 1.x)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_dakera import DakeraMemory

memory = DakeraMemory(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    agent_id="chat-agent",
    recall_k=5,  # memories recalled per turn
    importance=0.7,  # importance of the stored turns
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. What you remember about the user:\n{history}"),
        ("human", "{input}"),
    ]
)
chain = prompt | ChatOpenAI(model="gpt-4o")


def chat(user_input: str) -> str:
    variables = memory.load_memory_variables({"input": user_input})
    answer = chain.invoke({"input": user_input, **variables}).content
    memory.save_context({"input": user_input}, {"output": answer})
    return answer


chat("My name is Alice and I'm building a chatbot.")
print(chat("What was I building?"))  # "You're building a chatbot." — even after a restart
```

### With the legacy chains (`langchain-classic`)

LangChain 1.0 moved `BaseMemory` and the memory-based chains to `langchain-classic`. With it
installed, `DakeraMemory` is a `BaseMemory` and plugs into those chains:

```bash
pip install "langchain-dakera[classic]" langchain-openai
```

```python
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_dakera import DakeraMemory

memory = DakeraMemory(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent")
chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o"), memory=memory)
print(chain.predict(input="What was I building?"))
```

(With `langchain-core` < 1.0, `BaseMemory` comes from `langchain_core.memory`.)

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `api_key` | `str` | `""` | Dakera API key (empty when the server runs without auth) |
| `agent_id` | `str` | — | Agent whose memories are stored and recalled |
| `recall_k` | `int` | `5` | Memories recalled per turn |
| `min_importance` | `float` | `0.0` | Recall only memories at least this important |
| `memory_key` | `str` | `"history"` | Prompt variable the memories are injected as |
| `input_key` | `str \| None` | first input | Input used as the recall query |
| `importance` | `float` | `0.7` | Importance (0–1) of stored turns |
| `memory_type` | `str` | `"episodic"` | Type of stored turns: `episodic`, `semantic`, `procedural`, `working` |
| `tags` | `list[str]` | `[]` | Tags added to every stored turn |
| `ttl_seconds` | `int \| None` | `None` | Expiry of stored turns (default: never) |
| `session_id` | `str \| None` | `None` | Session the stored turns belong to (see [Sessions](#sessions)) |

### Direct memory API

```python
from datetime import datetime, timezone

stored = memory.store(
    "The user hiked the Tour du Mont Blanc.",
    importance=0.8,
    memory_type="semantic",
    tags=["travel"],
    event_time=datetime(2025, 7, 14, tzinfo=timezone.utc),  # when it happened (v0.12.3)
)
memory_id = stored["id"]

hits = memory.recall("Where has the user hiked?", top_k=3, tags=["travel"], rerank=True)
# [{"id", "content", "importance", "score", "metadata"}, ...] best first

memory.search("Mont Blanc", top_k=5)  # server memory search, with "score"
memory.batch_recall(["hiking", "work"], top_k=2)  # one result list per query
memory.update_importance(memory_id, 0.95)
memory.forget(memory_id)
```

- `store()` returns the server's answer: the memory `id`, plus on a v0.12.3 server `truncated` /
  `embedded_tokens` when the text is longer than the embedding model reads, and `warnings` naming
  request fields the server ignored.
- `event_time` (Unix seconds, an RFC 3339 string or a timezone-aware `datetime`) is when the
  remembered event happened; recency and time filters use it instead of the time of storage.
- Recall results are ranked by the server's `rank_score` (`score`); `rerank=False` skips the
  cross-encoder reranking.
- `consolidate([id1, id2, ...], target_type="semantic")` merges memories into one new memory and
  forgets the sources. Memories with a TTL, an attachment or user metadata (including
  `event_time`) cannot be consolidated (`400`).
- `clear()` is a no-op: Dakera memories are persistent by design; use `forget()` /
  `batch_forget()`.

---

## DakeraVectorStore

A LangChain `VectorStore` over a Dakera namespace. The server embeds the texts (`bge-large` by
default) and indexes them for BM25 keyword search too — no `Embeddings` object needed.

### Indexing and retrieval

```python
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dakera import DakeraVectorStore

docs = [
    Document(page_content=p.read_text(), metadata={"source": str(p)})
    for p in Path("./docs").glob("**/*.md")
]
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

vectorstore = DakeraVectorStore(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    namespace="my-docs",
)
ids = vectorstore.add_documents(chunks)  # the server computes the embeddings

vectorstore.similarity_search("memory decay", k=4)
vectorstore.similarity_search("memory decay", k=4, filter={"source": {"$eq": "docs/decay.md"}})
vectorstore.similarity_search_with_score("memory decay", k=4)  # [(Document, cosine), ...]
```

`langchain-text-splitters` is a separate package (`pip install langchain-text-splitters`).

### RAG chain

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
prompt = ChatPromptTemplate.from_template(
    "Answer from the context only.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(documents):
    return "\n\n".join(d.page_content for d in documents)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)
print(rag_chain.invoke("How does Dakera handle memory decay?"))
```

### Hybrid and keyword search

```python
# alpha = weight of semantic similarity: 1.0 semantic only, 0.0 keywords (BM25) only
docs = vectorstore.hybrid_search("FastAPI Starlette", k=4, alpha=0.5)
docs[0].metadata  # {..., "score", "vector_score", "text_score", "id"}

# BM25 only; metadata["score"] is the BM25 score
docs = vectorstore.fulltext_search("Starlette", k=10)

vectorstore.delete(ids)  # removes the documents from both indexes
```

- `hybrid_search` takes the `fetch_k` (default `max(4·k, 20)`) documents semantically nearest to
  the query (matching `filter`) and scores each `alpha · vector_score + (1 − alpha) ·
  text_score`: the cosine similarity and the BM25 score normalised to the best candidate. Raise
  `fetch_k` when exact keyword matches far from the query's meaning must be found.
- `fulltext_search` ranks the whole namespace by BM25. The server's BM25 route returns ids only,
  so the text is read from the `fetch_k` semantically nearest documents; a hit outside them has
  an empty `page_content` and `metadata["text_found"] == False`. With a `filter`, only hits
  among the matching candidates are returned (the BM25 index keeps no metadata).
- `batch_search(queries, k)` runs several similarity searches; `asimilarity_search`,
  `aadd_texts`, `ahybrid_search` and `adelete` are the async variants.

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `namespace` | `str` | — | Namespace to read/write (created on the first write) |
| `api_key` | `str` | `""` | Dakera API key |
| `embedding_model` | `str \| None` | server default | Server-side embedding model, e.g. `"bge-large"`, `"minilm"`; use one model per namespace |

---

## Sessions

Group the memories of one conversation. The server ends a session after 4 h without activity by
default (`ended_reason: "idle"`).

```python
from langchain_dakera import DakeraMemory, DakeraSessionManager

sessions = DakeraSessionManager(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent"
)
session_id = sessions.start(metadata={"topic": "onboarding"}, idle_timeout_secs=1800)

memory = DakeraMemory(
    api_url="http://localhost:3000",
    api_key="dk-mykey",
    agent_id="chat-agent",
    session_id=session_id,
)
memory.save_context({"input": "How do I reset my password?"}, {"output": "Settings > Security."})

sessions.touch()  # record activity: "active" (or "ended")
sessions.memories(session_id)  # the session's memories, oldest first
sessions.end(summary="Password reset help")
sessions.get(session_id)  # {"id", "summary", "ended_reason", "memory_count", ...}
sessions.list_sessions(active_only=False)

with DakeraSessionManager(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent"
) as s:
    ...  # started on entry, ended on exit
```

## Knowledge graph

```python
from langchain_dakera import DakeraKnowledgeGraph

# memory_a, memory_b, memory_c: ids of the agent's memories (e.g. memory.store(...)["id"])
kg = DakeraKnowledgeGraph(
    api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent"
)
kg.link(memory_a, memory_b, label="same project")  # explicit link between two memories
kg.link(memory_b, memory_c)
kg.query(root_id=memory_a, max_depth=2)  # {"edges", "node_count", "edge_count"}
kg.find_path(memory_a, memory_c)  # {"path": [a, b, c], "hop_count": 2, ...}
kg.build(memory_a, depth=2)  # the graph around one memory
kg.export()  # every edge of the agent
kg.full_graph()  # nodes and edges, for visualisation
kg.summarize([memory_a, memory_b])  # stores a new summary memory
kg.deduplicate()  # merge duplicate memories
```

## Entities, namespaces and agent tools

```python
from langchain_dakera import DakeraAgentTools, DakeraEntityExtractor, DakeraNamespaceManager

entities = DakeraEntityExtractor(api_url="http://localhost:3000", api_key="dk-mykey")
entities.extract(
    "Sarah Chen joined Acme in Paris.", entity_types=["person", "organization", "location"]
)
entities.configure("_dakera_agent_chat-agent", extract_entities=True, entity_types=["person"])
entities.memory_entities(memory_a)  # entities extracted from a stored memory

namespaces = DakeraNamespaceManager(api_url="http://localhost:3000", api_key="dk-mykey")
namespaces.create("embeddings-768", dimensions=768)
namespaces.list_namespaces()
namespaces.stats("my-docs")

agent = DakeraAgentTools(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent")
agent.stats()  # {"total_memories", "memories_by_type", "total_sessions", ...}
agent.memories(limit=20)  # newest first
records = agent.export_memories()  # format="jsonl" (default), "csv", "mem0" or "zep"
agent.import_memories([{"memory": "The user prefers dark mode."}], format="mem0")
```

- Named entity types load the server's GLiNER model on first use (a ~782 MB download from
  Hugging Face); a server that cannot load it serves its rule-based extractor only. An agent's
  memories live in the namespace `_dakera_agent_<agent_id>`.
- `import_memories(format="jsonl")` takes records as `export_memories()` returns them; `mem0`
  takes `[{"memory": ...}]`, `zep` `[{"content": ...}]` and `csv` the CSV text.

---

## Examples

Every example runs against a live server and needs no LLM API key:

| Example | Shows |
|---------|-------|
| [`basic_memory.py`](examples/basic_memory.py) | `save_context` / `load_memory_variables` |
| [`advanced_memory.py`](examples/advanced_memory.py) | importance, tags, types, TTL, `event_time`, filtered recall, consolidate, forget |
| [`batch_operations.py`](examples/batch_operations.py) | batch store, `batch_recall`, `batch_forget` |
| [`sessions.py`](examples/sessions.py) | sessions: idle timeout, `touch`, summary, context manager |
| [`knowledge_graph.py`](examples/knowledge_graph.py) | link, query, path, build, export, summarize |
| [`rag_pipeline.py`](examples/rag_pipeline.py) | `add_documents`, retriever, LCEL RAG prompt |
| [`hybrid_search.py`](examples/hybrid_search.py) | semantic, hybrid (`alpha`), BM25 and filtered search |

```bash
export DAKERA_API_URL=http://localhost:3000
export DAKERA_API_KEY=dk-mykey   # leave unset if the server runs without auth
python examples/basic_memory.py
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) — unit tests run without a server; the live tests in
`tests/test_integration.py` run when `DAKERA_API_URL` is set.

## Related packages

| Package | Framework | Language |
|---------|-----------|----------|
| [`llamaindex-dakera`](https://pypi.org/project/llamaindex-dakera/) | LlamaIndex | Python |
| [`crewai-dakera`](https://pypi.org/project/crewai-dakera/) | CrewAI | Python |
| [`autogen-dakera`](https://pypi.org/project/autogen-dakera/) | AutoGen | Python |
| [`strands-dakera`](https://pypi.org/project/strands-dakera/) | Strands Agents | Python |
| [`@dakera-ai/langchain`](https://www.npmjs.com/package/@dakera-ai/langchain) | LangChain.js | TypeScript |

## Links

- [Dakera](https://dakera.ai) · [Documentation](https://dakera.ai/docs) ·
  [LangChain integration guide](https://dakera.ai/integrations/langchain.html)
- [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy) — Docker Compose, Kubernetes, Helm
- [Dakera Python SDK](https://github.com/Dakera-AI/dakera-py) — the low-level client this package uses

## License

MIT © [Dakera AI](https://dakera.ai)
