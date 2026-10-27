# langchain-dakera

**Persistent, decay-weighted memory and server-side RAG for LangChain — backed by a self-hosted
[Dakera](https://dakera.ai) server.**

[![CI](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Dakera-AI/dakera-langchain/blob/main/LICENSE)
[![Dakera server](https://img.shields.io/badge/Dakera%20server-v0.12.3-0ea5e9)](https://dakera.ai)
[![LoCoMo](https://img.shields.io/badge/LoCoMo-88.2%25-22c55e)](https://dakera.ai/benchmark)

Give a LangChain app a memory that survives restarts: every conversation turn is stored on your
Dakera server and the most relevant memories, ranked by importance, recency and access, are
recalled into the next prompt. Dakera embeds text on the server, so neither the memory nor the
vector store needs an embedding API key — and the whole backend is one self-hosted binary.

## Why Dakera

- **Memory that outlives the process** — `DakeraMemory` stores each turn on the server; a new
  process, a new day or another replica recalls it.
- **Recall that ranks like memory should** — decay-weighted by importance, recency and access,
  with optional cross-encoder reranking, tag / importance filters and event time.
- **No embedding bill** — `DakeraVectorStore` sends text and the server embeds it (`bge-large`
  by default); no `Embeddings` object, no OpenAI or Hugging Face embedding calls.
- **Semantic, hybrid and BM25 search** in one store, with metadata filters.
- **More than a buffer** — sessions, a memory knowledge graph, entity extraction and
  consolidation come with the same server.
- **Measured** — 88.2% Recall@20 on the [LoCoMo](https://dakera.ai/benchmark) long-conversation
  memory benchmark.

## Contents

- [Quick start](#quick-start)
- [Compatibility](#compatibility)
- [What's new in 0.4.0](#whats-new-in-040)
- [DakeraMemory](#dakeramemory)
- [DakeraVectorStore](#dakeravectorstore)
- [DakeraSessionManager](#dakerasessionmanager)
- [DakeraKnowledgeGraph](#dakeraknowledgegraph)
- [DakeraEntityExtractor](#dakeraentityextractor)
- [DakeraNamespaceManager](#dakeranamespacemanager)
- [DakeraAgentTools](#dakeraagenttools)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Related packages](#related-packages)
- [Links](#links)
- [License](#license)

## Quick start

**1. Run Dakera**

```bash
docker run -d --name dakera -p 3000:3000 \
  -v dakera-data:/data \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  -e DAKERA_STORAGE=filesystem \
  ghcr.io/dakera-ai/dakera:0.12.3

curl http://localhost:3000/health   # {"service":"dakera","status":"healthy","version":"0.12.3",...}
```

The image ships its default embedding model, so the server is ready in seconds; `dk-mykey` is
the root API key, and the `dakera-data` volume keeps the memories across container restarts.
Docker Compose, Kubernetes and Helm setups are in
[dakera-deploy](https://github.com/Dakera-AI/dakera-deploy).

**2. Install**

```bash
pip install langchain-dakera
# with LangChain 1.x legacy chains (ConversationChain, ...): pip install "langchain-dakera[classic]"
```

**3. Remember across sessions**

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
# Human: I'm Alice, I build chatbots.
# AI: Nice to meet you!
```

## Compatibility

| langchain-dakera | Dakera server | `dakera` SDK | LangChain | Python |
|------------------|---------------|--------------|-----------|--------|
| **0.4.x** | v0.12.0 – **v0.12.3** | ≥ 0.14.1 | `langchain-core` ≥ 0.1, incl. 1.x (+ `langchain-classic` for `BaseMemory`) | ≥ 3.10 |
| 0.3.x | v0.12.0 | ≥ 0.13.1 | `langchain-core` ≥ 0.1 | ≥ 3.10 |

Event time, the long-text report and `warnings` (marked *v0.12.3* below) need a v0.12.3 server;
older v0.12 servers ignore them.

## What's new in 0.4.0

- **Dakera server v0.12.3** support with `dakera>=0.14.1`.
- Recalls return at most `recall_k` / `top_k` ranked memories: the session-neighbourhood context
  a v0.12.3 server can add is requested separately and left out, and a memory is never recalled
  twice next to the sentences the server derives from it.
- `DakeraMemory.store(event_time=...)` (*v0.12.3*), `recall(rerank=...)` with a `score` per
  result, `consolidate(memory_ids)`.
- `DakeraMemory` is a LangChain `BaseMemory` on LangChain 1.x (via `langchain-classic`).
- `DakeraVectorStore`: `hybrid_search` really blends semantic and BM25 scores by `alpha`,
  `fulltext_search` returns the documents' text, filters work in both, `delete(ids)`, and
  `embedding_model` is applied.
- Sessions: `start(idle_timeout_secs=...)`, `touch()`, `summary` / `ended_reason`.
- `DakeraAgentTools.import_memories` / `export_memories(format=...)` (`jsonl`, `csv`, `mem0`,
  `zep`).

Full details in the [CHANGELOG](https://github.com/Dakera-AI/dakera-langchain/blob/main/CHANGELOG.md).

## DakeraMemory

Persistent conversation memory. `save_context()` stores each turn as one memory
(`"Human: ...\nAI: ..."`); `load_memory_variables()` recalls the memories most relevant to the
new input and returns them, one per line, under `memory_key` (`"history"`).

### With a chat model (LCEL)

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
  `embedded_tokens` when the text is longer than the embedding model reads, and `warnings`
  naming request fields the server ignored.
- `event_time` (Unix seconds, an RFC 3339 string or a timezone-aware `datetime`) is when the
  remembered event happened; recency and time filters use it instead of the time of storage.
- Recall results are ranked by the server's `rank_score` (`score`); `rerank=False` skips the
  cross-encoder reranking.
- `consolidate([id1, id2, ...], target_type="semantic")` merges memories into one new memory and
  forgets the sources. Memories with a TTL, an attachment or user metadata (including
  `event_time`) cannot be consolidated (`400`).
- `clear()` is a no-op: Dakera memories are persistent by design; use `forget()` /
  `batch_forget()`.

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
| `session_id` | `str \| None` | `None` | Session the stored turns belong to (see [DakeraSessionManager](#dakerasessionmanager)) |

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

- `hybrid_search` takes the `fetch_k` (default `max(4·k, 20)`) documents semantically nearest
  to the query (matching `filter`) and scores each `alpha · vector_score + (1 − alpha) ·
  text_score`: the cosine similarity and the BM25 score normalised to the best candidate.
  Raise `fetch_k` when exact keyword matches far from the query's meaning must be found.
- `fulltext_search` ranks the whole namespace by BM25. The server's BM25 route returns ids
  only, so the text is read from the `fetch_k` semantically nearest documents; a hit outside
  them has an empty `page_content` and `metadata["text_found"] == False`. With a `filter`, only
  hits among the matching candidates are returned (the BM25 index keeps no metadata).
- `batch_search(queries, k)` runs several similarity searches; `asimilarity_search`,
  `aadd_texts`, `ahybrid_search` and `adelete` are the async variants.

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | `str` | — | Dakera server URL |
| `namespace` | `str` | — | Namespace to read/write (created on the first write) |
| `api_key` | `str` | `""` | Dakera API key |
| `embedding_model` | `str \| None` | server default | Server-side embedding model, e.g. `"bge-large"`, `"minilm"`; use one model per namespace |

## DakeraSessionManager

Groups the memories of one conversation. The server ends a session after 4 h without activity
by default (`ended_reason: "idle"`).

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

| Method | Description |
|--------|-------------|
| `start(metadata=None, *, idle_timeout_secs=None)` | Start a session and make it the active one; `idle_timeout_secs=0` never ends it for inactivity |
| `touch(session_id=None)` | Record activity; returns `"active"` or `"ended"` |
| `end(summary=None)` | End the active session, optionally with a summary |
| `get(session_id)` / `list_sessions(active_only=False)` | Session details: `summary`, `ended_reason`, `memory_count`, ... |
| `memories(session_id, *, include_derived=False)` | The session's memories, without the server's derived sentence sub-memories unless asked |

Constructor: `api_url`, `agent_id`, `api_key=""`.

## DakeraKnowledgeGraph

Links between an agent's memories: explicit ones you add and the ones the server infers.

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
kg.deduplicate()  # merge the agent's near-duplicate memories
```

Constructor: `api_url`, `agent_id`, `api_key=""`.

## DakeraEntityExtractor

Named-entity extraction from text and from stored memories.

```python
from langchain_dakera import DakeraEntityExtractor

entities = DakeraEntityExtractor(api_url="http://localhost:3000", api_key="dk-mykey")
entities.extract(
    "Sarah Chen joined Acme in Paris.", entity_types=["person", "organization", "location"]
)  # [{"type", "value", "score"}, ...]
entities.configure("_dakera_agent_chat-agent", extract_entities=True, entity_types=["person"])
entities.memory_entities(memory_a)  # entities extracted from a stored memory
```

- Named entity types load the server's GLiNER model on first use (a ~782 MB download from
  Hugging Face); a server that cannot load it serves its rule-based extractor only.
- An agent's memories live in the namespace `_dakera_agent_<agent_id>`, which exists once the
  agent has stored a memory — that is the namespace to `configure()`.

Constructor: `api_url`, `api_key=""`.

## DakeraNamespaceManager

Create and inspect namespaces (vector collections) for your own data.

```python
from langchain_dakera import DakeraNamespaceManager

namespaces = DakeraNamespaceManager(api_url="http://localhost:3000", api_key="dk-mykey")
namespaces.create("embeddings-768", dimensions=768)  # dimensions is required by the server
namespaces.get("embeddings-768")  # {"name", "dimensions", "vector_count"}
namespaces.list_namespaces()
namespaces.stats("my-docs")
namespaces.delete("embeddings-768")  # deletes the namespace and all its data
```

`DakeraVectorStore` creates its namespace on the first write, so you only need this class for
namespaces with a fixed dimension or distance.

Constructor: `api_url`, `api_key=""`.

## DakeraAgentTools

Statistics, listings and import / export of one agent's memories.

```python
from langchain_dakera import DakeraAgentTools

agent = DakeraAgentTools(api_url="http://localhost:3000", api_key="dk-mykey", agent_id="chat-agent")
agent.stats()  # {"total_memories", "memories_by_type", "total_sessions", ...}
agent.memories(limit=20)  # newest first
agent.sessions(active_only=True)
records = agent.export_memories()  # format="jsonl" (default), "csv", "mem0" or "zep"
agent.import_memories([{"memory": "The user prefers dark mode."}], format="mem0")
```

`import_memories(format="jsonl")` takes records as `export_memories()` returns them; `mem0`
takes `[{"memory": ...}]`, `zep` `[{"content": ...}]` and `csv` the CSV text.

Constructor: `api_url`, `agent_id`, `api_key=""`.

## Configuration

Every class takes the server URL and API key as constructor arguments — the package reads no
environment variables itself. The examples, tests and the snippets in this README use:

| Variable | Used for | Example |
|----------|----------|---------|
| `DAKERA_API_URL` | Dakera server URL | `http://localhost:3000` |
| `DAKERA_API_KEY` | API key (leave unset for a server running without auth) | `dk-mykey` |

```python
import os

from langchain_dakera import DakeraMemory

memory = DakeraMemory(
    api_url=os.environ["DAKERA_API_URL"],
    api_key=os.environ.get("DAKERA_API_KEY", ""),
    agent_id="my-agent",
)
```

**Auth.** `DAKERA_ROOT_API_KEY` sets the server's root key. For applications, create scoped keys
(`read`, `write` or `admin`, optionally limited to namespaces) with the
[`dakera`](https://pypi.org/project/dakera/) SDK (`DakeraClient.create_key()`). An app that
stores and recalls memories needs a `write` key (`write` includes `read`).

## Examples

Every example runs against a live server and needs no LLM API key:

| Example | Shows | LLM key? |
|---------|-------|----------|
| [`basic_memory.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/basic_memory.py) | `save_context` / `load_memory_variables` | No |
| [`advanced_memory.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/advanced_memory.py) | importance, tags, types, TTL, `event_time`, filtered recall, consolidate, forget | No |
| [`batch_operations.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/batch_operations.py) | batch store, `batch_recall`, `batch_forget` | No |
| [`sessions.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/sessions.py) | sessions: idle timeout, `touch`, summary, context manager | No |
| [`knowledge_graph.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/knowledge_graph.py) | link, query, path, build, export, summarize | No |
| [`rag_pipeline.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/rag_pipeline.py) | `add_documents`, retriever, LCEL RAG prompt | No |
| [`hybrid_search.py`](https://github.com/Dakera-AI/dakera-langchain/blob/main/examples/hybrid_search.py) | semantic, hybrid (`alpha`), BM25 and filtered search | No |

```bash
git clone https://github.com/Dakera-AI/dakera-langchain && cd dakera-langchain
pip install -e .
export DAKERA_API_URL=http://localhost:3000
export DAKERA_API_KEY=dk-mykey   # leave unset if the server runs without auth
python examples/basic_memory.py
```

## Troubleshooting

**`ConnectionError` / connection refused.** The server listens on port 3000 (`curl
http://localhost:3000/health`). This repository's CI maps it to 3300 — use whichever port your
`docker run -p` published.

**`AuthenticationError` (401).** A server started with `DAKERA_ROOT_API_KEY` requires a key on
every request: pass `api_key=` (the root key or a scoped key). A `403` means the key lacks the
scope or namespace.

**`NotFoundError` (404 `NAMESPACE_NOT_FOUND`) on the first recall.** Servers before v0.12.3
answer a recall for an agent that has stored nothing with a 404; v0.12.3 returns no memories.
Upgrade the server, or store a memory (e.g. one `save_context()`) before the first
`load_memory_variables()`.

**Fewer or more results than expected.** Recalls return at most `recall_k` / `top_k` ranked
memories. A v0.12.3 server can add session-neighbourhood context; this package requests it
separately and leaves it out, so the prompt history holds ranked memories only.

**`event_time`, `warnings` or `truncated` missing.** These are v0.12.3 features; older servers
ignore the field or leave it out of the answer. Check the version with `curl
http://localhost:3000/health`.

**`DakeraMemory` is rejected by a LangChain chain.** On LangChain 1.x, `BaseMemory` lives in
`langchain-classic`: `pip install "langchain-dakera[classic]"`. LCEL chains (above) need no
memory class at all.

**`entities.extract(...)` returns `[]`.** Named entity types need the GLiNER model, which the
server downloads from Hugging Face on first use; a server without network access serves its
rule-based extractor only.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy src/
pytest tests/                                          # unit tests, no server needed
DAKERA_API_URL=http://localhost:3000 pytest tests/     # + live tests against a server
```

See [CONTRIBUTING.md](https://github.com/Dakera-AI/dakera-langchain/blob/main/CONTRIBUTING.md)
for the full workflow.

## Related packages

| Package | What it is |
|---|---|
| [`dakera`](https://pypi.org/project/dakera/) · [`@dakera-ai/dakera`](https://www.npmjs.com/package/@dakera-ai/dakera) · [`dakera-client`](https://crates.io/crates/dakera-client) · [`dakera-go`](https://github.com/Dakera-AI/dakera-go) | Python, TypeScript, Rust and Go SDKs |
| **[`langchain-dakera`](https://pypi.org/project/langchain-dakera/)** | **LangChain (Python): memory, vector store, sessions, knowledge graph** |
| [`@dakera-ai/langchain`](https://www.npmjs.com/package/@dakera-ai/langchain) | LangChain.js: memory, vector store, sessions, knowledge graph |
| [`llamaindex-dakera`](https://pypi.org/project/llamaindex-dakera/) | LlamaIndex: memory block, vector store, memory store |
| [`autogen-dakera`](https://pypi.org/project/autogen-dakera/) | AutoGen: `Memory` for `AssistantAgent`, sessions, knowledge graph |
| [`crewai-dakera`](https://pypi.org/project/crewai-dakera/) | CrewAI: storage for a crew's memories |
| [`strands-dakera`](https://pypi.org/project/strands-dakera/) | Strands Agents: memory tool and `MemoryStore` |
| [`@dakera-ai/ai-sdk`](https://www.npmjs.com/package/@dakera-ai/ai-sdk) | Vercel AI SDK: memory middleware and tools |

## Links

- [dakera.ai](https://dakera.ai) · [Documentation](https://dakera.ai/docs) ·
  [LangChain integration guide](https://dakera.ai/integrations/langchain.html) ·
  [LoCoMo benchmark](https://dakera.ai/benchmark)
- Server image: `ghcr.io/dakera-ai/dakera` ·
  [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy) (Docker Compose, Kubernetes) ·
  [dakera-helm](https://github.com/Dakera-AI/dakera-helm)
- [dakera-mcp](https://github.com/Dakera-AI/dakera-mcp) (MCP server) ·
  [dakera-cli](https://github.com/Dakera-AI/dakera-cli) (`dk` CLI)
- [Issues](https://github.com/Dakera-AI/dakera-langchain/issues)

## License

[MIT](https://github.com/Dakera-AI/dakera-langchain/blob/main/LICENSE) © [Dakera AI](https://dakera.ai)
