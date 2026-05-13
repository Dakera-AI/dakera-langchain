# langchain-dakera

[![PyPI](https://img.shields.io/pypi/v/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-dakera)](https://pypi.org/project/langchain-dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LangChain integration for the [Dakera AI](https://dakera.ai) memory platform.**

Drop-in LangChain components using Dakera's server-side embedding — no local model required.

## Installation

```bash
pip install langchain-dakera
```

## Quick Start

```python
from langchain_dakera import DakeraMemory, DakeraVectorStore

# Conversational memory
memory = DakeraMemory(api_url="https://your-dakera-instance.com", api_key="dk-...", agent_id="my-agent")

# RAG vector store (no local embedding model needed)
vectorstore = DakeraVectorStore(api_url="https://your-dakera-instance.com", api_key="dk-...", namespace="docs")
vectorstore.add_texts(["Document content..."])
results = vectorstore.similarity_search("my query", k=4)
```

## Links

- [Dakera Documentation](https://docs.dakera.ai/integrations/langchain)
- [Dakera AI](https://dakera.ai)
