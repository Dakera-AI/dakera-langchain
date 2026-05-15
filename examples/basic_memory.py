"""Basic conversational memory with LangChain and Dakera.

Stores each conversation turn in Dakera and recalls relevant context
on the next turn using semantic search.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    export DAKERA_API_KEY="dk-..."          # optional
    pip install langchain-dakera langchain-core
    python basic_memory.py
"""

import os

from langchain_dakera import DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-demo",
    recall_k=3,
    importance=0.8,
)

memory.save_context(
    {"input": "My favorite color is blue."},
    {"output": "Got it! I'll remember that your favorite color is blue."},
)
memory.save_context(
    {"input": "I work as a software engineer at Acme Corp."},
    {"output": "Nice! Software engineering at Acme Corp — noted."},
)

result = memory.load_memory_variables({"input": "What do you know about me?"})
print("Recalled memories:")
print(result["history"])
