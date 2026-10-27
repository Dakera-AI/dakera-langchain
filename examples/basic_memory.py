"""Basic conversational memory with LangChain and Dakera.

Each conversation turn is stored as a Dakera memory; on the next turn the
memories most relevant to the input are recalled and injected into the prompt
as ``history``. The memories live on the server, so they survive restarts.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/basic_memory.py
"""

import os
import uuid

from langchain_dakera import DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id=f"langchain-basic-{uuid.uuid4().hex[:8]}",  # a fresh agent per run
    recall_k=3,  # memories recalled per turn
    importance=0.8,  # importance of the stored turns
)

# What a chain does after each turn: store the exchange.
memory.save_context(
    {"input": "My favorite color is blue."},
    {"output": "Got it! I'll remember that your favorite color is blue."},
)
memory.save_context(
    {"input": "I work as a software engineer at Acme Corp."},
    {"output": "Nice! Software engineering at Acme Corp — noted."},
)

# What a chain does before each turn: recall what is relevant to the new input.
variables = memory.load_memory_variables({"input": "Where do I work?"})
print("Recalled for 'Where do I work?':")
print(variables["history"])
