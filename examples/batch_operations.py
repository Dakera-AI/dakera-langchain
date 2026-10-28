"""Batch operations with LangChain and Dakera.

Stores a batch of conversation turns, recalls for several queries in one
call, and forgets several memories at once.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/batch_operations.py
"""

import os
import uuid

from langchain_dakera import DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id=f"langchain-batch-{uuid.uuid4().hex[:8]}",
    importance=0.7,
    tags=["support"],  # added to every stored turn
)

conversations = [
    ("What's your return policy?", "30-day returns for unopened items."),
    ("Do you ship internationally?", "Yes, we ship to 40+ countries."),
    ("Which payment methods do you take?", "We accept Visa, Mastercard, and PayPal."),
    ("Is there a loyalty program?", "Yes! Earn 1 point per dollar spent."),
    ("Do you sell gift cards?", "Available in $25, $50, and $100 denominations."),
]
for user_input, assistant_output in conversations:
    memory.save_context({"input": user_input}, {"output": assistant_output})
print(f"Stored {len(conversations)} conversation turns.")

print("\n--- Batch recall ---")
queries = ["shipping abroad", "paying by card", "sending items back"]
results = memory.batch_recall(queries, top_k=1)
for query, matches in zip(queries, results):
    print(f"  {query!r}:")
    for m in matches:
        print(f"    [{m['importance']:.1f}] {m['content'].splitlines()[-1]}")

print("\n--- Batch forget ---")
to_forget = [m["id"] for matches in results for m in matches]
memory.batch_forget(to_forget)
print(f"  Forgot {len(to_forget)} memories.")
