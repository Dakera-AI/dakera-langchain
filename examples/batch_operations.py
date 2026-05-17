"""Batch operations with LangChain and Dakera.

Demonstrates batch recall (multiple queries at once) and batch
storage for efficient bulk operations.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    python batch_operations.py
"""

import os

from langchain_dakera import DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-batch-demo",
    importance=0.7,
)

print("Storing batch of memories...")
conversations = [
    ("What's your return policy?", "30-day returns for unopened items."),
    ("Do you ship internationally?", "Yes, we ship to 40+ countries."),
    ("What payment methods?", "We accept Visa, Mastercard, and PayPal."),
    ("Is there a loyalty program?", "Yes! Earn 1 point per dollar spent."),
    ("What about gift cards?", "Available in $25, $50, and $100 denominations."),
]

for user_input, assistant_output in conversations:
    memory.save_context({"input": user_input}, {"output": assistant_output})

print(f"Stored {len(conversations)} memories.")

print("\n--- Batch recall ---")
queries = [
    "shipping and delivery",
    "payment options",
    "returns and refunds",
]

results = memory.batch_search(queries, limit=2)
for query, matches in zip(queries, results):
    print(f"\n  Query: '{query}'")
    for m in matches:
        print(f"    [{m['importance']:.1f}] {m['content'][:50]}")
