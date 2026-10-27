"""Advanced memory features with LangChain and Dakera.

Importance, tags, memory types, TTL and event time on store; filtered and
reranked recall; search; importance updates; consolidating memories into one;
forgetting; and agent statistics.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/advanced_memory.py
"""

import os
import uuid
from datetime import datetime, timezone

from langchain_dakera import DakeraAgentTools, DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")
agent_id = f"langchain-advanced-{uuid.uuid4().hex[:8]}"

memory = DakeraMemory(api_url=api_url, api_key=api_key, agent_id=agent_id)

print("--- Store with importance, tags, type, TTL and event time ---")
memory.store("The user said hello.", importance=0.2, ttl_seconds=3600)  # expires in 1 h
allergy = memory.store(
    "The user is allergic to peanuts.",
    importance=0.95,
    memory_type="semantic",
    tags=["health"],
)
memory.store(
    "The user's birthday is March 15.",
    importance=0.8,
    memory_type="semantic",
    tags=["personal"],
)
# event_time (server v0.12.3+): when it happened, for history written after the fact.
# Recency and time filters use it instead of the time of storage.
trip = memory.store(
    "The user hiked the Tour du Mont Blanc.",
    importance=0.6,
    tags=["personal", "travel"],
    event_time=datetime(2025, 7, 14, 9, 0, tzinfo=timezone.utc),
)
lisbon = memory.store("The user visited Lisbon for a conference.", tags=["travel"])
porto = memory.store("The user visited Porto after Lisbon.", tags=["travel"])
print(f"Stored 6 memories for agent {agent_id}")

print("\n--- Recall: the important things ---")
for m in memory.recall("What should I know about the user?", min_importance=0.7, top_k=5):
    print(f"  [{m['importance']:.2f}] score={m['score']:.3f} {m['content']}")

print("\n--- Recall: only memories tagged 'travel', without reranking ---")
for m in memory.recall("Where has the user been?", tags=["travel"], rerank=False):
    print(f"  [{m['importance']:.2f}] {m['content']}")

print("\n--- Search ---")
for m in memory.search("birthday", top_k=2):
    print(f"  score={m['score']:.3f} {m['content']}")

print("\n--- Update importance ---")
memory.update_importance(lisbon["id"], 0.9)
print(f"  {lisbon['id']} -> 0.9")

print("\n--- Consolidate the two Portugal memories into one ---")
# Memories with a TTL or user metadata (event_time included) are never consolidated.
merged = memory.consolidate([lisbon["id"], porto["id"]], target_type="semantic")
print(f"  {merged['memories_removed']} memories -> {merged['consolidated_memory']['content']!r}")

print("\n--- Forget ---")
memory.forget(allergy["id"])
print(f"  forgot {allergy['id']}")

print("\n--- Agent statistics ---")
stats = DakeraAgentTools(api_url=api_url, api_key=api_key, agent_id=agent_id).stats()
print(f"  {stats['total_memories']} memories, by type: {stats['memories_by_type']}")
