"""Advanced memory features with LangChain and Dakera.

Demonstrates importance levels, tags, memory types, TTL, and
the consolidation/deduplication utilities.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    python advanced_memory.py
"""

import os

from langchain_dakera import DakeraMemory
from langchain_dakera.knowledge_graph import DakeraKnowledgeGraph

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-advanced-demo",
)

print("--- Storing with different importance levels ---")
memory.save_context(
    {"input": "casual chat"},
    {"output": "Just a greeting."},
    importance=0.3,
)
memory.save_context(
    {"input": "I'm allergic to peanuts"},
    {"output": "Noted — peanut allergy."},
    importance=0.95,
)
memory.save_context(
    {"input": "My birthday is March 15"},
    {"output": "I'll remember your birthday!"},
    importance=0.8,
    tags=["personal", "birthday"],
)

print("--- Recall with importance threshold ---")
results = memory.load_memory_variables(
    {"input": "What important things do you know?"},
    min_importance=0.7,
)
print(f"High-importance memories:\n{results['history']}")

print("\n--- Agent tools: stats ---")
from langchain_dakera.agents import DakeraAgentTools

agent = DakeraAgentTools(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-advanced-demo",
)
stats = agent.stats()
print(f"Total memories: {stats}")

print("\n--- Deduplication ---")
kg = DakeraKnowledgeGraph(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-advanced-demo",
)
dedup_result = kg.deduplicate()
print(f"Deduplication: {dedup_result}")
