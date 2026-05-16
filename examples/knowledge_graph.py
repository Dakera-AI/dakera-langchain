"""Knowledge graph operations with LangChain and Dakera.

Demonstrates entity extraction, graph querying, and traversal.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    python knowledge_graph.py
"""

import os

from langchain_dakera import DakeraMemory
from langchain_dakera.knowledge_graph import DakeraKnowledgeGraph

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-kg-demo",
    importance=0.9,
)

memory.save_context(
    {"input": "Tell me about the project."},
    {"output": "Project Alpha uses Python and is led by Sarah Chen."},
)
memory.save_context(
    {"input": "What about the team?"},
    {"output": "Sarah Chen works with Bob Smith on the backend."},
)
memory.save_context(
    {"input": "What tech stack?"},
    {"output": "They use FastAPI, PostgreSQL, and Redis."},
)

kg = DakeraKnowledgeGraph(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-kg-demo",
)

graph = kg.export()
print(f"Knowledge graph: {graph['node_count']} nodes, {graph['edge_count']} edges")

results = kg.query(max_depth=3, limit=10)
print(f"\nQuery results: {results['edge_count']} edges found")
for edge in results["edges"][:5]:
    print(f"  {edge}")
