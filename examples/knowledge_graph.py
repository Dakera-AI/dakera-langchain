"""Knowledge graph operations with LangChain and Dakera.

Links an agent's memories, then queries, traverses, builds, exports and
summarizes the resulting graph.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/knowledge_graph.py
"""

import os
import uuid

from langchain_dakera import DakeraKnowledgeGraph, DakeraMemory

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")
agent_id = f"langchain-kg-{uuid.uuid4().hex[:8]}"

memory = DakeraMemory(api_url=api_url, api_key=api_key, agent_id=agent_id, importance=0.9)
project = memory.store("Project Alpha is a Python service led by Sarah Chen.")
team = memory.store("Sarah Chen works with Bob Smith on the Project Alpha backend.")
stack = memory.store("Project Alpha runs on FastAPI, PostgreSQL and Redis.")

kg = DakeraKnowledgeGraph(api_url=api_url, api_key=api_key, agent_id=agent_id)

print("--- Link memories ---")
kg.link(project["id"], team["id"], label="same project")
kg.link(team["id"], stack["id"], label="same project")
print("  project -> team -> stack")

print("\n--- Query edges from the project memory ---")
result = kg.query(root_id=project["id"], max_depth=2)
print(f"  {result['node_count']} nodes, {result['edge_count']} edges")
for edge in result["edges"]:
    print(f"    {edge['source_id']} -[{edge['edge_type']}]-> {edge['target_id']}")

print("\n--- Shortest path from project to stack ---")
path = kg.find_path(project["id"], stack["id"])
print(f"  {path['hop_count']} hops: {' -> '.join(path['path'])}")

print("\n--- Graph around the team memory ---")
graph = kg.build(team["id"], depth=2)
print(f"  root: {graph['root']['memory']['content']}")

print("\n--- Export ---")
export = kg.export()
print(f"  {export['node_count']} nodes, {export['edge_count']} edges")

print("\n--- Summarize two memories into a new one ---")
summary = kg.summarize([project["id"], stack["id"]])
print(f"  {summary['source_count']} sources -> {summary['summary_memory']['id']}")
