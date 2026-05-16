"""DakeraKnowledgeGraph — knowledge graph operations via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraKnowledgeGraph:
    """Interface to Dakera's knowledge graph for entity relationships.

    Supports graph traversal, querying, linking memories to entities,
    path finding, and export.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id

    def query(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Query the knowledge graph with natural language."""
        result = self._client.knowledge_query(self._agent_id, query=query, **kwargs)
        return {"nodes": result.nodes, "edges": result.edges}

    def traverse(
        self,
        entity_id: str,
        *,
        depth: int = 2,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Traverse the graph from an entity node."""
        result = self._client.knowledge_path(
            self._agent_id, source=entity_id, depth=depth, direction=direction
        )
        return {"nodes": result.nodes, "edges": result.edges}

    def link(self, memory_id: str, entity_id: str, relation: str = "relates_to") -> None:
        """Link a memory to an entity in the knowledge graph."""
        self._client.memory_link(
            self._agent_id, memory_id=memory_id, entity_id=entity_id, relation=relation
        )

    def export(self) -> dict[str, Any]:
        """Export the full knowledge graph for this agent."""
        result = self._client.knowledge_export(self._agent_id)
        return {"nodes": result.nodes, "edges": result.edges}

    def summarize(self) -> dict[str, Any]:
        """Generate a summary of the knowledge graph."""
        return self._client.summarize(self._agent_id)

    def deduplicate(self) -> dict[str, Any]:
        """Find and merge duplicate entities in the knowledge graph."""
        return self._client.deduplicate(self._agent_id)

    def build(self) -> dict[str, Any]:
        """Build/rebuild the knowledge graph from agent memories."""
        result = self._client.knowledge_graph(self._agent_id)
        return {"nodes": result.nodes, "edges": result.edges}

    def full_graph(self) -> dict[str, Any]:
        """Get the complete knowledge graph with all relationships."""
        result = self._client.full_knowledge_graph(self._agent_id)
        return {"nodes": result.nodes, "edges": result.edges}
