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

    def query(self, **kwargs: Any) -> dict[str, Any]:
        """Query the knowledge graph with filter parameters.

        Keyword args are passed to ``knowledge_query`` (e.g. root_id, edge_type,
        min_weight, max_depth, limit).
        """
        result = self._client.knowledge_query(self._agent_id, **kwargs)
        return {
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in result.edges
            ],
            "node_count": result.node_count,
            "edge_count": result.edge_count,
        }

    def find_path(self, from_id: str, to_id: str) -> dict[str, Any]:
        """Find shortest path between two memory IDs."""
        result = self._client.knowledge_path(
            self._agent_id, from_id=from_id, to_id=to_id
        )
        return {
            "path": result.path,
            "hop_count": result.hop_count,
            "from_id": result.from_id,
            "to_id": result.to_id,
        }

    def link(
        self, source_id: str, target_id: str, edge_type: str = "linked_by"
    ) -> dict[str, Any]:
        """Link two memories in the knowledge graph."""
        result = self._client.memory_link(
            source_id=source_id, target_id=target_id, edge_type=edge_type
        )
        return {
            "edge_id": result.edge.id,
            "source_id": result.edge.source_id,
            "target_id": result.edge.target_id,
            "edge_type": result.edge.edge_type.value,
        }

    def export(self, format: str = "json") -> dict[str, Any]:
        """Export the knowledge graph for this agent."""
        result = self._client.knowledge_export(self._agent_id, format=format)
        return {
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in result.edges
            ],
            "node_count": result.node_count,
            "edge_count": result.edge_count,
        }

    def summarize(self) -> dict[str, Any]:
        """Generate a summary of the knowledge graph."""
        return self._client.summarize(self._agent_id)

    def deduplicate(self) -> dict[str, Any]:
        """Find and merge duplicate entities in the knowledge graph."""
        return self._client.deduplicate(self._agent_id)

    def build(self) -> dict[str, Any]:
        """Build/rebuild the knowledge graph from agent memories."""
        return self._client.knowledge_graph(self._agent_id)

    def full_graph(self) -> dict[str, Any]:
        """Get the complete knowledge graph with all relationships."""
        return self._client.full_knowledge_graph(self._agent_id)
