"""DakeraEntityExtractor — named entity extraction via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraEntityExtractor:
    """Extract and manage named entities from text using Dakera's NER pipeline.

    Entities are automatically linked to the knowledge graph and can be
    used for structured retrieval.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text.

        Returns a list of entities with type, value, and confidence.
        """
        result = self._client.extract_entities(self._agent_id, text=text)
        return [
            {"type": e.entity_type, "value": e.value, "confidence": e.confidence}
            for e in result.entities
        ]

    def extract_from_text(self, text: str) -> dict[str, Any]:
        """Extract structured content from text (entities + key-value pairs)."""
        return self._client.extract_text(self._agent_id, text=text)

    def memory_entities(self, memory_id: str) -> list[dict[str, Any]]:
        """Get entities associated with a specific memory."""
        result = self._client.memory_entities(self._agent_id, memory_id=memory_id)
        return [
            {"type": e.entity_type, "value": e.value, "confidence": e.confidence}
            for e in result.entities
        ]

    def configure(self, entity_types: list[str] | None = None, **kwargs: Any) -> None:
        """Configure entity extraction settings for the agent's namespace."""
        self._client.configure_namespace_ner(self._agent_id, entity_types=entity_types, **kwargs)

    def list_providers(self) -> list[dict[str, Any]]:
        """List available entity extraction providers."""
        return self._client.list_extract_providers()
