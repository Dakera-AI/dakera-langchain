"""DakeraEntityExtractor — named entity extraction via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraEntityExtractor:
    """Extract and manage named entities with Dakera's NER pipeline.

    The server combines a rule-based extractor with GLiNER zero-shot NER. Turn on
    extraction for an agent's memories with :meth:`configure` on the agent's memory
    namespace (``_dakera_agent_<agent_id>``); the entities of each stored memory are
    then read with :meth:`memory_entities`.
    """

    def __init__(self, api_url: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)

    def extract(self, text: str, entity_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Extract entities from text (``POST /v1/memories/extract``, needs a Write key).

        Without ``entity_types`` the server uses its rule-based extractor (and GLiNER
        only if its model is already cached); naming types, e.g. ``["person",
        "organization", "location"]``, loads GLiNER on demand (a ~782 MB download on
        first use). Returns ``[{"type", "value", "score"}, ...]``.
        """
        result = self._client.extract_entities(text, entity_types=entity_types)
        return [
            {"type": e.entity_type, "value": e.value, "score": e.score} for e in result.entities
        ]

    def extract_from_text(
        self,
        text: str,
        namespace: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Extract structured content from text using a pluggable provider (EXT-1)."""
        result = self._client.extract_text(
            text, namespace=namespace, provider=provider, model=model
        )
        return {
            "entities": result.entities,
            "provider": result.provider,
            "model": result.model,
            "duration_ms": result.duration_ms,
        }

    def memory_entities(self, memory_id: str) -> list[dict[str, Any]]:
        """Get entities associated with a specific memory."""
        result = self._client.memory_entities(memory_id)
        return [
            {"type": e.entity_type, "value": e.value, "score": e.score} for e in result.entities
        ]

    def configure(
        self,
        namespace: str,
        extract_entities: bool = True,
        entity_types: list[str] | None = None,
    ) -> None:
        """Turn entity extraction on or off for a namespace.

        For an agent's memories use its memory namespace, ``_dakera_agent_<agent_id>``
        (it exists once the agent has stored a memory). ``entity_types`` are the
        GLiNER labels to extract, e.g. ``["person", "organization"]``.
        """
        self._client.configure_namespace_ner(
            namespace, extract_entities=extract_entities, entity_types=entity_types
        )

    def get_extractor(self, namespace: str) -> dict[str, Any]:
        """Get the extraction provider configured for a namespace.

        Reads ``GET /v1/namespaces/{namespace}/extractor``. This replaces
        ``list_providers()``, which called ``GET /v1/extract/providers``, a route
        no Dakera server serves (removed from the dakera SDK in 0.13.0).
        """
        return self._client.get_namespace_extractor(namespace)
