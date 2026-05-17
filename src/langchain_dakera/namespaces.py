"""DakeraNamespaceManager — namespace CRUD operations via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraNamespaceManager:
    """Manage Dakera namespaces for data isolation and configuration.

    Namespaces provide logical separation for different data domains,
    each with independent vector indexes and configuration.
    """

    def __init__(self, api_url: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)

    def create(
        self,
        name: str,
        *,
        dimensions: int | None = None,
        index_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new namespace."""
        result = self._client.create_namespace(
            name, dimensions=dimensions, index_type=index_type, metadata=metadata, **kwargs
        )
        return {
            "name": result.name,
            "dimensions": result.dimensions,
            "vector_count": result.vector_count,
        }

    def get(self, name: str) -> dict[str, Any]:
        """Get namespace details."""
        result = self._client.get_namespace(name)
        return {
            "name": result.name,
            "dimensions": result.dimensions,
            "vector_count": result.vector_count,
        }

    def list_namespaces(self) -> list[dict[str, Any]]:
        """List all namespaces."""
        namespaces = self._client.list_namespaces()
        return [
            {"name": ns.name, "dimensions": ns.dimensions, "vector_count": ns.vector_count}
            for ns in namespaces
        ]

    def configure(self, name: str, **kwargs: Any) -> None:
        """Update namespace configuration."""
        self._client.configure_namespace(name, **kwargs)

    def delete(self, name: str) -> None:
        """Delete a namespace and all its data."""
        self._client.delete_namespace(name)

    def stats(self, name: str) -> dict[str, Any]:
        """Get index statistics for a namespace."""
        result = self._client.get_index_stats(name)
        return {
            "total_vectors": result.total_vectors,
            "dimensions": result.dimensions,
            "index_type": result.index_type,
            "memory_usage_bytes": result.memory_usage_bytes,
            "disk_usage_bytes": result.disk_usage_bytes,
        }
