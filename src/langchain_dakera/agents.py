"""DakeraAgentTools — agent statistics and management via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient


class DakeraAgentTools:
    """Inspect agent state, statistics, and memory health.

    Provides visibility into memory counts, session history, and
    agent-level analytics.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id

    def stats(self) -> dict[str, Any]:
        """Get agent memory statistics (counts, types, importance distribution)."""
        return self._client.agent_stats(self._agent_id)

    def memories(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """List the agent's memories, newest first (at most ``limit``).

        The server's derived sentence sub-memories are not listed.
        """
        mems = self._client.agent_memories(self._agent_id, limit=limit)
        return [
            {
                "id": m.get("id", ""),
                "content": m.get("content", ""),
                "importance": m.get("importance", 0.0),
                "memory_type": m.get("memory_type", ""),
                "metadata": m.get("metadata"),
                "created_at": m.get("created_at"),
            }
            for m in mems
        ]

    def sessions(self, active_only: bool = False) -> list[dict[str, Any]]:
        """List agent sessions."""
        sess = self._client.agent_sessions(self._agent_id, active_only=active_only)
        return [
            {
                "id": s.get("session_id", s.get("id", "")),
                "started_at": s.get("started_at"),
                "ended_at": s.get("ended_at"),
            }
            for s in sess
        ]

    def import_memories(self, memories: Any, *, format: str = "jsonl") -> Any:
        """Bulk import memories into this agent (``POST /v1/import``).

        ``jsonl`` takes memory records as :meth:`export_memories` returns them (each
        with ``id``, ``content``, ``agent_id``, ``created_at`` and
        ``last_accessed_at``); ``mem0`` takes ``[{"memory": "..."}, ...]``, ``zep``
        ``[{"content": "..."}, ...]`` and ``csv`` the CSV text. Returns the import
        job status (``imported_count``, ``skipped_count``, ``errors``).
        """
        return self._client.import_memories(memories, format=format, agent_id=self._agent_id)

    def export_memories(self, *, format: str = "jsonl") -> list[dict[str, Any]]:
        """Export all memories of this agent as dicts (``GET /v1/export``)."""
        result = self._client.export_memories(format=format, agent_id=self._agent_id)
        return result.data
