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
        """List agent memories with pagination."""
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

    def import_memories(self, memories: list[dict[str, Any]]) -> Any:
        """Bulk import memories for this agent."""
        return self._client.import_memories(memories, agent_id=self._agent_id)

    def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories for this agent."""
        result = self._client.export_memories(agent_id=self._agent_id)
        return result.data
