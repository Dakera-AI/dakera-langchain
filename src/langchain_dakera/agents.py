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

    def memories(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List agent memories with pagination."""
        result = self._client.agent_memories(self._agent_id, limit=limit, offset=offset)
        return [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "memory_type": m.memory_type,
                "tags": m.tags,
                "created_at": m.created_at,
            }
            for m in result.memories
        ]

    def sessions(self, active_only: bool = False) -> list[dict[str, Any]]:
        """List agent sessions."""
        result = self._client.agent_sessions(self._agent_id, active_only=active_only)
        return [
            {"id": s.id, "started_at": s.started_at, "ended_at": s.ended_at}
            for s in result.sessions
        ]

    def import_memories(self, memories: list[dict[str, Any]]) -> Any:
        """Bulk import memories for this agent."""
        return self._client.import_memories(self._agent_id, memories=memories)

    def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories for this agent."""
        result = self._client.export_memories(self._agent_id)
        return result.memories
