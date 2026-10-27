"""DakeraSessionManager — conversation session tracking via Dakera."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient

# Tag of the sentence sub-memories the server derives from a stored memory.
_DERIVED_TAG = "dakera-curated"


class DakeraSessionManager:
    """Manage conversation sessions for agent memory grouping.

    Sessions allow grouping memories by conversation, enabling retrieval of
    context from specific interactions.
    """

    def __init__(self, api_url: str, agent_id: str, api_key: str = "") -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._agent_id = agent_id
        self._active_session_id: str | None = None

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    def start(
        self, metadata: dict[str, Any] | None = None, *, idle_timeout_secs: int | None = None
    ) -> str:
        """Start a new session. Returns the session ID.

        The server ends a session after a period without activity (default 4 h,
        ``ended_reason: "idle"``). ``idle_timeout_secs`` sets this session's own
        timeout (``0`` = never); :meth:`touch` keeps an idle session open.
        """
        result = self._client.start_session(
            self._agent_id, metadata=metadata, idle_timeout_secs=idle_timeout_secs
        )
        sid: str = str(result.get("session_id", result.get("id", "")))
        self._active_session_id = sid
        return sid

    def end(self, summary: str | None = None) -> None:
        """End the active session with an optional summary."""
        if self._active_session_id is None:
            raise RuntimeError("No active session to end")
        self._client.end_session(self._active_session_id, summary)
        self._active_session_id = None

    def touch(self, session_id: str | None = None) -> str:
        """Record activity on a session so the server does not end it for inactivity.

        Defaults to the active session. Returns the session state, ``"active"`` or
        ``"ended"`` (touching never re-opens an ended session).
        """
        sid = session_id or self._active_session_id
        if sid is None:
            raise RuntimeError("No active session to touch")
        return self._client.touch_session(sid).session_state

    def get(self, session_id: str) -> dict[str, Any]:
        """Get details of a specific session."""
        result = self._client.get_session(session_id)
        return {
            "id": result.get("session_id", result.get("id", "")),
            "agent_id": result.get("agent_id", ""),
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "metadata": result.get("metadata"),
            "memory_count": result.get("memory_count", 0),
            "summary": result.get("summary"),
            "ended_reason": result.get("ended_reason"),
        }

    def list_sessions(self, active_only: bool = False) -> list[dict[str, Any]]:
        """List sessions for this agent."""
        raw: Any = self._client.list_sessions(self._agent_id, active_only=active_only)
        # GET /v1/sessions answers {"sessions": [...], "total": n}; the SDK passes
        # that envelope through, so unwrap it (a bare list is accepted too).
        sessions = raw.get("sessions", []) if isinstance(raw, dict) else raw
        return [
            {
                "id": s.get("session_id", s.get("id", "")),
                "agent_id": s.get("agent_id", ""),
                "started_at": s.get("started_at"),
                "ended_at": s.get("ended_at"),
                "memory_count": s.get("memory_count", 0),
                "ended_reason": s.get("ended_reason"),
            }
            for s in sessions
        ]

    def memories(self, session_id: str, *, include_derived: bool = False) -> list[dict[str, Any]]:
        """Get the memories stored in a specific session, oldest first.

        The server also lists the sentence sub-memories it derives from them (tagged
        ``dakera-curated``); they are left out unless ``include_derived=True``.
        """
        mems = self._client.session_memories(session_id)
        if not include_derived:
            mems = [m for m in mems if _DERIVED_TAG not in (m.get("tags") or [])]
        return [
            {
                "id": m.get("id", ""),
                "content": m.get("content", ""),
                "importance": m.get("importance", 0.0),
                "metadata": m.get("metadata"),
            }
            for m in mems
        ]

    def __enter__(self) -> DakeraSessionManager:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._active_session_id:
            self.end()
