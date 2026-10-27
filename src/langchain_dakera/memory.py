"""DakeraMemory — LangChain memory backed by the Dakera AI memory platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dakera import DakeraClient, RecalledMemory, RecallResponse
from pydantic import Field

try:
    from langchain_core.memory import BaseMemory
except ImportError:
    from pydantic import BaseModel as BaseMemory


class DakeraMemory(BaseMemory):
    """LangChain conversational memory backed by Dakera AI.

    Supports memory types, tags, TTL, batch operations, and importance scoring.
    """

    api_url: str = Field(description="Dakera API base URL.")
    api_key: str = Field(default="", description="Dakera API key.")
    agent_id: str = Field(description="Agent identifier for memory storage.")
    recall_k: int = Field(default=5)
    min_importance: float = Field(default=0.0)
    memory_key: str = Field(default="history")
    input_key: str | None = Field(default=None)
    importance: float = Field(default=0.7)
    memory_type: str = Field(default="episodic")
    tags: list[str] = Field(default_factory=list)
    ttl_seconds: int | None = Field(default=None)
    session_id: str | None = Field(default=None)
    _client: DakeraClient | None = None
    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def _ranked(result: RecallResponse, top_k: int) -> list[RecalledMemory]:
        """The ranked results of a recall, at most ``top_k`` of them.

        Recalls ask for ``neighborhood_mode="separate"``, so a v0.12.3+ server keeps
        the session neighbourhood context out of ``memories``; a server up to
        v0.12.2 appends it inline (unmarked), so the list is cut at ``top_k`` too.
        """
        return result.ranked[:top_k]

    def model_post_init(self, __context: Any) -> None:
        self._client = DakeraClient(self.api_url, api_key=self.api_key)

    def _get_client(self) -> DakeraClient:
        if self._client is None:
            raise RuntimeError(
                "DakeraMemory: client was not initialized; model_post_init may not have run"
            )
        return self._client

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def _get_query(self, inputs: dict[str, Any]) -> str:
        if self.input_key is not None:
            return str(inputs[self.input_key])
        return str(next(iter(inputs.values()), ""))

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = self._get_query(inputs)
        if not query:
            return {self.memory_key: ""}
        result = self._get_client().recall(
            self.agent_id,
            query=query,
            top_k=self.recall_k,
            min_importance=self.min_importance if self.min_importance > 0 else None,
            neighborhood_mode="separate",
        )
        history = "\n".join(m.content for m in self._ranked(result, self.recall_k))
        return {self.memory_key: history}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        human = str(next(iter(inputs.values()), ""))
        ai = str(next(iter(outputs.values()), ""))
        kwargs: dict[str, Any] = {
            "memory_type": self.memory_type,
            "importance": self.importance,
        }
        if self.tags:
            kwargs["tags"] = self.tags
        if self.ttl_seconds is not None:
            kwargs["ttl_seconds"] = self.ttl_seconds
        if self.session_id:
            kwargs["session_id"] = self.session_id
        self._get_client().store_memory(
            self.agent_id,
            content=f"Human: {human}\nAI: {ai}",
            **kwargs,
        )

    def clear(self) -> None:
        """No-op: Dakera memories are persistent by design."""

    def store(
        self,
        content: str,
        *,
        memory_type: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        event_time: int | float | str | datetime | None = None,
    ) -> Any:
        """Store a memory directly with full control over parameters.

        ``event_time`` is when the memory's event happened (Unix seconds, an RFC 3339
        string or a timezone-aware ``datetime``); the server (v0.12.3+) uses it for
        recency and ``since`` / ``until`` instead of the ingest time.
        """
        kwargs: dict[str, Any] = {}
        if memory_type:
            kwargs["memory_type"] = memory_type
        if importance is not None:
            kwargs["importance"] = importance
        if tags:
            kwargs["tags"] = tags
        if ttl_seconds is not None:
            kwargs["ttl_seconds"] = ttl_seconds
        if metadata:
            kwargs["metadata"] = metadata
        if event_time is not None:
            kwargs["event_time"] = event_time
        if self.session_id:
            kwargs["session_id"] = self.session_id
        return self._get_client().store_memory(self.agent_id, content=content, **kwargs)

    def recall(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_importance: float | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories with filtering by type, importance and tags.

        Returns at most ``top_k`` (default 5) ranked memories, best first; ``score`` is
        the server's ranking score (``rank_score`` on v0.12.3+). ``tags`` keeps
        memories carrying at least one of the given tags; ``rerank=False`` skips the
        cross-encoder reranking (server default: on).
        """
        top_k = 5 if top_k is None else top_k
        kwargs: dict[str, Any] = {"top_k": top_k, "neighborhood_mode": "separate"}
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if memory_type:
            kwargs["memory_type"] = memory_type
        if tags:
            kwargs["tags"] = tags
        if rerank is not None:
            kwargs["rerank"] = rerank
        result = self._get_client().recall(self.agent_id, query=query, **kwargs)
        return [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "score": m.score,
                "metadata": m.metadata,
            }
            for m in self._ranked(result, top_k)
        ]

    def batch_recall(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        min_importance: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Batch recall across multiple queries."""
        client = self._get_client()
        results = []
        for q in queries:
            kwargs: dict[str, Any] = {"top_k": top_k, "neighborhood_mode": "separate"}
            if min_importance is not None:
                kwargs["min_importance"] = min_importance
            result = client.recall(self.agent_id, query=q, **kwargs)
            results.append(
                [
                    {"id": m.id, "content": m.content, "importance": m.importance}
                    for m in self._ranked(result, top_k)
                ]
            )
        return results

    def batch_forget(self, memory_ids: list[str]) -> None:
        """Forget multiple memories by ID."""
        for mid in memory_ids:
            self._get_client().forget(self.agent_id, mid)

    def forget(self, memory_id: str) -> None:
        """Forget a single memory by ID."""
        self._get_client().forget(self.agent_id, memory_id)

    def update_importance(self, memory_id: str, importance: float) -> None:
        """Update the importance score of a memory."""
        self._get_client().update_importance(
            self.agent_id, memory_ids=[memory_id], importance=importance
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_importance: float | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across agent memories (``tags``: at least one of them)."""
        kwargs: dict[str, Any] = {"top_k": top_k}
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if tags:
            kwargs["tags"] = tags
        result = self._get_client().search_memories(self.agent_id, query=query, **kwargs)
        return [
            {
                "id": m.get("id", ""),
                "content": m.get("content", ""),
                "importance": m.get("importance", 0.0),
                "score": m.get("score", 0.0),
            }
            for m in result
        ]

    def consolidate(
        self, memory_ids: list[str] | None = None, *, target_type: str | None = None
    ) -> Any:
        """Merge memories into one new memory; the merged sources are forgotten.

        ``memory_ids`` picks the memories (at least two); without it the server
        merges up to 5 memories of the agent's most common type. ``target_type`` is
        the memory type of the merged memory.
        """
        kwargs: dict[str, Any] = {}
        if memory_ids is not None:
            if len(memory_ids) < 2:
                raise ValueError("consolidate() needs at least two memory ids")
            kwargs["memory_ids"] = memory_ids
        if target_type is not None:
            kwargs["target_type"] = target_type
        return self._get_client().consolidate(self.agent_id, **kwargs)
