"""DakeraMemory — LangChain memory backed by the Dakera AI memory platform."""

from __future__ import annotations

from typing import Any

from dakera import DakeraClient
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
        memories = self._get_client().recall(
            self.agent_id,
            query=query,
            top_k=self.recall_k,
            min_importance=self.min_importance if self.min_importance > 0 else None,
        )
        history = "\n".join(m.content for m in memories.memories)
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
    ) -> Any:
        """Store a memory directly with full control over parameters."""
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
    ) -> list[dict[str, Any]]:
        """Recall memories with filtering by type and importance."""
        kwargs: dict[str, Any] = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if memory_type:
            kwargs["memory_type"] = memory_type
        result = self._get_client().recall(self.agent_id, query=query, **kwargs)
        return [
            {
                "id": m.id,
                "content": m.content,
                "importance": m.importance,
                "metadata": m.metadata,
            }
            for m in result.memories
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
            kwargs: dict[str, Any] = {"top_k": top_k}
            if min_importance is not None:
                kwargs["min_importance"] = min_importance
            result = client.recall(self.agent_id, query=q, **kwargs)
            results.append(
                [
                    {"id": m.id, "content": m.content, "importance": m.importance}
                    for m in result.memories
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
    ) -> list[dict[str, Any]]:
        """Semantic search across agent memories."""
        kwargs: dict[str, Any] = {"top_k": top_k}
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
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

    def consolidate(self) -> Any:
        """Deduplicate and consolidate agent memories."""
        return self._get_client().consolidate(self.agent_id)
