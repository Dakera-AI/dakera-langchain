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
    """LangChain conversational memory backed by Dakera AI."""

    api_url: str = Field(description="Dakera API base URL.")
    api_key: str = Field(default="", description="Dakera API key.")
    agent_id: str = Field(description="Agent identifier for memory storage.")
    recall_k: int = Field(default=5)
    min_importance: float = Field(default=0.0)
    memory_key: str = Field(default="history")
    input_key: str | None = Field(default=None)
    importance: float = Field(default=0.7)
    _client: DakeraClient | None = None
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._client = DakeraClient(self.api_url, api_key=self.api_key)

    @property
    def _dakera_client(self) -> DakeraClient:
        if self._client is None:
            raise RuntimeError("DakeraMemory: client was not initialized; model_post_init may not have run")
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
        memories = self._dakera_client.recall(
            self.agent_id, query=query, top_k=self.recall_k,
            min_importance=self.min_importance if self.min_importance > 0 else None)
        history = "\n".join(m.content for m in memories.memories)
        return {self.memory_key: history}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        human = str(next(iter(inputs.values()), ""))
        ai = str(next(iter(outputs.values()), ""))
        self._dakera_client.store_memory(self.agent_id,
                                         content=f"Human: {human}\nAI: {ai}",
                                         memory_type="episodic", importance=self.importance)

    def clear(self) -> None:
        """No-op: Dakera memories are persistent by design."""
