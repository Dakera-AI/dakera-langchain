"""Live tests against a real Dakera server.

Skipped unless ``DAKERA_API_URL`` is set (CI runs them against the pinned server image):

    DAKERA_API_URL=http://localhost:3300 DAKERA_API_KEY=test-key pytest tests/test_integration.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from langchain_dakera import (
    DakeraAgentTools,
    DakeraKnowledgeGraph,
    DakeraMemory,
    DakeraSessionManager,
    DakeraVectorStore,
)

API_URL = os.environ.get("DAKERA_API_URL", "")
API_KEY = os.environ.get("DAKERA_API_KEY", "test-key")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_URL, reason="DAKERA_API_URL not set"),
]


def _agent() -> str:
    return f"lc-it-{uuid.uuid4().hex[:10]}"


def _memory(agent_id: str, **kwargs: object) -> DakeraMemory:
    return DakeraMemory(api_url=API_URL, api_key=API_KEY, agent_id=agent_id, **kwargs)


def test_recall_of_an_agent_without_memories_is_empty():
    mem = _memory(_agent())
    assert mem.recall("anything") == []
    assert mem.load_memory_variables({"input": "anything"}) == {"history": ""}


def test_session_recall_returns_at_most_top_k():
    agent = _agent()
    sessions = DakeraSessionManager(api_url=API_URL, api_key=API_KEY, agent_id=agent)
    with sessions:
        sid = sessions.active_session_id
        assert sid
        mem = _memory(agent, session_id=sid, recall_k=1)
        for turn in ("I have a dog named Rex", "Rex is three years old", "Rex likes the park"):
            mem.save_context({"input": turn}, {"output": "Noted."})
        assert sessions.touch() == "active"
        # Neighbourhood context of the session stays out of the ranked results.
        recalled = mem.recall("Tell me about my dog", top_k=1)
        assert len(recalled) == 1
        assert isinstance(recalled[0]["score"], float)
        assert len(mem.load_memory_variables({"input": "dog"})["history"].split("\nAI: ")) == 2
        stored = sessions.memories(sid)
        assert len(stored) == 3
        assert all("dakera-curated" not in (m.get("tags") or []) for m in stored)
    info = sessions.get(sid)
    assert info["ended_at"] is not None
    assert info["ended_reason"] == "client"


def test_store_with_event_time():
    stored = _memory(_agent()).store("Alice moved to Lisbon", event_time="2024-05-01T10:00:00Z")
    assert stored["metadata"]["_dakera_event_time"] == 1714557600


def test_consolidate_given_memories():
    mem = _memory(_agent())
    first = mem.store("Alice moved to Lisbon")
    second = mem.store("Alice works as a designer in Lisbon")
    merged = mem.consolidate([first["id"], second["id"]])
    assert sorted(merged["source_memory_ids"]) == sorted([first["id"], second["id"]])


def test_export_import_round_trip():
    source = DakeraAgentTools(api_url=API_URL, api_key=API_KEY, agent_id=_agent())
    _memory(source._agent_id).store("Bob prefers tea over coffee")
    exported = source.export_memories()
    assert [m["content"] for m in exported] == ["Bob prefers tea over coffee"]
    target = DakeraAgentTools(api_url=API_URL, api_key=API_KEY, agent_id=_agent())
    assert target.import_memories(exported).imported_count == 1
    assert (
        target.import_memories([{"memory": "Bob rides a bike"}], format="mem0").imported_count == 1
    )
    assert len(target.memories()) == 2


def test_knowledge_graph_link_and_path():
    agent = _agent()
    mem = _memory(agent)
    a = mem.store("Carol leads Project Alpha")["id"]
    b = mem.store("Project Alpha ships in May")["id"]
    kg = DakeraKnowledgeGraph(api_url=API_URL, api_key=API_KEY, agent_id=agent)
    assert kg.link(a, b, label="same project")["edge_type"] == "linked_by"
    assert kg.find_path(a, b)["path"] == [a, b]


def test_vector_store_similarity_search():
    store = DakeraVectorStore(
        api_url=API_URL, api_key=API_KEY, namespace=f"lc-it-{uuid.uuid4().hex[:8]}"
    )
    store.add_texts(
        ["Rust provides memory safety", "Python is readable"],
        metadatas=[{"lang": "rust"}, {"lang": "python"}],
    )
    docs = store.similarity_search("memory safe language", k=1)
    assert docs[0].page_content == "Rust provides memory safety"
    assert docs[0].metadata["lang"] == "rust"
