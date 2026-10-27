"""Calls into the dakera SDK checked against its real signatures.

``create_autospec`` mocks fail on a method the SDK does not have or on
arguments its signature does not accept, which plain ``MagicMock`` hides.
"""

import asyncio
from unittest.mock import create_autospec, patch

import pytest
from dakera import DakeraClient
from dakera.models import FullTextSearchResult, HybridSearchResult, IndexStats

from langchain_dakera import (
    DakeraEntityExtractor,
    DakeraNamespaceManager,
    DakeraSessionManager,
    DakeraVectorStore,
)


def _spec_client() -> DakeraClient:
    return create_autospec(DakeraClient, instance=True)


def test_entity_extractor_get_extractor_reads_namespace_config():
    client = _spec_client()
    client.get_namespace_extractor.return_value = {"provider": "gliner"}
    with patch("langchain_dakera.entities.DakeraClient", return_value=client):
        ex = DakeraEntityExtractor(api_url="http://localhost:3000")
    assert ex.get_extractor("docs") == {"provider": "gliner"}
    client.get_namespace_extractor.assert_called_once_with("docs")


def test_entity_extractor_has_no_list_providers():
    # GET /v1/extract/providers is served by no Dakera server (SDK 0.13.0 removed it).
    assert not hasattr(DakeraEntityExtractor, "list_providers")
    assert not hasattr(DakeraClient, "list_extract_providers")


def test_list_sessions_unwraps_server_envelope():
    client = _spec_client()
    client.list_sessions.return_value = {
        "sessions": [
            {"id": "sess_1", "agent_id": "a", "started_at": 1, "memory_count": 2},
        ],
        "total": 1,
    }
    with patch("langchain_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    result = mgr.list_sessions()
    assert result == [
        {
            "id": "sess_1",
            "agent_id": "a",
            "started_at": 1,
            "ended_at": None,
            "memory_count": 2,
            "ended_reason": None,
        }
    ]
    client.list_sessions.assert_called_once_with("a", active_only=False)


def test_list_sessions_accepts_bare_list():
    client = _spec_client()
    client.list_sessions.return_value = [{"id": "sess_2", "agent_id": "a"}]
    with patch("langchain_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert [s["id"] for s in mgr.list_sessions(active_only=True)] == ["sess_2"]


def test_start_session_reads_session_id():
    client = _spec_client()
    client.start_session.return_value = {"id": "sess_3", "agent_id": "a"}
    with patch("langchain_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert mgr.start() == "sess_3"
    assert mgr.active_session_id == "sess_3"


def test_namespace_stats_maps_index_stats():
    client = _spec_client()
    client.get_index_stats.return_value = IndexStats.from_dict(
        {"vector_count": 7, "dimension": 384, "index_type": "hnsw",
         "estimated_storage_bytes": 1024}
    )
    with patch("langchain_dakera.namespaces.DakeraClient", return_value=client):
        mgr = DakeraNamespaceManager(api_url="http://localhost:3000")
    assert mgr.stats("docs") == {
        "total_vectors": 7,
        "dimensions": 384,
        "index_type": "hnsw",
        "memory_usage_bytes": None,
        "disk_usage_bytes": 1024,
    }


@pytest.fixture
def store():
    client = _spec_client()
    with patch("langchain_dakera.vectorstore.DakeraClient", return_value=client), patch(
        "langchain_dakera.vectorstore.AsyncDakeraClient"
    ):
        s = DakeraVectorStore(api_url="http://localhost:3000", namespace="docs")
    yield s, client


def test_hybrid_search_sends_text_query_without_vector(store):
    s, client = store
    client.hybrid_search.return_value = [HybridSearchResult(id="d1", score=0.8)]
    docs = s.hybrid_search("python web", k=3, alpha=0.7)
    client.hybrid_search.assert_called_once_with(
        "docs", query="python web", top_k=3, filter=None, vector_weight=0.7
    )
    assert docs[0].metadata == {"score": 0.8, "id": "d1"}


def test_ahybrid_search_uses_server_side_embedding(store):
    s, client = store
    client.hybrid_search.return_value = [HybridSearchResult(id="d2", score=0.5)]
    docs = asyncio.run(s.ahybrid_search("rust safety", k=2))
    client.hybrid_search.assert_called_once_with(
        "docs", query="rust safety", top_k=2, filter=None, vector_weight=0.5
    )
    assert docs[0].metadata["id"] == "d2"


def test_fulltext_search_signature(store):
    s, client = store
    client.fulltext_search.return_value = [FullTextSearchResult(id="d3", score=5.1)]
    docs = s.fulltext_search("google", k=1)
    client.fulltext_search.assert_called_once_with("docs", query="google", top_k=1, filter=None)
    assert docs[0].metadata["score"] == 5.1


def test_summarize_sends_memory_ids():
    from langchain_dakera import DakeraKnowledgeGraph

    client = _spec_client()
    client.summarize.return_value = {"summary_memory": {"id": "s"}, "source_count": 2}
    with patch("langchain_dakera.knowledge_graph.DakeraClient", return_value=client):
        kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", agent_id="a")
    kg.summarize(["m1", "m2"])
    client.summarize.assert_called_once_with("a", memory_ids=["m1", "m2"], target_type=None)
    # The server needs two ids and has no dry run (it always stores the summary).
    with pytest.raises(ValueError):
        kg.summarize(["m1"])
    with pytest.raises(TypeError):
        kg.summarize(["m1", "m2"], dry_run=True)  # type: ignore[call-arg]


def test_link_and_build_pass_the_agent_and_seed():
    from dakera.models import GraphLinkResponse

    from langchain_dakera import DakeraKnowledgeGraph

    client = _spec_client()
    client.memory_link.return_value = GraphLinkResponse.from_dict(
        {"from_id": "m1", "to_id": "m2", "edge_type": "linked_by"}
    )
    client.knowledge_graph.return_value = {"root": {}, "total_nodes": 1}
    with patch("langchain_dakera.knowledge_graph.DakeraClient", return_value=client):
        kg = DakeraKnowledgeGraph(api_url="http://localhost:3000", agent_id="a")
    assert kg.link("m1", "m2", label="why") == {
        "from_id": "m1",
        "to_id": "m2",
        "edge_type": "linked_by",
    }
    client.memory_link.assert_called_once_with("m1", "m2", agent_id="a", label="why")
    kg.build("m1")
    client.knowledge_graph.assert_called_once_with("a", memory_id="m1", depth=None)


def test_memory_recall_and_search_forward_tags():
    from dakera.models import RecallResponse

    from langchain_dakera import DakeraMemory

    client = _spec_client()
    client.recall.return_value = RecallResponse.from_dict({"memories": []})
    client.search_memories.return_value = []
    with patch("langchain_dakera.memory.DakeraClient", return_value=client):
        mem = DakeraMemory(api_url="http://localhost:3000", agent_id="a")
        assert mem.recall("q", tags=["work"]) == []
        assert mem.search("q", tags=["work"]) == []
    client.recall.assert_called_once_with(
        "a", query="q", top_k=5, neighborhood_mode="separate", tags=["work"]
    )
    client.search_memories.assert_called_once_with("a", query="q", top_k=10, tags=["work"])


# --- Server v0.12.3 / dakera 0.14.1 ---------------------------------------------


def _memory_with(client: DakeraClient):
    from langchain_dakera import DakeraMemory

    with patch("langchain_dakera.memory.DakeraClient", return_value=client):
        return DakeraMemory(api_url="http://localhost:3000", agent_id="a", recall_k=2)


def _recall_with_inline_context():
    from dakera import RecallResponse

    # A v0.12.3 server in neighborhood_mode="inline" appends session context after
    # the ranked results; a v0.12.2 server does the same without marking it.
    return RecallResponse.from_dict(
        {
            "memories": [
                {"id": "m1", "content": "first", "rank_score": 0.9},
                {"id": "m2", "content": "second", "rank_score": 0.8},
                {"id": "n1", "content": "context", "score": 0.99, "retrieval": "neighborhood"},
                {"id": "m3", "content": "unmarked extra", "rank_score": 0.1},
            ]
        }
    )


def test_recall_returns_at_most_top_k_ranked_results():
    client = _spec_client()
    client.recall.return_value = _recall_with_inline_context()
    mem = _memory_with(client)
    result = mem.recall("q", top_k=2, rerank=False)
    assert [m["id"] for m in result] == ["m1", "m2"]
    assert [m["score"] for m in result] == [0.9, 0.8]
    client.recall.assert_called_once_with(
        "a", query="q", top_k=2, neighborhood_mode="separate", rerank=False
    )


def test_load_memory_variables_keeps_neighbourhood_out_of_history():
    client = _spec_client()
    client.recall.return_value = _recall_with_inline_context()
    mem = _memory_with(client)
    assert mem.load_memory_variables({"input": "q"}) == {"history": "first\nsecond"}
    client.recall.assert_called_once_with(
        "a", query="q", top_k=2, min_importance=None, neighborhood_mode="separate"
    )


def test_batch_recall_caps_each_query():
    client = _spec_client()
    client.recall.return_value = _recall_with_inline_context()
    mem = _memory_with(client)
    assert [[m["id"] for m in r] for r in mem.batch_recall(["a", "b"], top_k=1)] == [
        ["m1"],
        ["m1"],
    ]
    client.recall.assert_called_with("a", query="b", top_k=1, neighborhood_mode="separate")


def test_store_forwards_event_time():
    client = _spec_client()
    client.store_memory.return_value = {"id": "m1"}
    mem = _memory_with(client)
    mem.store("met Bob", event_time="2026-01-02T03:04:05Z")
    client.store_memory.assert_called_once_with(
        "a", content="met Bob", event_time="2026-01-02T03:04:05Z"
    )


def test_consolidate_sends_memory_ids():
    client = _spec_client()
    client.consolidate.return_value = {"consolidated_memory": {"id": "c"}}
    mem = _memory_with(client)
    mem.consolidate(["m1", "m2"], target_type="semantic")
    client.consolidate.assert_called_once_with("a", memory_ids=["m1", "m2"], target_type="semantic")
    with pytest.raises(ValueError):
        mem.consolidate(["m1"])


def test_session_start_idle_timeout_and_touch():
    from dakera.models import SessionTouchResponse

    client = _spec_client()
    client.start_session.return_value = {"id": "urn:chat:42", "agent_id": "a"}
    client.touch_session.return_value = SessionTouchResponse.from_dict(
        {"session": {"id": "urn:chat:42"}, "session_state": "active"}
    )
    with patch("langchain_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert mgr.start(idle_timeout_secs=0) == "urn:chat:42"
    client.start_session.assert_called_once_with("a", metadata=None, idle_timeout_secs=0)
    assert mgr.touch() == "active"
    client.touch_session.assert_called_once_with("urn:chat:42")


def test_session_memories_leave_out_derived_sub_memories():
    client = _spec_client()
    client.session_memories.return_value = [
        {"id": "m1", "content": "Human: hi\nAI: hello", "tags": []},
        {"id": "mem_s" + "0" * 24, "content": "Human: hi", "tags": ["dakera-curated"]},
    ]
    with patch("langchain_dakera.sessions.DakeraClient", return_value=client):
        mgr = DakeraSessionManager(api_url="http://localhost:3000", agent_id="a")
    assert [m["id"] for m in mgr.memories("s")] == ["m1"]
    assert len(mgr.memories("s", include_derived=True)) == 2


def test_agent_import_export_pass_agent_and_format():
    from dakera.models import MemoryExportResponse, MemoryImportResponse

    from langchain_dakera import DakeraAgentTools

    client = _spec_client()
    client.import_memories.return_value = MemoryImportResponse.from_dict(
        {"imported": 1, "skipped": 0, "status": "done"}
    )
    client.export_memories.return_value = MemoryExportResponse(
        data=[{"id": "m1"}], format="jsonl", count=1
    )
    with patch("langchain_dakera.agents.DakeraClient", return_value=client):
        tools = DakeraAgentTools(api_url="http://localhost:3000", agent_id="a")
    tools.import_memories([{"memory": "likes tea"}], format="mem0")
    client.import_memories.assert_called_once_with(
        [{"memory": "likes tea"}], format="mem0", agent_id="a"
    )
    assert tools.export_memories() == [{"id": "m1"}]
    client.export_memories.assert_called_once_with(format="jsonl", agent_id="a")
