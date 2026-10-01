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
        {"id": "sess_1", "agent_id": "a", "started_at": 1, "ended_at": None, "memory_count": 2}
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
    kg.summarize(["m1", "m2"], dry_run=True)
    client.summarize.assert_called_once_with(
        "a", memory_ids=["m1", "m2"], target_type=None, dry_run=True
    )
    with pytest.raises(ValueError):
        kg.summarize([])
