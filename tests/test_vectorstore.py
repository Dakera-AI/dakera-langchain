"""Tests for DakeraVectorStore (LangChain integration)."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_dakera import DakeraVectorStore


@pytest.fixture
def store():
    with patch("langchain_dakera.vectorstore.DakeraClient") as MC, \
         patch("langchain_dakera.vectorstore.AsyncDakeraClient"):
        mock_client = MagicMock()
        MC.return_value = mock_client
        vs = DakeraVectorStore(api_url="http://localhost:3000", api_key="test", namespace="test-ns")
        vs._client = mock_client
        yield vs, mock_client


def test_add_texts_upserts(store):
    vs, mock_client = store
    ids = vs.add_texts(["Hello world"], ids=["id-1"])
    assert ids == ["id-1"]
    mock_client.upsert_text.assert_called_once()


def test_similarity_search_returns_documents(store):
    vs, mock_client = store
    mock_result = MagicMock()
    mock_result.text = "Hello world"
    mock_result.metadata = {}
    mock_result.score = 0.95
    mock_result.id = "id-1"
    mock_client.query_text.return_value = MagicMock(results=[mock_result])
    docs = vs.similarity_search("hello", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello world"


def test_embeddings_property_returns_none(store):
    vs, _ = store
    assert vs.embeddings is None
