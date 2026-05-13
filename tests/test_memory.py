"""Tests for DakeraMemory (LangChain integration)."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_dakera import DakeraMemory


@pytest.fixture
def memory():
    with patch("langchain_dakera.memory.DakeraClient") as MC:
        mock_client = MagicMock()
        MC.return_value = mock_client
        m = DakeraMemory(api_url="http://localhost:3000", api_key="test",
                         agent_id="agent-1", recall_k=3)
        m._client = mock_client
        yield m, mock_client


def test_memory_variables(memory):
    m, _ = memory
    assert m.memory_variables == ["history"]


def test_load_memory_variables_recalls(memory):
    m, mock_client = memory
    mock_client.recall.return_value = [{"content": "User likes Python"}]
    result = m.load_memory_variables({"input": "What do I like?"})
    assert "User likes Python" in result["history"]


def test_save_context_stores(memory):
    m, mock_client = memory
    m.save_context({"input": "Hello"}, {"output": "Hi there!"})
    mock_client.store_memory.assert_called_once()
    call_kwargs = mock_client.store_memory.call_args
    assert "Human: Hello" in call_kwargs[1]["content"]
    assert "AI: Hi there!" in call_kwargs[1]["content"]


def test_clear_is_noop(memory):
    m, mock_client = memory
    m.clear()
    mock_client.forget.assert_not_called()
