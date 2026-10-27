"""Tests for DakeraMemory (LangChain integration)."""

from unittest.mock import MagicMock, patch

import pytest
from dakera import RecallResponse

from langchain_dakera import DakeraMemory


@pytest.fixture
def memory():
    with patch("langchain_dakera.memory.DakeraClient") as MC:
        mock_client = MagicMock()
        MC.return_value = mock_client
        m = DakeraMemory(
            api_url="http://localhost:3000", api_key="test", agent_id="agent-1", recall_k=3
        )
        m._client = mock_client
        yield m, mock_client


def test_memory_variables(memory):
    m, _ = memory
    assert m.memory_variables == ["history"]


def test_load_memory_variables_recalls(memory):
    m, mock_client = memory
    mock_client.recall.return_value = RecallResponse.from_dict(
        {"memories": [{"id": "m-1", "content": "User likes Python", "rank_score": 0.9}]}
    )
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


def test_recall_collapses_derived_sentences_into_their_source(memory):
    # The server derives sentence sub-memories from a stored memory (metadata
    # ``source_memory_id``) and may recall them next to it: they repeat its text.
    m, mock_client = memory
    mock_client.recall.return_value = RecallResponse.from_dict(
        {
            "memories": [
                {"memory": {"id": "m-1", "content": "Human: I work at Acme\nAI: Noted."}},
                {
                    "memory": {
                        "id": "mem_s1",
                        "content": "I work at Acme",
                        "metadata": {"source_memory_id": "m-1"},
                    }
                },
                {
                    "memory": {
                        "id": "mem_s2",
                        "content": "My favorite color is blue.",
                        "metadata": {"source_memory_id": "m-2"},
                    }
                },
            ]
        }
    )
    history = m.load_memory_variables({"input": "Where do I work?"})["history"]
    assert history == "Human: I work at Acme\nAI: Noted.\nMy favorite color is blue."


def test_is_a_langchain_base_memory_in_a_conversation_chain(memory):
    # LangChain 1.x keeps BaseMemory and ConversationChain in langchain-classic.
    classic = pytest.importorskip("langchain_classic")
    from langchain_classic.base_memory import BaseMemory
    from langchain_classic.chains import ConversationChain
    from langchain_core.language_models.fake import FakeListLLM

    assert classic
    m, mock_client = memory
    assert isinstance(m, BaseMemory)
    mock_client.recall.return_value = RecallResponse.from_dict(
        {"memories": [{"id": "m-1", "content": "Human: I build chatbots\nAI: Nice."}]}
    )
    chain = ConversationChain(llm=FakeListLLM(responses=["You build chatbots."]), memory=m)
    assert chain.predict(input="What do I build?") == "You build chatbots."
    assert mock_client.store_memory.call_args.kwargs["content"] == (
        "Human: What do I build?\nAI: You build chatbots."
    )
