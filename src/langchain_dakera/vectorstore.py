"""DakeraVectorStore — LangChain vector store with hybrid search and fulltext."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from typing import Any

from dakera import AsyncDakeraClient, DakeraClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class DakeraVectorStore(VectorStore):
    """LangChain vector store backed by Dakera AI — server-side embedding.

    Supports vector search, hybrid search (vector + BM25), fulltext search,
    batch queries, and filter expressions.
    """

    def __init__(
        self,
        api_url: str,
        namespace: str,
        api_key: str = "",
        embedding_model: str | None = None,
    ) -> None:
        self._client = DakeraClient(api_url, api_key=api_key)
        self._async_client = AsyncDakeraClient(api_url, api_key=api_key)
        self._namespace = namespace
        self._embedding_model = embedding_model

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        fill = metadatas or [{} for _ in texts_list]
        docs: list[Any] = [
            {"id": did, "text": t, "metadata": m or {}} for did, t, m in zip(ids, texts_list, fill)
        ]
        self._client.upsert_text(self._namespace, docs)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        response = self._client.query_text(
            self._namespace, text=query, top_k=k, filter=filter, include_text=True
        )
        return [
            Document(
                page_content=r.text or "",
                metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
            )
            for r in response.results
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        response = self._client.query_text(
            self._namespace, text=query, top_k=k, filter=filter, include_text=True
        )
        return [
            (
                Document(page_content=r.text or "", metadata={**(r.metadata or {}), "id": r.id}),
                r.score,
            )
            for r in response.results
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Combined vector + BM25 search with configurable weighting.

        Args:
            query: Search query text.
            k: Number of results.
            filter: Metadata filter expression.
            alpha: Balance between vector (1.0) and keyword (0.0) search.
        """
        results = self._client.hybrid_search(
            self._namespace, query=query, top_k=k, filter=filter, vector_weight=alpha, **kwargs
        )
        return [
            Document(
                page_content=r.content or "",
                metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
            )
            for r in results
        ]

    def fulltext_search(
        self,
        query: str,
        k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """BM25-only fulltext search."""
        results = self._client.fulltext_search(
            self._namespace, query=query, top_k=k, filter=filter
        )
        return [
            Document(
                page_content=r.content or "",
                metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
            )
            for r in results
        ]

    def batch_search(
        self,
        queries: list[str],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
    ) -> list[list[Document]]:
        """Run multiple similarity searches in a single batch."""
        results = []
        for q in queries:
            response = self._client.query_text(
                self._namespace, text=q, top_k=k, filter=filter, include_text=True
            )
            results.append(
                [
                    Document(
                        page_content=r.text or "",
                        metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
                    )
                    for r in response.results
                ]
            )
        return results

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        fill = metadatas or [{} for _ in texts_list]
        docs: list[Any] = [
            {"id": did, "text": t, "metadata": m or {}} for did, t, m in zip(ids, texts_list, fill)
        ]
        await self._async_client.upsert_text(self._namespace, docs)
        return ids

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        response = await self._async_client.query_text(
            self._namespace, text=query, top_k=k, filter=filter, include_text=True
        )
        return [
            Document(
                page_content=r.text or "",
                metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
            )
            for r in response.results
        ]

    async def ahybrid_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Async combined vector + BM25 search."""
        results = await self._async_client.hybrid_search(
            self._namespace, query=query, top_k=k, filter=filter, vector_weight=alpha, **kwargs
        )
        return [
            Document(
                page_content=r.content or "",
                metadata={**(r.metadata or {}), "score": r.score, "id": r.id},
            )
            for r in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        api_url: str = "",
        namespace: str = "",
        api_key: str = "",
        **kwargs: Any,
    ) -> DakeraVectorStore:
        store = cls(api_url=api_url, namespace=namespace, api_key=api_key, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings | None = None,
        api_url: str = "",
        namespace: str = "",
        api_key: str = "",
        **kwargs: Any,
    ) -> DakeraVectorStore:
        return cls.from_texts(
            [d.page_content for d in documents],
            embedding=embedding,
            metadatas=[d.metadata for d in documents],
            api_url=api_url,
            namespace=namespace,
            api_key=api_key,
            **kwargs,
        )

    @property
    def embeddings(self) -> Embeddings | None:
        return None

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: score
