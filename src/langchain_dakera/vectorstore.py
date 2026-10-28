"""DakeraVectorStore — LangChain vector store with server-side embedding, hybrid and BM25 search."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from dakera import AsyncDakeraClient, DakeraClient
from dakera.models import TextSearchResult
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class DakeraVectorStore(VectorStore):
    """LangChain vector store backed by a Dakera namespace — server-side embedding.

    Texts are embedded by the Dakera server (``upsert-text`` / ``query-text``), so no
    LangChain ``Embeddings`` object is needed; the server also indexes them for BM25.
    Supports similarity search (with metadata filters), :meth:`hybrid_search`
    (semantic + BM25), :meth:`fulltext_search` (BM25), :meth:`batch_search`,
    :meth:`delete`, and the async variants.

    Args:
        api_url: Dakera server URL, e.g. ``http://localhost:3000``.
        namespace: Namespace holding the documents (created on first write).
        api_key: Dakera API key (empty when the server runs without auth).
        embedding_model: Server-side embedding model for writes and queries (e.g.
            ``"bge-large"``, ``"minilm"``); default: the server's / namespace's model.
            Use the same model for every write and query of a namespace.
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
        self._client.upsert_text(self._namespace, docs, model=self._embedding_model)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        response = self._client.query_text(
            self._namespace,
            text=query,
            top_k=k,
            filter=filter,
            include_text=True,
            model=self._embedding_model,
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
            self._namespace,
            text=query,
            top_k=k,
            filter=filter,
            include_text=True,
            model=self._embedding_model,
        )
        return [
            (
                Document(page_content=r.text or "", metadata={**(r.metadata or {}), "id": r.id}),
                r.score,
            )
            for r in response.results
        ]

    def _semantic_pool(
        self, query: str, fetch_k: int, filter: dict[str, Any] | None
    ) -> list[TextSearchResult]:
        """The ``fetch_k`` documents nearest to ``query`` (server-side embedding), with text."""
        response = self._client.query_text(
            self._namespace,
            text=query,
            top_k=fetch_k,
            filter=filter,
            include_text=True,
            model=self._embedding_model,
        )
        return list(response.results)

    def _bm25(self, query: str, top_k: int) -> dict[str, float]:
        """BM25 scores of the namespace's best ``top_k`` keyword matches, best first.

        Not filtered: the server's BM25 index of ``upsert-text`` documents keeps no
        metadata, so a filtered full-text query matches nothing. Callers apply the
        filter through the semantic leg instead.
        """
        hits = self._client.fulltext_search(self._namespace, query=query, top_k=top_k)
        return {h.id: float(h.score) for h in hits}

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        alpha: float = 0.5,
        fetch_k: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Semantic + keyword (BM25) search, weighted by ``alpha``.

        Dakera's ``/hybrid`` route blends in vector similarity only when the request
        carries a query vector, which a store relying on server-side embedding does
        not have (the route alone is BM25-only, whatever the weight). The two signals
        are therefore fused here: the ``fetch_k`` (default ``max(4 * k, 20)``)
        documents semantically nearest to ``query`` that match ``filter`` are the
        candidates, and each is scored ``alpha * vector_score + (1 - alpha) *
        text_score``, where ``vector_score`` is the cosine similarity (clipped at 0)
        and ``text_score`` the document's BM25 score divided by the best BM25 score
        among the candidates.

        Args:
            query: Search query text.
            k: Number of results.
            filter: Metadata filter expression.
            alpha: Weight of semantic similarity: ``1.0`` ranks by vector similarity
                only, ``0.0`` by BM25 only (among the candidates).
            fetch_k: Number of semantic candidates; raise it when exact keyword
                matches should be found further from the query's meaning.

        Returns:
            Documents, best first; ``metadata`` carries ``score``, ``vector_score``,
            ``text_score`` and ``id`` besides the document's own metadata.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        fetch_k = fetch_k if fetch_k is not None else max(4 * k, 20)
        pool = self._semantic_pool(query, fetch_k, filter)
        # Without a filter the candidates' keyword matches are among the namespace's
        # best fetch_k; with one, look further, as other documents may rank above them.
        bm25 = self._bm25(query, fetch_k if filter is None else 5 * fetch_k)
        best = max((bm25[r.id] for r in pool if r.id in bm25), default=0.0)
        scored = []
        for r in pool:
            vector_score = max(float(r.score), 0.0)
            text_score = bm25.get(r.id, 0.0) / best if best > 0 else 0.0
            score = alpha * vector_score + (1.0 - alpha) * text_score
            scored.append((score, vector_score, text_score, r))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Document(
                page_content=r.text or "",
                metadata={
                    **(r.metadata or {}),
                    "score": score,
                    "vector_score": vector_score,
                    "text_score": text_score,
                    "id": r.id,
                },
            )
            for score, vector_score, text_score, r in scored[:k]
        ]

    def fulltext_search(
        self,
        query: str,
        k: int = 10,
        *,
        filter: dict[str, Any] | None = None,
        fetch_k: int | None = None,
    ) -> list[Document]:
        """BM25 keyword search, best first (``metadata["score"]`` is the BM25 score).

        The server's full-text route answers ids and scores only, so each hit's text
        and metadata are read from the ``fetch_k`` (default ``max(4 * k, 20)``)
        documents semantically nearest to ``query``. Without ``filter``, a hit
        outside them keeps an empty ``page_content`` and ``metadata["text_found"] =
        False`` (raise ``fetch_k`` to avoid it). With ``filter``, only hits among the
        matching candidates are returned, as the BM25 index cannot apply the filter.
        """
        fetch_k = fetch_k if fetch_k is not None else max(4 * k, 20)
        bm25 = self._bm25(query, k if filter is None else 5 * max(k, fetch_k))
        if not bm25:
            return []
        texts = {r.id: r for r in self._semantic_pool(query, fetch_k, filter)}
        if filter is not None:
            bm25 = {i: score for i, score in bm25.items() if i in texts}
        docs = []
        for doc_id, score in list(bm25.items())[:k]:
            found = texts.get(doc_id)
            docs.append(
                Document(
                    page_content=(found.text or "") if found else "",
                    metadata={
                        **((found.metadata or {}) if found else {}),
                        "score": score,
                        "id": doc_id,
                        "text_found": found is not None,
                    },
                )
            )
        return docs

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
                self._namespace,
                text=q,
                top_k=k,
                filter=filter,
                include_text=True,
                model=self._embedding_model,
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
        await self._async_client.upsert_text(self._namespace, docs, model=self._embedding_model)
        return ids

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        response = await self._async_client.query_text(
            self._namespace,
            text=query,
            top_k=k,
            filter=filter,
            include_text=True,
            model=self._embedding_model,
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
        fetch_k: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async :meth:`hybrid_search` (runs it in a worker thread)."""
        return await asyncio.to_thread(
            self.hybrid_search, query, k, filter=filter, alpha=alpha, fetch_k=fetch_k
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by id (from the vector and the BM25 index).

        Returns ``True`` when the request succeeded; ``ids=None`` deletes nothing.
        """
        if not ids:
            return None
        self._client.delete(self._namespace, ids=list(ids))
        return True

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
