from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client.models import Filter

from src.indexer.config import QDRANT_COLLECTION
from src.indexer.pipelines import get_qwen_embedder
from src.indexer.qdrant_db import get_qdrant_client


@dataclass(slots=True)
class SearchResult:
    id: str | int
    score: float
    payload: dict[str, Any] | None
    vector: list[float] | dict[str, list[float]] | None = None


@dataclass(slots=True)
class FileSearchResult:
    file_name: str
    source_path: str
    score: float


def search_similar(
    prompt: str,
    k: int = 5,
    *,
    collection_name: str = QDRANT_COLLECTION,
    query_filter: Filter | None = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    embedder = get_qwen_embedder()
    query_vector = embedder.embed(prompt).squeeze(0).tolist()

    client = get_qdrant_client()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=with_payload,
        with_vectors=with_vectors,
        score_threshold=score_threshold,
    )

    return [
        SearchResult(
            id=point.id,
            score=point.score,
            payload=point.payload,
            vector=point.vector,
        )
        for point in response.points
    ]


def aggregate_file_results(results: list[SearchResult]) -> list[FileSearchResult]:
    best_by_path: dict[str, FileSearchResult] = {}

    for result in results:
        payload = result.payload or {}
        source_path = payload.get("source_path")
        if not source_path:
            continue

        absolute_path = str(Path(source_path).resolve())
        current = best_by_path.get(absolute_path)
        if current is not None and current.score >= result.score:
            continue

        best_by_path[absolute_path] = FileSearchResult(
            file_name=payload.get("file_name") or Path(absolute_path).name,
            source_path=absolute_path,
            score=result.score,
        )

    return sorted(best_by_path.values(), key=lambda item: item.score, reverse=True)


def search_similar_files(
    prompt: str,
    k: int = 5,
    *,
    collection_name: str = QDRANT_COLLECTION,
    query_filter: Filter | None = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    score_threshold: float | None = None,
    embedding_fetch_multiplier: int = 4,
) -> list[FileSearchResult]:
    if embedding_fetch_multiplier <= 0:
        raise ValueError("embedding_fetch_multiplier must be greater than 0")

    embedding_results = search_similar(
        prompt,
        k=max(k * embedding_fetch_multiplier, k),
        collection_name=collection_name,
        query_filter=query_filter,
        with_payload=with_payload,
        with_vectors=with_vectors,
        score_threshold=score_threshold,
    )
    return aggregate_file_results(embedding_results)[:k]
