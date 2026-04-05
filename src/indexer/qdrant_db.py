from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.indexer.config import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from src.indexer.schemas import EmbeddingRecord, make_point_payload


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection() -> None:
    client = get_qdrant_client()

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def upsert_records(records: list[EmbeddingRecord]) -> None:
    if not records:
        return

    client = get_qdrant_client()

    points = [
        PointStruct(
            id=record.id,
            vector=record.vector,
            payload=make_point_payload(record),
        )
        for record in records
    ]

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )


def _source_path_filter(source_path: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="source_path",
                match=MatchValue(value=source_path),
            )
        ]
    )


def get_existing_points_for_source_path(source_path: str) -> list:
    client = get_qdrant_client()

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=_source_path_filter(source_path),
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )
    return points


def get_existing_content_hash_for_source_path(source_path: str) -> str | None:
    points = get_existing_points_for_source_path(source_path)
    if not points:
        return None

    payload = points[0].payload or {}
    return payload.get("content_hash")


def delete_points_for_source_path(source_path: str) -> int:
    client = get_qdrant_client()
    points = get_existing_points_for_source_path(source_path)

    if not points:
        return 0

    point_ids = [point.id for point in points]

    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=point_ids,
    )

    return len(point_ids)
