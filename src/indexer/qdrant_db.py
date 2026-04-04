from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.indexer.config import (
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_DIM,
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
