import random

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = QdrantClient(host="localhost", port=6333)

COLLECTION = "catapult_index"

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )

client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(
            id=1,
            vector=[random.random() for _ in range(4)],
            payload={
                "source_file_id": "file_001",
                "source_path": "/trusted/demo.txt",
                "modality": "text",
                "chunk_id": "file_001_chunk_0",
                "chunk_index": 0,
                "extracted_text": "hello world",
            },
        )
    ],
)

print("Inserted test point.")
