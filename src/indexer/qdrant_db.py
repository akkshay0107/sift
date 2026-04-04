from qdrant_client import QdrantClient

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host="localhost", port=6333)