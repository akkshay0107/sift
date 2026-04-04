from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_DIR = PROJECT_ROOT / "trusted"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "catapult_index"

EMBEDDING_DIM = 2048
DISTANCE = "cosine"
