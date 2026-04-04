from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class EmbeddingRecord:
    id: str
    vector: list[float]

    source_file_id: str
    source_path: str
    file_name: str
    extension: str
    mime_type: str | None

    modality: str
    pipeline_name: str

    chunk_id: str
    chunk_index: int | None
    embedding_family: str | None

    extracted_text: str | None
    content_hash: str

    created_at: str
    updated_at: str

    source_type: str | None = None
    org_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def make_point_payload(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "source_file_id": record.source_file_id,
        "source_path": record.source_path,
        "file_name": record.file_name,
        "extension": record.extension,
        "mime_type": record.mime_type,
        "modality": record.modality,
        "pipeline_name": record.pipeline_name,
        "chunk_id": record.chunk_id,
        "chunk_index": record.chunk_index,
        "embedding_family": record.embedding_family,
        "extracted_text": record.extracted_text,
        "content_hash": record.content_hash,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source_type": record.source_type,
        "org_id": record.org_id,
        "metadata": record.metadata,
    }


def new_id() -> str:
    return str(uuid4())
