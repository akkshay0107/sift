from datetime import datetime, timezone
from pathlib import Path

from src.embed.qwen import QwenEmbedder
from src.embed.ocr_chain import OCREmbeddingPipeline
from src.indexer.file_utils import compute_file_hash, guess_mime_type, file_extension
from src.indexer.schemas import EmbeddingRecord, new_id


_qwen_embedder: QwenEmbedder | None = None
_ocr_pipeline: OCREmbeddingPipeline | None = None


def get_qwen_embedder() -> QwenEmbedder:
    global _qwen_embedder
    if _qwen_embedder is None:
        _qwen_embedder = QwenEmbedder()
    return _qwen_embedder


def get_ocr_pipeline() -> OCREmbeddingPipeline:
    global _ocr_pipeline
    if _ocr_pipeline is None:
        _ocr_pipeline = OCREmbeddingPipeline()
    return _ocr_pipeline


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_file_id_for(path: Path, content_hash: str) -> str:
    return f"{path.name}:{content_hash[:16]}"


def build_text_record(path: Path) -> list[EmbeddingRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    embedder = get_qwen_embedder()
    tensor = embedder.embed(text)
    vector = tensor.squeeze(0).tolist()

    content_hash = compute_file_hash(path)
    ts = now_iso()
    source_file_id = source_file_id_for(path, content_hash)

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            source_file_id=source_file_id,
            source_path=str(path),
            file_name=path.name,
            extension=file_extension(path),
            mime_type=guess_mime_type(path),
            modality="text",
            pipeline_name="qwen_text",
            chunk_id=f"{source_file_id}:text:0",
            chunk_index=0,
            embedding_family="primary_text",
            extracted_text=text,
            content_hash=content_hash,
            created_at=ts,
            updated_at=ts,
            metadata={},
        )
    ]


def build_image_record(path: Path) -> list[EmbeddingRecord]:
    embedder = get_qwen_embedder()
    tensor = embedder.embed(str(path))
    vector = tensor.squeeze(0).tolist()

    content_hash = compute_file_hash(path)
    ts = now_iso()
    source_file_id = source_file_id_for(path, content_hash)

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            source_file_id=source_file_id,
            source_path=str(path),
            file_name=path.name,
            extension=file_extension(path),
            mime_type=guess_mime_type(path),
            modality="image",
            pipeline_name="qwen_image",
            chunk_id=f"{source_file_id}:image:0",
            chunk_index=0,
            embedding_family="primary_image",
            extracted_text=None,
            content_hash=content_hash,
            created_at=ts,
            updated_at=ts,
            metadata={},
        )
    ]


def build_ocr_text_record(path: Path) -> list[EmbeddingRecord]:
    ocr = get_ocr_pipeline()
    text, tensor = ocr.process(str(path), return_embedding=True)

    text = text.strip()
    if not text:
        return []

    vector = tensor.squeeze(0).tolist()

    content_hash = compute_file_hash(path)
    ts = now_iso()
    source_file_id = source_file_id_for(path, content_hash)

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            source_file_id=source_file_id,
            source_path=str(path),
            file_name=path.name,
            extension=file_extension(path),
            mime_type=guess_mime_type(path),
            modality="ocr_text",
            pipeline_name="easyocr_qwen_text",
            chunk_id=f"{source_file_id}:ocr_text:0",
            chunk_index=0,
            embedding_family="ocr",
            extracted_text=text,
            content_hash=content_hash,
            created_at=ts,
            updated_at=ts,
            metadata={},
        )
    ]
