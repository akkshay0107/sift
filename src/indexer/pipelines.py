from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from src.embed.audio import AudioEmbedder, AudioSegment
from src.embed.ocr_chain import OCREmbeddingPipeline
from src.embed.qwen import QwenEmbedder
from src.embed.whisper_chain import WhisperChain, WhisperTranscriber
from src.indexer.file_utils import compute_file_hash, guess_mime_type, file_extension
from src.indexer.schemas import EmbeddingRecord, new_id


_qwen_embedder: QwenEmbedder | None = None
_ocr_pipeline: OCREmbeddingPipeline | None = None
_audio_embedder: AudioEmbedder | None = None
_whisper_transcriber: WhisperTranscriber | None = None
_whisper_chain: WhisperChain | None = None


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


def get_audio_embedder() -> AudioEmbedder:
    global _audio_embedder
    if _audio_embedder is None:
        _audio_embedder = AudioEmbedder()
    return _audio_embedder


def get_whisper_transcriber() -> WhisperTranscriber:
    global _whisper_transcriber
    if _whisper_transcriber is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _whisper_transcriber = WhisperTranscriber(device=device)
    return _whisper_transcriber


def get_whisper_chain() -> WhisperChain:
    global _whisper_chain
    if _whisper_chain is None:
        _whisper_chain = WhisperChain(
            transcriber=get_whisper_transcriber(),
            qwen_embedder=get_qwen_embedder(),
        )
    return _whisper_chain


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_file_id_for(path: Path, content_hash: str) -> str:
    return f"{path.name}:{content_hash[:16]}"


def load_audio_file(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    return audio, sample_rate


def make_base_kwargs(path: Path) -> dict:
    content_hash = compute_file_hash(path)
    stat = path.stat()
    source_file_id = source_file_id_for(path, content_hash)

    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
    updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    return {
        "source_file_id": source_file_id,
        "source_path": str(path),
        "file_name": path.name,
        "extension": file_extension(path),
        "mime_type": guess_mime_type(path),
        "content_hash": content_hash,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def build_metadata_text(path: Path, base: dict, modality: str) -> str:
    return "\n".join(
        [
            f"filename: {path.stem}",
            f"extension: {base['extension']}",
            f"modality: {modality}",
            f"embedding family: metadata",
            f"created at: {base['created_at']}",
            f"updated at: {base['updated_at']}",
        ]
    )


def build_metadata_record(path: Path, modality: str) -> list[EmbeddingRecord]:
    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    metadata_text = build_metadata_text(path, base, modality)
    embedder = get_qwen_embedder()
    tensor = embedder.embed(metadata_text)
    vector = tensor.squeeze(0).tolist()

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality=modality,
            pipeline_name="qwen_metadata_text",
            chunk_id=f"{source_file_id}:metadata:0",
            chunk_index=0,
            embedding_family="metadata",
            extracted_text=metadata_text,
            metadata={"is_metadata_embedding": True},
            **base,
        )
    ]


def build_text_record(path: Path) -> list[EmbeddingRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    embedder = get_qwen_embedder()
    tensor = embedder.embed(text)
    vector = tensor.squeeze(0).tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="text",
            pipeline_name="qwen_text",
            chunk_id=f"{source_file_id}:text:0",
            chunk_index=0,
            embedding_family="primary_text",
            extracted_text=text,
            metadata={},
            **base,
        )
    ]


def build_image_record(path: Path) -> list[EmbeddingRecord]:
    embedder = get_qwen_embedder()
    tensor = embedder.embed(str(path))
    vector = tensor.squeeze(0).tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="image",
            pipeline_name="qwen_image",
            chunk_id=f"{source_file_id}:image:0",
            chunk_index=0,
            embedding_family="primary_image",
            extracted_text=None,
            metadata={},
            **base,
        )
    ]


def build_ocr_text_record(path: Path) -> list[EmbeddingRecord]:
    ocr = get_ocr_pipeline()
    text, tensor = ocr.process(str(path), return_embedding=True)

    text = text.strip()
    if not text:
        return []

    vector = tensor.squeeze(0).tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="ocr_text",
            pipeline_name="easyocr_qwen_text",
            chunk_id=f"{source_file_id}:ocr_text:0",
            chunk_index=0,
            embedding_family="ocr",
            extracted_text=text,
            metadata={},
            **base,
        )
    ]


def build_audio_record(path: Path) -> list[EmbeddingRecord]:
    audio, sample_rate = load_audio_file(path)

    embedder = get_audio_embedder()
    tensor = embedder.embed(audio, sample_rate)
    vector = tensor.squeeze(0).tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="audio",
            pipeline_name="clap_projected_audio",
            chunk_id=f"{source_file_id}:audio:0",
            chunk_index=0,
            embedding_family="primary_audio",
            extracted_text=None,
            metadata={
                "sample_rate": sample_rate,
                "duration_seconds": float(len(audio) / sample_rate) if sample_rate else None,
            },
            **base,
        )
    ]


def build_transcript_text_record(path: Path) -> list[EmbeddingRecord]:
    audio, sample_rate = load_audio_file(path)

    segment = AudioSegment(
        data=audio,
        sample_rate=sample_rate,
        t_start=0.0,
        t_end=float(len(audio) / sample_rate) if sample_rate else 0.0,
        source_id=str(path),
    )

    whisper_chain = get_whisper_chain()
    tensor = whisper_chain.embed(segment)
    transcript = (segment.transcript or "").strip()

    if not transcript:
        return []

    vector = tensor.squeeze(0).tolist() if tensor.ndim > 1 else tensor.tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="transcript_text",
            pipeline_name="whisper_qwen_text",
            chunk_id=f"{source_file_id}:transcript_text:0",
            chunk_index=0,
            embedding_family="transcript",
            extracted_text=transcript,
            metadata={
                "sample_rate": sample_rate,
                "duration_seconds": segment.t_end,
            },
            **base,
        )
    ]

def build_video_record(path: Path) -> list[EmbeddingRecord]:
    embedder = get_qwen_embedder()
    tensor = embedder.embed(str(path))
    vector = tensor.squeeze(0).tolist()

    base = make_base_kwargs(path)
    source_file_id = base["source_file_id"]

    return [
        EmbeddingRecord(
            id=new_id(),
            vector=vector,
            modality="video",
            pipeline_name="qwen_video",
            chunk_id=f"{source_file_id}:video:0",
            chunk_index=0,
            embedding_family="primary_video",
            extracted_text=None,
            metadata={},
            **base,
        )
    ]
