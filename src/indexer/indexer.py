import logging
import os
from pathlib import Path

from src.indexer.config import MONITORED_DIRECTORIES
from src.indexer.file_router import get_pipelines_for_file
from src.indexer.file_utils import compute_file_hash
from src.indexer.pipelines import (
    build_audio_record,
    build_image_record,
    build_metadata_record,
    build_ocr_text_record,
    build_text_record,
    build_transcript_text_record,
    build_video_record,
)
from src.indexer.qdrant_db import (
    delete_points_for_source_path,
    ensure_collection,
    get_existing_content_hash_for_source_path,
    upsert_records,
)

logger = logging.getLogger(__name__)


def index_file(path: Path) -> tuple[int, str]:
    current_content_hash = compute_file_hash(path)
    source_path = str(path)

    existing_content_hash = get_existing_content_hash_for_source_path(source_path)

    if existing_content_hash == current_content_hash:
        return 0, "skipped_unchanged"

    if (
        existing_content_hash is not None
        and existing_content_hash != current_content_hash
    ):
        deleted_count = delete_points_for_source_path(source_path)
        logger.info("Reindexing %s: deleted %d old record(s)", path, deleted_count)

    pipeline_names = get_pipelines_for_file(path)
    all_records = []
    metadata_modality: str | None = None

    for pipeline_name in pipeline_names:
        if pipeline_name == "text":
            all_records.extend(build_text_record(path))
            metadata_modality = "text"
        elif pipeline_name == "image":
            all_records.extend(build_image_record(path))
            if metadata_modality is None:
                metadata_modality = "image"
        elif pipeline_name == "ocr_text":
            ocr_records = build_ocr_text_record(path)
            all_records.extend(ocr_records)
            if ocr_records:
                metadata_modality = "ocr_text"
        elif pipeline_name == "audio":
            all_records.extend(build_audio_record(path))
            if metadata_modality is None:
                metadata_modality = "audio"
        elif pipeline_name == "transcript_text":
            transcript_records = build_transcript_text_record(path)
            all_records.extend(transcript_records)
            if transcript_records:
                metadata_modality = "transcript_text"
        elif pipeline_name == "video":
            all_records.extend(build_video_record(path))
            if metadata_modality is None:
                metadata_modality = "video"

    if pipeline_names and metadata_modality is not None:
        all_records.extend(build_metadata_record(path, modality=metadata_modality))

    upsert_records(all_records)
    return len(all_records), "indexed"


def index_monitored_directories() -> None:
    ensure_collection()

    total_files = 0
    total_records = 0
    skipped_files = 0

    for directory in MONITORED_DIRECTORIES:
        if not directory.exists():
            logger.warning("Directory does not exist: %s", directory)
            continue

        logger.info("Indexing directory: %s", directory)

        for root, dirs, files in os.walk(directory):
            # Prune hidden directories in-place to avoid recursion.
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.startswith("."):
                    continue

                path = Path(root) / file
                total_files += 1

                try:
                    inserted, status = index_file(path)

                    if status == "skipped_unchanged":
                        skipped_files += 1
                        logger.debug("Skipped %s -> unchanged", path)
                    else:
                        total_records += inserted
                        logger.info("Indexed %s -> %d record(s)", path, inserted)

                except Exception as e:
                    logger.error("Failed indexing %s: %s", path, e)

    logger.info(
        "Done. Total files scanned: %d, records inserted: %d, files skipped: %d",
        total_files,
        total_records,
        skipped_files,
    )
