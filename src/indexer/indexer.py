from pathlib import Path

from src.indexer.config import TRUSTED_DIR
from src.indexer.file_router import get_pipelines_for_file
from src.indexer.file_utils import compute_file_hash
from src.indexer.pipelines import (
    build_audio_record,
    build_image_record,
    build_ocr_text_record,
    build_text_record,
    build_transcript_text_record,
)
from src.indexer.qdrant_db import (
    delete_points_for_source_path,
    ensure_collection,
    get_existing_content_hash_for_source_path,
    upsert_records,
)


def index_file(path: Path) -> tuple[int, str]:
    current_content_hash = compute_file_hash(path)
    source_path = str(path)

    existing_content_hash = get_existing_content_hash_for_source_path(source_path)

    if existing_content_hash == current_content_hash:
        return 0, "skipped_unchanged"

    if existing_content_hash is not None and existing_content_hash != current_content_hash:
        deleted_count = delete_points_for_source_path(source_path)
        print(f"Reindexing {path}: deleted {deleted_count} old record(s)")

    pipeline_names = get_pipelines_for_file(path)
    all_records = []

    for pipeline_name in pipeline_names:
        if pipeline_name == "text":
            all_records.extend(build_text_record(path))
        elif pipeline_name == "image":
            all_records.extend(build_image_record(path))
        elif pipeline_name == "ocr_text":
            all_records.extend(build_ocr_text_record(path))
        elif pipeline_name == "audio":
            all_records.extend(build_audio_record(path))
        elif pipeline_name == "transcript_text":
            all_records.extend(build_transcript_text_record(path))

    upsert_records(all_records)
    return len(all_records), "indexed"


def index_trusted_directory() -> None:
    ensure_collection()

    if not TRUSTED_DIR.exists():
        print(f"Trusted directory does not exist: {TRUSTED_DIR}")
        return

    total_files = 0
    total_records = 0
    skipped_files = 0

    for path in TRUSTED_DIR.rglob("*"):
        if not path.is_file():
            continue

        total_files += 1

        try:
            inserted, status = index_file(path)

            if status == "skipped_unchanged":
                skipped_files += 1
                print(f"Skipped {path} -> unchanged")
            else:
                total_records += inserted
                print(f"Indexed {path} -> {inserted} record(s)")

        except Exception as e:
            print(f"Failed indexing {path}: {e}")

    print(
        f"Done. Files scanned: {total_files}, "
        f"records inserted: {total_records}, "
        f"files skipped: {skipped_files}"
    )

