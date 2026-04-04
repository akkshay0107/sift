from pathlib import Path

from src.indexer.config import TRUSTED_DIR
from src.indexer.file_router import get_pipelines_for_file
from src.indexer.pipelines import (
    build_text_record,
    build_image_record,
    build_ocr_text_record,
)
from src.indexer.qdrant_db import ensure_collection, upsert_records


def index_file(path: Path) -> int:
    pipeline_names = get_pipelines_for_file(path)
    all_records = []

    for pipeline_name in pipeline_names:
        if pipeline_name == "text":
            all_records.extend(build_text_record(path))
        elif pipeline_name == "image":
            all_records.extend(build_image_record(path))
        elif pipeline_name == "ocr_text":
            all_records.extend(build_ocr_text_record(path))

    upsert_records(all_records)
    return len(all_records)


def index_trusted_directory() -> None:
    ensure_collection()

    if not TRUSTED_DIR.exists():
        print(f"Trusted directory does not exist: {TRUSTED_DIR}")
        return

    total_files = 0
    total_records = 0

    for path in TRUSTED_DIR.rglob("*"):
        if not path.is_file():
            continue

        total_files += 1
        try:
            inserted = index_file(path)
            total_records += inserted
            print(f"Indexed {path} -> {inserted} record(s)")
        except Exception as e:
            print(f"Failed indexing {path}: {e}")

    print(f"Done. Files scanned: {total_files}, records inserted: {total_records}")
