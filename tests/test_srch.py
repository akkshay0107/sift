import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer.qdrant_db import ensure_collection, upsert_records
from src.indexer.pipelines import get_qwen_embedder
from src.indexer.schemas import EmbeddingRecord, new_id
from src.search import search_similar_files


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_record(
    *,
    text: str,
    idx: int,
    run_id: str,
    file_stub: str,
    modality: str,
    embedding_family: str,
) -> EmbeddingRecord:
    embedder = get_qwen_embedder()
    vector = embedder.embed(text).squeeze(0).tolist()
    ts = now_iso()
    source_file_id = f"{file_stub}:{run_id}"
    source_path = f"/search-tests/{run_id}/{file_stub}.txt"

    return EmbeddingRecord(
        id=new_id(),
        vector=vector,
        source_file_id=source_file_id,
        source_path=source_path,
        file_name=f"{file_stub}.txt",
        extension=".txt",
        mime_type="text/plain",
        modality=modality,
        pipeline_name="test_pipeline",
        chunk_id=f"{source_file_id}:{embedding_family}:{idx}",
        chunk_index=None,
        embedding_family=embedding_family,
        extracted_text=text,
        content_hash=f"search-test-{run_id}-{idx}",
        created_at=ts,
        updated_at=ts,
        source_type="search_test",
        metadata={"test_run_id": run_id},
    )


def main() -> None:
    run_id = uuid4().hex[:8]
    print(f"Search test run id: {run_id}")

    records = [
        make_record(
            text="A golden retriever is running across the beach at sunset.",
            idx=0,
            run_id=run_id,
            file_stub="dog_file",
            modality="text",
            embedding_family="primary_text",
        ),
        make_record(
            text="OCR text: dog chasing a tennis ball near the ocean shoreline.",
            idx=1,
            run_id=run_id,
            file_stub="dog_file",
            modality="ocr_text",
            embedding_family="ocr",
        ),
        make_record(
            text="Metadata summary: beach photo with a dog at sunset.",
            idx=2,
            run_id=run_id,
            file_stub="dog_file",
            modality="text",
            embedding_family="metadata",
        ),
        make_record(
            text="Quarterly earnings increased after the semiconductor market rebounded.",
            idx=3,
            run_id=run_id,
            file_stub="finance_file",
            modality="text",
            embedding_family="primary_text",
        ),
    ]

    print("Ensuring collection exists...")
    ensure_collection()

    print("Embedding and inserting test records...")
    upsert_records(records)
    print(f"Inserted {len(records)} records into the live collection.")

    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.test_run_id",
                match=MatchValue(value=run_id),
            )
        ]
    )

    query = "dog playing on the beach"
    print(f"\nQuery: {query!r}")

    file_results = search_similar_files(
        query,
        k=3,
        query_filter=query_filter,
        score_threshold=0.2,
    )

    print("\nTop file results:")
    for idx, result in enumerate(file_results, start=1):
        print(f"{idx}. file={result.file_name} location={result.source_path}")

    if len(file_results) < 1:
        raise AssertionError("Expected at least one file result")

    best_file = file_results[0]
    if best_file.file_name != "dog_file.txt":
        raise AssertionError(f"Expected dog_file.txt as top result, got {best_file.file_name}")

    normalized_path = best_file.source_path.replace("\\", "/")
    if not normalized_path.endswith(f"/{run_id}/dog_file.txt"):
        raise AssertionError(f"Expected absolute path for dog_file.txt, got {best_file.source_path}")

    returned_names = [result.file_name for result in file_results]
    if "finance_file.txt" in returned_names:
        raise AssertionError("Expected score_threshold to filter out the finance negative example")

    print("\nChecks passed:")
    print("- multiple embeddings for the same file were aggregated into one file result")
    print("- absolute file locations were returned")
    print("- score_threshold filtered weak negatives")


if __name__ == "__main__":
    main()
