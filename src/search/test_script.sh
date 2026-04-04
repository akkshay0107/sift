uv run python - <<'PY'
from src.search import search_similar, search_similar_files

queries = [
    "hi hello",
    "loser shut up",
    "happy quote",
    "superman",
    "4 dice",
    "dancing masquerade",
    "princess",
    "jester",
]

for q in queries:
    print(f"\nQUERY: {q}")
    print("FILES:")
    for r in search_similar_files(q, k=5):
        print(f"  {r.score:.4f}  {r.file_name}")

    print("RAW POINTS:")
    for r in search_similar(q, k=8):
        payload = r.payload or {}
        print(
            f"  {r.score:.4f}  {payload.get('file_name')}  "
            f"modality={payload.get('modality')}  "
            f"family={payload.get('embedding_family')}  "
            f"pipeline={payload.get('pipeline_name')}"
        )
PY