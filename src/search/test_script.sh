uv run python - <<'PY'
from src.search import search_similar
from src.search.bundler import build_bundles

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

    results = search_similar(
        q,
        k=20,
        with_payload=True,
        with_vectors=True,
    )

    print("RAW POINTS:")
    for r in results:
        payload = r.payload or {}
        print(
            f"  {r.score:.4f}  {payload.get('file_name')}  "
            f"modality={payload.get('modality')}  "
            f"family={payload.get('embedding_family')}  "
            f"pipeline={payload.get('pipeline_name')}"
        )

    bundles = build_bundles(
        results,
        score_threshold=0.45,
        grouping_threshold=0.60,
        max_pool_size=20,
    )

    print("BUNDLES:")
    for b in bundles:
        print(f"  bundle={b.title} score={b.score:.4f} files={len(b.source_files)} views={len(b.views)}")
        for v in b.views[:5]:
            payload = v.payload or {}
            print(
                f"    - {v.score:.4f} {payload.get('file_name')} "
                f"modality={payload.get('modality')} "
                f"family={payload.get('embedding_family')}"
            )
PY