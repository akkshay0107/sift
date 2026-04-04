from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(slots=True)
class SearchResult:
    id: str | int
    score: float
    payload: dict[str, Any] | None
    vector: list[float] | dict[str, list[float]] | None = None


@dataclass
class SearchBundle:
    bundle_id: str
    title: str
    score: float
    views: list[SearchResult]
    source_files: list[str]
    explanation: str


def parse_iso_time(time_str: str) -> datetime:
    # Handle potentially missing 'Z' or offset
    if time_str.endswith("Z"):
        time_str = time_str[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        return datetime.now(timezone.utc)


def calculate_temporal_similarity(t1_str: str, t2_str: str) -> float:
    if not t1_str or not t2_str:
        return 0.0

    t1 = parse_iso_time(t1_str)
    t2 = parse_iso_time(t2_str)

    diff_seconds = abs((t1 - t2).total_seconds())
    # 1 day = 86400 seconds. Scale linearly from 1.0 at 0s diff to 0.0 at 1 day diff
    max_diff = 86400.0
    sim = max(0.0, 1.0 - (diff_seconds / max_diff))
    return sim


def calculate_name_similarity(name1: str, name2: str) -> float:
    if not name1 or not name2:
        return 0.0
    name1 = name1.lower()
    name2 = name2.lower()
    if name1 == name2:
        return 1.0

    # basic similarity using Jaccard on characters
    set1 = set(name1)
    set2 = set(name2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def calculate_embedding_similarity(
    v1: list[float] | dict[str, list[float]] | None,
    v2: list[float] | dict[str, list[float]] | None,
) -> float:
    if not v1 or not v2:
        return 0.0
    # Qdrant can return vectors as dict when using named vectors.
    if isinstance(v1, dict):
        vec1_list = list(v1.values())[0] if v1 else []
    else:
        vec1_list = v1

    if isinstance(v2, dict):
        vec2_list = list(v2.values())[0] if v2 else []
    else:
        vec2_list = v2

    if not vec1_list or not vec2_list:
        return 0.0

    vec1 = np.array(vec1_list)
    vec2 = np.array(vec2_list)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def item_belongs_in_bundle(
    item: SearchResult, bundle: SearchBundle, grouping_threshold: float = 0.6
) -> bool:
    top_view = bundle.views[0]

    item_payload = item.payload or {}
    bundle_payload = top_view.payload or {}

    # 1. Embedding Similarity
    embed_sim = calculate_embedding_similarity(item.vector, top_view.vector)

    # 2. Temporal Similarity
    item_time = item_payload.get("updated_at") or item_payload.get("created_at") or ""
    bundle_time = (
        bundle_payload.get("updated_at") or bundle_payload.get("created_at") or ""
    )
    time_sim = calculate_temporal_similarity(item_time, bundle_time)

    # 3. Name Similarity
    item_name = item_payload.get("file_name", "")
    bundle_name = bundle_payload.get("file_name", "")
    name_sim = calculate_name_similarity(item_name, bundle_name)

    # Sensible defaults for weights
    combined_score = (0.7 * embed_sim) + (0.2 * time_sim) + (0.1 * name_sim)

    return combined_score >= grouping_threshold


def build_bundles(
    results: list[SearchResult],
    score_threshold: float = 0.5,
    grouping_threshold: float = 0.6,
    max_pool_size: int = 50,
) -> list[SearchBundle]:
    # 1. Take the top results as our pool
    pool = results[:max_pool_size]

    # 2. Separate into seeds (above threshold) and others (below threshold)
    seeds = [r for r in pool if r.score >= score_threshold]
    others = [r for r in pool if r.score < score_threshold]

    bundles: list[SearchBundle] = []

    # 3. Process seeds: they can create new bundles or merge into existing ones
    for item in seeds:
        placed = False
        for bundle in bundles:
            if item_belongs_in_bundle(item, bundle, grouping_threshold):
                bundle.views.append(item)
                source_path = item.payload.get("source_path") if item.payload else None
                if source_path and source_path not in bundle.source_files:
                    bundle.source_files.append(source_path)
                placed = True
                break

        if not placed:
            source_path = item.payload.get("source_path") if item.payload else None
            title = item.payload.get("file_name") if item.payload else None
            if not title:
                title = (
                    source_path.split("/")[-1]
                    if source_path
                    else f"Bundle {len(bundles) + 1}"
                )

            bundles.append(
                SearchBundle(
                    bundle_id=str(item.id),
                    title=title,
                    score=item.score,
                    views=[item],
                    source_files=[source_path] if source_path else [],
                    explanation="Grouped by similarity of content and metadata",
                )
            )

    # 4. Process others: they can ONLY merge into existing bundles, never create new ones
    for item in others:
        best_bundle = None
        # We find the best matching bundle instead of just the first one for 'others'
        # to ensure better quality grouping for these lower-score items.
        # But for simplicity and consistency with seeds, we can just use the first match
        # as item_belongs_in_bundle already checks the threshold.
        for bundle in bundles:
            if item_belongs_in_bundle(item, bundle, grouping_threshold):
                bundle.views.append(item)
                source_path = item.payload.get("source_path") if item.payload else None
                if source_path and source_path not in bundle.source_files:
                    bundle.source_files.append(source_path)
                break

    # 5. Finalize bundles
    for bundle in bundles:
        bundle.views.sort(key=lambda v: v.score, reverse=True)
        # Average the scores of the top views to represent the bundle's relevance
        top_views = bundle.views[:10]
        bundle.score = sum(v.score for v in top_views) / len(top_views)

    return sorted(bundles, key=lambda b: b.score, reverse=True)
