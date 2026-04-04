from src.search.bundler import SearchResult, build_bundles


def mock_result(id_val, score, vector=None, payload=None):
    return SearchResult(
        id=id_val,
        score=score,
        vector=vector or [1.0, 0.0],
        payload=payload
        or {"file_name": f"File {id_val}", "source_path": f"/path/{id_val}"},
    )


def test_build_bundles_seed_and_others():
    # Setup results:
    # 2 high score seeds (0.9, 0.85) - different content
    # 1 seed (0.8) - similar to seed 1
    # 2 others (0.4, 0.3) - one similar to seed 2, one similar to nothing

    # Vector [1.0, 0.0] for Seed 1
    # Vector [0.0, 1.0] for Seed 2
    # Vector [1.0, 0.1] for Seed 3 (similar to Seed 1)
    # Vector [0.1, 1.0] for Other 1 (similar to Seed 2)
    # Vector [-1.0, 0.0] for Other 2 (similar to nothing)

    results = [
        mock_result(1, 0.9, vector=[1.0, 0.0]),
        mock_result(2, 0.85, vector=[0.0, 1.0]),
        mock_result(3, 0.81, vector=[1.0, 0.1]),
        mock_result(4, 0.45, vector=[0.1, 1.0]),
        mock_result(5, 0.40, vector=[-1.0, 0.0]),
    ]

    # Thresholds: seed >= 0.8, grouping >= 0.6
    # 1. Seed 1 (0.9) starts Bundle A.
    # 2. Seed 2 (0.85) starts Bundle B (dissimilar to A).
    # 3. Seed 3 (0.81) merges into Bundle A (similar to Seed 1).
    # 4. Other 4 (0.45) merges into Bundle B (similar to Seed 2).
    # 5. Other 5 (0.40) is ignored (no match, cannot start new bundle).

    bundles = build_bundles(results, score_threshold=0.8, grouping_threshold=0.6)

    assert len(bundles) == 2

    # Bundle A (started by 1, joined by 3)
    bundle_a = next(b for b in bundles if b.bundle_id == "1")
    assert len(bundle_a.views) == 2
    assert {v.id for v in bundle_a.views} == {1, 3}

    # Bundle B (started by 2, joined by 4)
    bundle_b = next(b for b in bundles if b.bundle_id == "2")
    assert len(bundle_b.views) == 2
    assert {v.id for v in bundle_b.views} == {2, 4}

    # Verify Other 5 is nowhere
    all_view_ids = {v.id for b in bundles for v in b.views}
    assert 5 not in all_view_ids


def test_build_bundles_no_seeds():
    results = [
        mock_result(1, 0.4, vector=[1.0, 0.0]),
        mock_result(2, 0.3, vector=[1.0, 0.0]),
    ]
    # No items >= 0.5
    bundles = build_bundles(results, score_threshold=0.5)
    assert len(bundles) == 0


def test_build_bundles_max_pool_size():
    results = [mock_result(i, 0.9 - i * 0.01) for i in range(100)]
    # All are above threshold 0.5
    # If max_pool_size = 10, only first 10 should be processed
    bundles = build_bundles(results, score_threshold=0.5, max_pool_size=10)

    all_view_ids = {v.id for b in bundles for v in b.views}
    assert len(all_view_ids) <= 10  # Some might merge
    assert max(all_view_ids) < 10
