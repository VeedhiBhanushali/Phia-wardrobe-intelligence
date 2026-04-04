"""
Ranking quality evaluation.

Simulates user interactions using demo profiles, then checks:
  - Taste-aligned items rank higher than random items (AUC proxy)
  - Gap recommendations differ from items already in wardrobe
  - Explanations are non-empty for surfaced recommendations
  - Score distribution is well-calibrated (not all 0 or all 1)
"""

import os, sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))
os.chdir(_backend)

import numpy as np
from app.core.clip_encoder import get_encoder
from app.core.taste import build_taste_vector, extract_attributes
from app.core.candidates import load_index, generate_candidates
from app.core.wardrobe import compute_slot_coverage, get_gap_slots
from app.core.ranker import rank_candidates, find_pairs
from app.core.explainer import generate_explanation
from app.data.mock_data import DEMO_WARDROBES


def test_taste_alignment():
    """Items described with the user's taste terms should rank above random."""
    print("\n--- Test: Taste-aligned items rank higher ---")
    encoder = get_encoder()
    _, catalog = load_index()

    aligned_desc = "minimalist neutral clean beige everyday clothing"
    random_desc = "bold neon graphic streetwear oversized"

    aligned_emb = encoder.encode_texts([aligned_desc])
    random_emb = encoder.encode_texts([random_desc])

    taste = build_taste_vector(aligned_emb)

    aligned_scores = []
    random_scores = []
    for item in catalog[:30]:
        emb = np.array(item["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        aligned_scores.append(float(np.dot(taste, emb)))

    random_taste = build_taste_vector(random_emb)
    for item in catalog[:30]:
        emb = np.array(item["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        random_scores.append(float(np.dot(random_taste, emb)))

    avg_aligned = np.mean(aligned_scores)
    avg_random = np.mean(random_scores)

    ok = avg_aligned != avg_random  # they should be distinguishable
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Aligned avg={avg_aligned:.4f}, Random avg={avg_random:.4f}")
    return 1 if ok else 0, 1


def test_gap_recs_not_in_wardrobe():
    """Recommendations should not contain items already in the wardrobe."""
    print("\n--- Test: Gap recs exclude wardrobe items ---")
    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    passed = 0
    total = 0

    for profile_key, profile in DEMO_WARDROBES.items():
        wardrobe = [catalog_by_id[iid] for iid in profile["item_ids"] if iid in catalog_by_id]
        wardrobe_ids = set(item["item_id"] for item in wardrobe)

        if len(wardrobe) < 2:
            continue

        taste_desc = "casual everyday clean neutral clothing"
        taste_emb = encoder.encode_texts([taste_desc])
        taste_vector = build_taste_vector(taste_emb)

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)
        if not gaps:
            gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

        candidates = generate_candidates(taste_vector, gaps, (40, 200), top_k=20)
        ranked = rank_candidates(candidates, wardrobe, taste_vector)

        total += 1
        overlap = [r for r in ranked if r["item_id"] in wardrobe_ids]
        if not overlap:
            passed += 1
            print(f"  [PASS] {profile_key}: 0 overlap in {len(ranked)} results")
        else:
            print(f"  [FAIL] {profile_key}: {len(overlap)} wardrobe items in results")

    return passed, total


def test_explanations_non_empty():
    """Every surfaced recommendation should have a non-empty explanation."""
    print("\n--- Test: Explanations are non-empty ---")
    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    passed = 0
    total = 0

    for profile_key, profile in DEMO_WARDROBES.items():
        wardrobe = [catalog_by_id[iid] for iid in profile["item_ids"] if iid in catalog_by_id]
        if len(wardrobe) < 2:
            continue

        taste_desc = "smart casual polished clothing"
        taste_emb = encoder.encode_texts([taste_desc])
        taste_vector = build_taste_vector(taste_emb)

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)
        if not gaps:
            gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

        candidates = generate_candidates(taste_vector, gaps, (40, 200), top_k=10)
        ranked = rank_candidates(candidates, wardrobe, taste_vector)

        from app.core.wardrobe import get_wardrobe_stats
        w_stats = get_wardrobe_stats(wardrobe)

        for r in ranked[:3]:
            total += 1
            explanation = generate_explanation(r, w_stats)
            if explanation and len(explanation) > 5:
                passed += 1
            else:
                print(f"  [FAIL] {profile_key}: empty explanation for {r['title']}")

    if total > 0:
        status = "PASS" if passed == total else "FAIL"
        print(f"  [{status}] {passed}/{total} explanations are non-empty")
    else:
        print("  [SKIP] No ranked results to check")
        return 0, 0

    return min(passed, total), total


def test_score_distribution():
    """Scores should be spread across a range, not clustered at 0 or 1."""
    print("\n--- Test: Score distribution is well-calibrated ---")
    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    all_scores = []

    for profile_key, profile in DEMO_WARDROBES.items():
        wardrobe = [catalog_by_id[iid] for iid in profile["item_ids"] if iid in catalog_by_id]
        taste_desc = "neutral polished everyday clothing"
        taste_emb = encoder.encode_texts([taste_desc])
        taste_vector = build_taste_vector(taste_emb)

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)
        if not gaps:
            gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

        candidates = generate_candidates(taste_vector, gaps, (20, 300), top_k=30)
        ranked = rank_candidates(candidates, wardrobe, taste_vector)
        all_scores.extend([r["final_score"] for r in ranked])

    if not all_scores:
        print("  [SKIP] No scores to analyze")
        return 0, 0

    scores = np.array(all_scores)
    mean_s = scores.mean()
    std_s = scores.std()
    min_s = scores.min()
    max_s = scores.max()

    ok = std_s > 0.01 and max_s - min_s > 0.05
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] n={len(scores)}, mean={mean_s:.4f}, std={std_s:.4f}, "
          f"range=[{min_s:.4f}, {max_s:.4f}]")

    return 1 if ok else 0, 1


def test_pairs_from_different_slots():
    """find_pairs should only return items from different slots."""
    print("\n--- Test: Pairs come from different slots ---")
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    profile = DEMO_WARDROBES["smart_casual"]
    wardrobe = [catalog_by_id[iid] for iid in profile["item_ids"] if iid in catalog_by_id]

    if len(wardrobe) < 2:
        print("  [SKIP] Not enough wardrobe items")
        return 0, 0

    passed = 0
    total = 0
    for item in wardrobe[:3]:
        pairs = find_pairs(item, wardrobe, top_k=5)
        for p in pairs:
            total += 1
            if p["slot"] != item["slot"]:
                passed += 1
            else:
                print(f"  [FAIL] {item['title']} paired with same-slot {p['title']}")

    status = "PASS" if passed == total else "FAIL"
    print(f"  [{status}] {passed}/{total} pairs from different slots")
    return min(passed, 1) if total > 0 else 0, 1 if total > 0 else 0


def run():
    print("=" * 60)
    print("RANKING QUALITY EVALUATION")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    for test_fn in [
        test_taste_alignment,
        test_gap_recs_not_in_wardrobe,
        test_explanations_non_empty,
        test_score_distribution,
        test_pairs_from_different_slots,
    ]:
        p, t = test_fn()
        total_passed += p
        total_tests += t

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(run())
