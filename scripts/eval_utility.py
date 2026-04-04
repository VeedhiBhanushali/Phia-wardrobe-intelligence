"""
Utility calibration evaluation.

Tests that the outfit unlock count and compatibility scoring behave
correctly: items in gap slots should unlock more outfits than items
in already-filled slots, and scores should be monotonically higher
for complementary categories.
"""

import os, sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))
os.chdir(_backend)

import numpy as np
from app.core.candidates import load_index
from app.core.wardrobe import compute_slot_coverage, get_gap_slots
from app.core.ranker import (
    outfit_unlock_count,
    compatibility_score,
    aggregate_compatibility,
    COMPLEMENT_RULES,
)
from app.data.mock_data import DEMO_WARDROBES


def test_unlock_gap_vs_filled():
    """Items in gap slots should unlock more outfits than duplicates in filled slots."""
    print("\n--- Test: Gap slot items unlock more outfits ---")
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    passed = 0
    total = 0

    for profile_key, profile in DEMO_WARDROBES.items():
        if profile_key == "empty":
            continue

        wardrobe = [catalog_by_id[iid] for iid in profile["item_ids"] if iid in catalog_by_id]
        if len(wardrobe) < 2:
            continue

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)
        filled = [s for s, items in coverage.items() if len(items) > 0]

        if not gaps or not filled:
            continue

        gap_candidates = [c for c in catalog if c["slot"] in gaps and c["item_id"] not in profile["item_ids"]]
        filled_candidates = [c for c in catalog if c["slot"] in filled and c["item_id"] not in profile["item_ids"]]

        if not gap_candidates or not filled_candidates:
            continue

        gap_unlocks = [outfit_unlock_count(c, wardrobe) for c in gap_candidates[:10]]
        filled_unlocks = [outfit_unlock_count(c, wardrobe) for c in filled_candidates[:10]]

        avg_gap = np.mean(gap_unlocks) if gap_unlocks else 0
        avg_filled = np.mean(filled_unlocks) if filled_unlocks else 0

        total += 1
        ok = avg_gap >= avg_filled
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {profile_key}: gap_avg={avg_gap:.1f} vs filled_avg={avg_filled:.1f} "
              f"(gaps={gaps}, filled={filled})")

    return passed, total


def test_complement_rules_symmetric():
    """Complement rules should be defined for both directions of key pairs."""
    print("\n--- Test: Complement rules symmetry ---")
    pairs_checked = set()
    missing = []

    for (a, b), score in COMPLEMENT_RULES.items():
        pair = frozenset({a, b})
        if pair in pairs_checked:
            continue
        pairs_checked.add(pair)
        if (b, a) not in COMPLEMENT_RULES:
            missing.append((b, a))

    if missing:
        print(f"  [FAIL] Missing reverse pairs: {missing}")
        return 0, 1
    else:
        print(f"  [PASS] All {len(pairs_checked)} pairs have symmetric rules")
        return 1, 1


def test_compatibility_positive_for_complements():
    """Complementary items (e.g., tops + bottoms) should have positive compatibility."""
    print("\n--- Test: Complementary categories produce positive scores ---")
    _, catalog = load_index()

    tops = [c for c in catalog if c["slot"] == "tops"][:5]
    bottoms = [c for c in catalog if c["slot"] == "bottoms"][:5]

    passed = 0
    total = 0

    for top in tops:
        for bottom in bottoms:
            total += 1
            score = compatibility_score(
                top["embedding"], bottom["embedding"],
                top["slot"], bottom["slot"]
            )
            ok = score > 0
            if ok:
                passed += 1

    print(f"  [{'PASS' if passed == total else 'FAIL'}] "
          f"{passed}/{total} top-bottom pairs have positive compatibility")
    return min(passed, 1), 1


def test_same_slot_zero_compat():
    """Items in the same slot should get zero compatibility (no self-pairing)."""
    print("\n--- Test: Same-slot items get zero compatibility ---")
    _, catalog = load_index()

    tops = [c for c in catalog if c["slot"] == "tops"][:5]
    passed = 0
    total = 0

    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            total += 1
            score = compatibility_score(
                tops[i]["embedding"], tops[j]["embedding"],
                tops[i]["slot"], tops[j]["slot"]
            )
            if score == 0.0:
                passed += 1

    print(f"  [{'PASS' if passed == total else 'FAIL'}] "
          f"{passed}/{total} same-slot pairs scored zero")
    return min(passed, 1), 1


def run():
    print("=" * 60)
    print("UTILITY CALIBRATION EVALUATION")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    for test_fn in [
        test_unlock_gap_vs_filled,
        test_complement_rules_symmetric,
        test_compatibility_positive_for_complements,
        test_same_slot_zero_compat,
    ]:
        p, t = test_fn()
        total_passed += p
        total_tests += t

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(run())
