"""
Cold-start evaluation.

Tests whether the taste pipeline produces sensible outputs from scratch
and whether the recommendation engine can surface useful items with zero
or very few wardrobe saves.
"""

import os, sys, json
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))
os.chdir(_backend)

import numpy as np
from app.core.clip_encoder import get_encoder
from app.core.taste import build_taste_vector, extract_attributes, infer_price_tier
from app.core.candidates import load_index, generate_candidates
from app.core.wardrobe import compute_slot_coverage, get_gap_slots
from app.core.ranker import rank_candidates
from app.data.mock_data import DEMO_WARDROBES

STYLE_QUERIES = [
    "minimalist neutral clean everyday clothing",
    "streetwear oversized bold graphic clothing",
    "elegant formal polished classic clothing",
]


def run():
    print("=" * 60)
    print("COLD-START EVALUATION")
    print("=" * 60)

    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    results = []

    for style_desc in STYLE_QUERIES:
        print(f"\n--- Style: {style_desc[:40]}... ---")

        text_emb = encoder.encode_texts([style_desc])
        taste_vector = build_taste_vector(text_emb)
        attributes = extract_attributes(taste_vector)
        price_tier = infer_price_tier(text_emb)

        print(f"  Attributes:")
        for attr, val in attributes.items():
            print(f"    {attr}: {val['label']} ({val['confidence']:.3f})")
        print(f"  Price tier: ${price_tier[0]:.0f} – ${price_tier[1]:.0f}")

        for profile_key, profile in DEMO_WARDROBES.items():
            wardrobe = []
            for iid in profile["item_ids"]:
                if iid in catalog_by_id:
                    wardrobe.append(catalog_by_id[iid])

            coverage = compute_slot_coverage(wardrobe)
            gaps = get_gap_slots(coverage)

            if not gaps:
                gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

            candidates = generate_candidates(taste_vector, gaps, price_tier, top_k=20)
            ranked = rank_candidates(candidates, wardrobe, taste_vector)

            n_ranked = len(ranked)
            top_score = ranked[0]["final_score"] if ranked else 0.0
            has_results = n_ranked > 0
            slot_diversity = len(set(r["slot"] for r in ranked)) if ranked else 0

            result = {
                "style": style_desc[:40],
                "profile": profile_key,
                "wardrobe_size": len(wardrobe),
                "gap_slots": gaps,
                "n_candidates": len(candidates),
                "n_ranked": n_ranked,
                "top_score": top_score,
                "slot_diversity": slot_diversity,
                "pass": has_results,
            }
            results.append(result)

            status = "PASS" if has_results else "FAIL"
            print(f"  [{status}] Profile '{profile_key}' (wardrobe={len(wardrobe)}): "
                  f"{n_ranked} results, top_score={top_score:.3f}, "
                  f"slots={slot_diversity}")

            if ranked:
                top = ranked[0]
                print(f"         Top rec: {top['title']} ({top['slot']}) "
                      f"taste={top['taste_score']:.3f} unlock={top['unlock_count']}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"SUMMARY: {passed}/{total} scenarios produced results")

    cold_results = [r for r in results if r["wardrobe_size"] == 0]
    cold_pass = sum(1 for r in cold_results if r["pass"])
    print(f"  Cold start (empty wardrobe): {cold_pass}/{len(cold_results)} passed")

    few_results = [r for r in results if 0 < r["wardrobe_size"] <= 5]
    few_pass = sum(1 for r in few_results if r["pass"])
    print(f"  Few items (1-5): {few_pass}/{len(few_results)} passed")

    warm_results = [r for r in results if r["wardrobe_size"] > 5]
    warm_pass = sum(1 for r in warm_results if r["pass"])
    print(f"  Warm (>5 items): {warm_pass}/{len(warm_results)} passed")

    avg_diversity = np.mean([r["slot_diversity"] for r in results if r["pass"]]) if passed else 0
    print(f"  Avg slot diversity (passing): {avg_diversity:.1f}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
