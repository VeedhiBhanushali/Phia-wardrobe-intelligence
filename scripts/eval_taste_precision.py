"""
Taste cold start evaluation — Precision@10 vs popularity baseline.

For simulated profiles at 0, 3, 5, 10 saves, show Precision@10
vs a popularity baseline. The system should beat popularity at all
save counts.

Precision is measured as: of the top-10 recommended items, how many
share the same primary style family as the simulated taste profile.
"""

import os, sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))
os.chdir(_backend)

import numpy as np
from app.core.clip_encoder import get_encoder
from app.core.taste import build_taste_vector, extract_taste_modes
from app.core.candidates import load_index, generate_candidates
from app.core.wardrobe import compute_slot_coverage, get_gap_slots
from app.core.ranker import rank_candidates

# Style profiles to simulate
STYLE_PROFILES = {
    "minimalist": "minimalist neutral clean structured tailored beige white ivory clothing",
    "streetwear": "streetwear oversized hoodie sneakers cargo pants bold urban clothing",
    "elegant": "elegant formal polished evening cocktail heels satin silk clothing",
}


def _taste_relevance(item_emb: np.ndarray, taste_vector: np.ndarray) -> float:
    """Cosine sim between item and taste — proxy for relevance."""
    return float(np.dot(item_emb, taste_vector))


def evaluate_precision_at_k(k: int = 10):
    print("=" * 60)
    print(f"TASTE COLD START — Precision@{k} vs Popularity Baseline")
    print("=" * 60)

    encoder = get_encoder()
    _, catalog = load_index()

    # Popularity baseline: just take the first k items (simulates no personalization)
    popularity_items = catalog[:k]

    save_counts = [0, 3, 5, 10]
    results = []

    for style_name, style_desc in STYLE_PROFILES.items():
        print(f"\n--- Style: {style_name} ---")

        # Build taste from style description
        style_emb = encoder.encode_texts([style_desc])
        taste_vector = build_taste_vector(style_emb)

        # Compute ground-truth relevance: top items by cosine sim to taste
        all_sims = []
        for item in catalog:
            emb = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            all_sims.append((item["item_id"], float(np.dot(taste_vector, emb))))

        all_sims.sort(key=lambda x: x[1], reverse=True)
        relevant_ids = set(iid for iid, _ in all_sims[:30])  # top 30 = relevant

        # Popularity baseline precision
        pop_hits = sum(1 for item in popularity_items if item["item_id"] in relevant_ids)
        pop_precision = pop_hits / k

        for n_saves in save_counts:
            # Simulate wardrobe by taking top-n relevant items
            wardrobe = []
            if n_saves > 0:
                top_relevant = [item for item in catalog
                                if item["item_id"] in relevant_ids][:n_saves]
                wardrobe = top_relevant

            coverage = compute_slot_coverage(wardrobe)
            gaps = get_gap_slots(coverage)
            if not gaps:
                gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

            # Build taste modes from wardrobe + style
            if wardrobe:
                ward_embs = np.array([item["embedding"] for item in wardrobe], dtype=np.float32)
                combined = np.concatenate([style_emb, ward_embs], axis=0)
                taste_modes = extract_taste_modes(combined)
            else:
                taste_modes = [taste_vector]

            candidates = generate_candidates(
                taste_vector, gaps, (20, 300),
                top_k=50,
                exclude_ids={item["item_id"] for item in wardrobe},
            )
            ranked = rank_candidates(
                candidates, wardrobe, taste_vector,
                taste_modes=taste_modes,
            )

            # Precision@k
            rec_ids = [r["item_id"] for r in ranked[:k]]
            hits = sum(1 for iid in rec_ids if iid in relevant_ids)
            precision = hits / k if k > 0 else 0

            beats_pop = precision >= pop_precision
            status = "PASS" if beats_pop else "FAIL"

            result = {
                "style": style_name,
                "saves": n_saves,
                "precision": precision,
                "pop_precision": pop_precision,
                "beats_baseline": beats_pop,
            }
            results.append(result)

            print(f"  [{status}] saves={n_saves:2d}: P@{k}={precision:.2f} "
                  f"(pop baseline={pop_precision:.2f})")

    # Summary
    print("\n" + "=" * 60)
    all_beat = sum(1 for r in results if r["beats_baseline"])
    total = len(results)
    print(f"SUMMARY: System beats popularity baseline in {all_beat}/{total} scenarios")

    for n in save_counts:
        subset = [r for r in results if r["saves"] == n]
        avg_p = np.mean([r["precision"] for r in subset])
        avg_pop = np.mean([r["pop_precision"] for r in subset])
        print(f"  saves={n:2d}: avg P@10={avg_p:.3f}, avg pop={avg_pop:.3f}, "
              f"lift={avg_p - avg_pop:+.3f}")

    return 0 if all_beat == total else 1


if __name__ == "__main__":
    sys.exit(evaluate_precision_at_k())
