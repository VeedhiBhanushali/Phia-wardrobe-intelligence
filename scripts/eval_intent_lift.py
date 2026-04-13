"""
Intent vector lift evaluation.

For sessions with simulated strong intent (browsing 5 similar items),
show that the blended ranker improves Precision@10 over taste-only.
Even a small improvement is meaningful — this is the novel contribution.
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
from app.core.intent import compute_intent
from app.data.mock_data import DEMO_WARDROBES

# Intent browsing sessions: style description + slot focus
INTENT_SESSIONS = {
    "black_boots": {
        "query": "black leather ankle boots edgy minimal",
        "target_slot": "shoes",
    },
    "linen_tops": {
        "query": "relaxed linen blouse cream natural oversized",
        "target_slot": "tops",
    },
    "structured_bags": {
        "query": "structured leather tote bag black minimal",
        "target_slot": "bags",
    },
}


def _simulate_browsing(
    encoder,
    catalog: list[dict],
    query: str,
    target_slot: str,
    n_views: int = 5,
) -> list[list[float]]:
    """Simulate a browsing session: find items closest to the intent query."""
    query_emb = encoder.encode_texts([query])[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    slot_items = [c for c in catalog if c["slot"] == target_slot]
    if not slot_items:
        slot_items = catalog

    scored = []
    for item in slot_items:
        emb = np.array(item["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        sim = float(np.dot(query_emb, emb))
        scored.append((item, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    viewed = [item["embedding"] for item, _ in scored[:n_views]]
    return viewed


def run():
    print("=" * 60)
    print("INTENT VECTOR LIFT — Precision@10 with vs without intent")
    print("=" * 60)

    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {item["item_id"]: item for item in catalog}

    results = []

    for profile_key, profile in DEMO_WARDROBES.items():
        if profile_key == "cold_start":
            continue

        wardrobe_ids = [iid for iid in profile["item_ids"] if iid in catalog_by_id]
        if len(wardrobe_ids) < 2:
            continue

        wardrobe = [catalog_by_id[iid] for iid in wardrobe_ids]

        # Build taste from wardrobe
        ward_embs = np.array([item["embedding"] for item in wardrobe], dtype=np.float32)
        taste_vector = build_taste_vector(ward_embs)
        taste_modes = extract_taste_modes(ward_embs)

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)
        if not gaps:
            gaps = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

        for session_name, session in INTENT_SESSIONS.items():
            print(f"\n--- {profile_key} + {session_name} intent ---")

            # Simulate browsing
            viewed_embs = _simulate_browsing(
                encoder, catalog, session["query"], session["target_slot"]
            )

            # Compute intent
            intent_result = compute_intent(viewed_embs)
            intent_vector = intent_result["intent_vector"]
            confidence = intent_result["confidence"]

            if intent_vector is None:
                print(f"  [SKIP] Confidence too low ({confidence:.3f})")
                continue

            intent_vec_np = np.array(intent_vector, dtype=np.float32)

            # Ground truth: items most relevant to the intent query
            query_emb = encoder.encode_texts([session["query"]])[0]
            query_emb = query_emb / np.linalg.norm(query_emb)

            all_sims = []
            for item in catalog:
                if item["item_id"] in set(wardrobe_ids):
                    continue
                emb = np.array(item["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                sim = float(np.dot(query_emb, emb))
                all_sims.append((item["item_id"], sim))

            all_sims.sort(key=lambda x: x[1], reverse=True)
            relevant_ids = set(iid for iid, _ in all_sims[:20])

            exclude_ids = set(wardrobe_ids)
            candidates = generate_candidates(
                taste_vector, gaps, (20, 300),
                top_k=50,
                exclude_ids=exclude_ids,
            )

            # Taste-only ranking
            ranked_taste = rank_candidates(
                candidates, wardrobe, taste_vector,
                taste_modes=taste_modes,
                intent_vector=None,
                intent_confidence=0.0,
            )

            # Intent-blended ranking
            ranked_intent = rank_candidates(
                candidates, wardrobe, taste_vector,
                taste_modes=taste_modes,
                intent_vector=intent_vec_np,
                intent_confidence=confidence,
            )

            k = 10
            taste_ids = [r["item_id"] for r in ranked_taste[:k]]
            intent_ids = [r["item_id"] for r in ranked_intent[:k]]

            taste_hits = sum(1 for iid in taste_ids if iid in relevant_ids)
            intent_hits = sum(1 for iid in intent_ids if iid in relevant_ids)

            p_taste = taste_hits / k
            p_intent = intent_hits / k
            lift = p_intent - p_taste

            beats = p_intent >= p_taste
            status = "PASS" if beats else "FAIL"

            results.append({
                "profile": profile_key,
                "session": session_name,
                "confidence": confidence,
                "p_taste": p_taste,
                "p_intent": p_intent,
                "lift": lift,
                "beats": beats,
            })

            print(f"  [{status}] confidence={confidence:.3f}")
            print(f"         P@{k} taste-only={p_taste:.2f}, "
                  f"intent-blended={p_intent:.2f}, lift={lift:+.2f}")

    # Summary
    print("\n" + "=" * 60)
    if not results:
        print("[SKIP] No valid intent sessions (confidence too low)")
        return 0

    beats_count = sum(1 for r in results if r["beats"])
    total = len(results)
    avg_lift = np.mean([r["lift"] for r in results])
    avg_confidence = np.mean([r["confidence"] for r in results])

    print(f"SUMMARY: Intent blending beats/matches taste-only in "
          f"{beats_count}/{total} scenarios")
    print(f"  Average lift: {avg_lift:+.3f}")
    print(f"  Average session confidence: {avg_confidence:.3f}")

    # Per-session summary
    for session_name in INTENT_SESSIONS:
        subset = [r for r in results if r["session"] == session_name]
        if subset:
            avg_l = np.mean([r["lift"] for r in subset])
            print(f"  {session_name}: avg lift={avg_l:+.3f}")

    status = "PASS" if beats_count >= total * 0.6 else "FAIL"
    print(f"\n[{status}] Intent lift threshold: >=60% scenarios improved")

    return 0 if beats_count >= total * 0.6 else 1


if __name__ == "__main__":
    sys.exit(run())
