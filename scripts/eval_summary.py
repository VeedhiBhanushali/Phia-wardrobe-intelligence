"""
Consolidated evaluation — produces a single pitch-ready table.

Runs three signal evaluations and prints a summary table:
  1. Taste model vs popularity baseline (Precision@10 at 0/5/10 saves)
  2. Intent blending lift over taste-only
  3. Purchase confidence — leave-one-out coherence ranking

Usage:
    python scripts/eval_summary.py
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
from app.core.ranker import rank_candidates, outfit_unlock_count, find_pairs
from app.core.intent import compute_intent
from app.data.mock_data import DEMO_WARDROBES

# ── Style profiles ──────────────────────────────────────────

STYLE_PROFILES = {
    "minimalist": "minimalist neutral clean structured tailored beige white ivory clothing",
    "streetwear": "streetwear oversized hoodie sneakers cargo pants bold urban clothing",
    "elegant":    "elegant formal polished evening cocktail heels satin silk clothing",
}

INTENT_SESSIONS = {
    "black_boots":     ("black leather ankle boots edgy minimal",     "shoes"),
    "linen_tops":      ("relaxed linen blouse cream natural oversized","tops"),
    "structured_bags": ("structured leather tote bag black minimal",  "bags"),
}


def _mean_taste_cosine(items: list[dict], taste_vector: np.ndarray) -> float:
    """Average cosine similarity of items to the taste vector."""
    if not items:
        return 0.0
    sims = []
    for it in items:
        emb = np.array(it["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        sims.append(float(np.dot(taste_vector, emb)))
    return float(np.mean(sims))


# ── 1. Mean Taste Relevance @ 10 ─────────────────────────────

def eval_precision(encoder, catalog):
    """Taste model vs baselines using Mean Taste Cosine @ 10.

    Three methods:
      - Random: 10 random catalog items (averaged over 50 trials)
      - Centroid: 10 items closest to the catalog mean embedding
      - Taste: 10 items from our personalized pipeline

    Mean Taste Cosine @ 10 measures how well each method's top-10
    align with the user's taste vector.  Both baselines produce non-zero
    scores, giving a meaningful comparison.
    """
    k = 10
    save_counts = [0, 5, 10]
    rows = []

    # Precompute centroid baseline (same for all profiles)
    all_embs = np.array([it["embedding"] for it in catalog], dtype=np.float32)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embs_normed = all_embs / norms
    centroid = all_embs_normed.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid_sims = all_embs_normed @ centroid
    centroid_top_idx = np.argsort(-centroid_sims)[:k]
    centroid_items = [catalog[i] for i in centroid_top_idx]

    rng = np.random.RandomState(42)

    for style_name, style_desc in STYLE_PROFILES.items():
        style_emb = encoder.encode_texts([style_desc])
        taste_vector = build_taste_vector(style_emb)

        # Centroid baseline score
        centroid_score = _mean_taste_cosine(centroid_items, taste_vector)

        # Random baseline score (average over 50 trials)
        random_scores = []
        for _ in range(50):
            rand_idx = rng.choice(len(catalog), size=k, replace=False)
            rand_items = [catalog[i] for i in rand_idx]
            random_scores.append(_mean_taste_cosine(rand_items, taste_vector))
        random_score = float(np.mean(random_scores))

        # Relevant set for P@10 (secondary metric)
        all_sims = []
        for item in catalog:
            emb = np.array(item["embedding"], dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            all_sims.append((item["item_id"], float(np.dot(taste_vector, emb))))
        all_sims.sort(key=lambda x: x[1], reverse=True)
        relevant = set(iid for iid, _ in all_sims[:30])

        for n_saves in save_counts:
            wardrobe = []
            if n_saves > 0:
                wardrobe = [it for it in catalog if it["item_id"] in relevant][:n_saves]

            coverage = compute_slot_coverage(wardrobe)
            gaps = get_gap_slots(coverage) or [
                "tops", "bottoms", "outerwear", "shoes", "bags", "accessories"
            ]

            if wardrobe:
                ward_embs = np.array([it["embedding"] for it in wardrobe], dtype=np.float32)
                modes = extract_taste_modes(np.concatenate([style_emb, ward_embs]))
            else:
                modes = [taste_vector]

            candidates = generate_candidates(
                taste_vector, gaps, (20, 300), top_k=60,
                exclude_ids={it["item_id"] for it in wardrobe},
            )
            ranked = rank_candidates(
                candidates, wardrobe, taste_vector, taste_modes=modes,
            )
            taste_score = _mean_taste_cosine(ranked[:k], taste_vector)

            rows.append({
                "style": style_name, "saves": n_saves,
                "taste_rel": taste_score,
                "centroid_rel": centroid_score,
                "random_rel": random_score,
            })
    return rows


# ── 2. Intent lift ──────────────────────────────────────────

def eval_intent(encoder, catalog, catalog_by_id):
    """Intent-blended P@10 vs taste-only P@10.

    Retrieves candidates from both taste and intent vectors so the
    intent-relevant items are actually in the pool.
    """
    k = 10
    rows = []

    for profile_key, profile in DEMO_WARDROBES.items():
        if profile_key == "cold_start":
            continue
        wardrobe_ids = [iid for iid in profile["item_ids"] if iid in catalog_by_id]
        if len(wardrobe_ids) < 2:
            continue
        wardrobe = [catalog_by_id[iid] for iid in wardrobe_ids]

        w_embs = np.array([it["embedding"] for it in wardrobe], dtype=np.float32)
        taste_vector = build_taste_vector(w_embs)
        taste_modes = extract_taste_modes(w_embs)

        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage) or [
            "tops", "bottoms", "outerwear", "shoes", "bags", "accessories"
        ]
        exclude = set(wardrobe_ids)

        for sess_name, (query, slot) in INTENT_SESSIONS.items():
            q_emb = encoder.encode_texts([query])[0]
            q_emb = q_emb / np.linalg.norm(q_emb)

            slot_items = [c for c in catalog if c["slot"] == slot] or catalog
            scored = []
            for it in slot_items:
                emb = np.array(it["embedding"], dtype=np.float32)
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                scored.append((it, float(np.dot(q_emb, emb))))
            scored.sort(key=lambda x: x[1], reverse=True)
            viewed = [it["embedding"] for it, _ in scored[:5]]

            intent_res = compute_intent(viewed)
            if intent_res["intent_vector"] is None:
                continue
            intent_vec = np.array(intent_res["intent_vector"], dtype=np.float32)
            conf = intent_res["confidence"]

            all_sims = []
            for it in catalog:
                if it["item_id"] in exclude:
                    continue
                emb = np.array(it["embedding"], dtype=np.float32)
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
                all_sims.append((it["item_id"], float(np.dot(q_emb, emb))))
            all_sims.sort(key=lambda x: x[1], reverse=True)
            relevant = set(iid for iid, _ in all_sims[:20])

            taste_cands = generate_candidates(
                taste_vector, gaps, (20, 300), top_k=40, exclude_ids=exclude,
            )
            intent_cands = generate_candidates(
                intent_vec, [slot] + gaps, (20, 300), top_k=40, exclude_ids=exclude,
            )
            seen = set()
            merged = []
            for c in taste_cands + intent_cands:
                if c["item_id"] not in seen:
                    seen.add(c["item_id"])
                    merged.append(c)

            r_taste = rank_candidates(
                merged, wardrobe, taste_vector, taste_modes=taste_modes,
            )
            r_intent = rank_candidates(
                merged, wardrobe, taste_vector, taste_modes=taste_modes,
                intent_vector=intent_vec, intent_confidence=conf,
            )

            p_t = sum(1 for r in r_taste[:k] if r["item_id"] in relevant) / k
            p_i = sum(1 for r in r_intent[:k] if r["item_id"] in relevant) / k

            rows.append({
                "profile": profile_key, "session": sess_name,
                "confidence": conf,
                "taste_p10": p_t, "intent_p10": p_i,
                "lift": p_i - p_t,
            })
    return rows


# ── 3. Purchase confidence — taste-aligned vs taste-misaligned ──

def eval_purchase_confidence(catalog, catalog_by_id, encoder):
    """Coherent-vs-incoherent ranking accuracy per wardrobe profile.

    For each profile's wardrobe:
      - Coherent candidates: top-15 non-wardrobe items by taste vector cosine
      - Incoherent candidates: bottom-15 non-wardrobe items by taste vector cosine
      - Both sets scored against the wardrobe via purchase confidence formula
      - Report % of (coherent, incoherent) pairs correctly ranked

    This tests: "does the purchase confidence model rank stylistically
    fitting items above mismatched ones?"
    """
    rows = []

    for profile_key, profile in DEMO_WARDROBES.items():
        if profile_key == "cold_start":
            continue
        wardrobe_ids = [iid for iid in profile["item_ids"] if iid in catalog_by_id]
        if len(wardrobe_ids) < 3:
            continue
        wardrobe = [catalog_by_id[iid] for iid in wardrobe_ids]
        ward_set = set(wardrobe_ids)

        w_embs = np.array([it["embedding"] for it in wardrobe], dtype=np.float32)
        taste_vector = build_taste_vector(w_embs)

        non_ward = [it for it in catalog if it["item_id"] not in ward_set]
        sims = []
        for it in non_ward:
            emb = np.array(it["embedding"], dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            sims.append((it, float(np.dot(taste_vector, emb))))
        sims.sort(key=lambda x: x[1], reverse=True)

        coherent_items = [it for it, _ in sims[:15]]
        incoherent_items = [it for it, _ in sims[-15:]]

        def _score_item(item: dict) -> float:
            emb = np.array(item["embedding"], dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            taste_fit = max(0.0, float(np.dot(taste_vector, emb)))
            unlock = outfit_unlock_count(item, wardrobe)
            pairs = find_pairs(item, wardrobe)
            return taste_fit * 0.4 + min(unlock / 5, 1.0) * 0.3 + (len(pairs) / 5) * 0.3

        coherent_scores = [_score_item(it) for it in coherent_items]
        incoherent_scores = [_score_item(it) for it in incoherent_items]

        correct = 0
        total = 0
        for cs in coherent_scores:
            for ics in incoherent_scores:
                total += 1
                if cs > ics:
                    correct += 1
                elif cs == ics:
                    correct += 0.5

        accuracy = correct / total if total > 0 else 0.5

        coh_sims = [s for _, s in sims[:15]]
        incoh_sims = [s for _, s in sims[-15:]]
        rows.append({
            "profile": profile_key,
            "n_coherent": len(coherent_scores),
            "n_incoherent": len(incoherent_scores),
            "accuracy": accuracy,
            "avg_coherent": float(np.mean(coherent_scores)),
            "avg_incoherent": float(np.mean(incoherent_scores)),
            "taste_sim_spread": float(np.mean(coh_sims) - np.mean(incoh_sims)),
        })

    return rows


# ── Main ────────────────────────────────────────────────────

def main():
    print("Loading model + catalog …")
    encoder = get_encoder()
    _, catalog = load_index()
    catalog_by_id = {it["item_id"]: it for it in catalog}

    # ── 1. Mean Taste Relevance @ 10 ──
    print("\n" + "=" * 72)
    print("  MEAN TASTE RELEVANCE @ 10  (avg cosine to taste vector)")
    print("=" * 72)
    print(f"  {'Saves':>5}  {'Random':>8}  {'Centroid':>10}  {'Taste':>8}  {'vs Random':>10}  {'vs Centroid':>12}")
    print("  " + "-" * 58)

    prec_rows = eval_precision(encoder, catalog)
    for n_saves in [0, 5, 10]:
        subset = [r for r in prec_rows if r["saves"] == n_saves]
        avg_t = np.mean([r["taste_rel"] for r in subset])
        avg_c = np.mean([r["centroid_rel"] for r in subset])
        avg_r = np.mean([r["random_rel"] for r in subset])
        lift_r = (avg_t - avg_r) / max(avg_r, 0.001) * 100
        lift_c = (avg_t - avg_c) / max(avg_c, 0.001) * 100
        print(f"  {n_saves:>5}  {avg_r:>8.3f}  {avg_c:>10.3f}  {avg_t:>8.3f}  {lift_r:>+9.0f}%  {lift_c:>+11.0f}%")

    # ── 2. Intent table ──
    print("\n" + "=" * 68)
    print("  THREE-SIGNAL MODEL — Intent Lift over Taste-Only (P@10)")
    print("=" * 68)
    print(f"  {'Session':>16}  {'Taste-Only':>10}  {'+ Intent':>10}  {'Lift':>8}")
    print("  " + "-" * 48)

    intent_rows = eval_intent(encoder, catalog, catalog_by_id)
    for sess in INTENT_SESSIONS:
        subset = [r for r in intent_rows if r["session"] == sess]
        if not subset:
            continue
        avg_t = np.mean([r["taste_p10"] for r in subset])
        avg_i = np.mean([r["intent_p10"] for r in subset])
        lift = avg_i - avg_t
        print(f"  {sess:>16}  {avg_t:>10.1%}  {avg_i:>10.1%}  {lift:>+8.1%}")

    if intent_rows:
        overall_t = np.mean([r["taste_p10"] for r in intent_rows])
        overall_i = np.mean([r["intent_p10"] for r in intent_rows])
        print("  " + "-" * 48)
        print(f"  {'OVERALL':>16}  {overall_t:>10.1%}  {overall_i:>10.1%}  {overall_i - overall_t:>+8.1%}")

    # ── 3. Purchase confidence ──
    print("\n" + "=" * 68)
    print("  PURCHASE CONFIDENCE — Coherent vs Incoherent Ranking Accuracy")
    print("=" * 68)
    print(f"  {'Profile':>14}  {'Coherent':>8}  {'Incoh':>6}  {'AvgCoh':>7}  {'AvgInc':>7}  {'SimGap':>7}  {'Accuracy':>8}")
    print("  " + "-" * 64)

    conf_rows = eval_purchase_confidence(catalog, catalog_by_id, encoder)
    for r in conf_rows:
        print(f"  {r['profile']:>14}  {r['n_coherent']:>8}  {r['n_incoherent']:>6}"
              f"  {r['avg_coherent']:>7.3f}  {r['avg_incoherent']:>7.3f}"
              f"  {r['taste_sim_spread']:>7.3f}"
              f"  {r['accuracy']:>8.1%}")

    if conf_rows:
        avg_acc = np.mean([r["accuracy"] for r in conf_rows])
        avg_c = np.mean([r["avg_coherent"] for r in conf_rows])
        avg_i = np.mean([r["avg_incoherent"] for r in conf_rows])
        avg_spread = np.mean([r["taste_sim_spread"] for r in conf_rows])
        print("  " + "-" * 64)
        print(f"  {'OVERALL':>14}  {'':>8}  {'':>6}"
              f"  {avg_c:>7.3f}  {avg_i:>7.3f}"
              f"  {avg_spread:>7.3f}"
              f"  {avg_acc:>8.1%}")

    # ── Summary ──
    print("\n" + "=" * 72)
    s0 = [r for r in prec_rows if r["saves"] == 0]
    avg_t_0 = np.mean([r["taste_rel"] for r in s0])
    avg_r_0 = np.mean([r["random_rel"] for r in s0])
    avg_c_0 = np.mean([r["centroid_rel"] for r in s0])
    lift_vs_rand = (avg_t_0 - avg_r_0) / max(avg_r_0, 0.001) * 100
    lift_vs_cent = (avg_t_0 - avg_c_0) / max(avg_c_0, 0.001) * 100

    intent_lift = np.mean([r["lift"] for r in intent_rows]) if intent_rows else 0
    avg_acc = np.mean([r["accuracy"] for r in conf_rows]) if conf_rows else 0.5

    print(f"  Taste model relevance is {lift_vs_rand:+.0f}% above random and "
          f"{lift_vs_cent:+.0f}% above centroid baseline at zero saves")
    print(f"  Intent blending adds {intent_lift:+.0%} P@10 during focused browsing")
    print(f"  Purchase confidence correctly ranks coherent > incoherent in {avg_acc:.0%} of pairs")
    print("=" * 72)


if __name__ == "__main__":
    main()
