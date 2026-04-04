"""
Outfit utility ranker.

Multi-signal scoring: taste modes, trend alignment, anti-taste
gating, category compatibility, and outfit unlock count.
"""

import numpy as np
from app.config import get_settings
from app.core.trends import trend_boost_score

COMPLEMENT_RULES: dict[tuple[str, str], float] = {
    ("tops", "bottoms"): 0.9,
    ("tops", "shoes"): 0.7,
    ("tops", "outerwear"): 0.85,
    ("tops", "bags"): 0.5,
    ("tops", "accessories"): 0.4,
    ("bottoms", "tops"): 0.9,
    ("bottoms", "shoes"): 0.8,
    ("bottoms", "outerwear"): 0.75,
    ("bottoms", "bags"): 0.6,
    ("bottoms", "accessories"): 0.4,
    ("outerwear", "tops"): 0.85,
    ("outerwear", "bottoms"): 0.75,
    ("outerwear", "shoes"): 0.65,
    ("shoes", "tops"): 0.7,
    ("shoes", "bottoms"): 0.8,
    ("shoes", "bags"): 0.55,
    ("bags", "tops"): 0.5,
    ("bags", "bottoms"): 0.6,
    ("bags", "shoes"): 0.55,
    ("accessories", "tops"): 0.4,
    ("accessories", "bottoms"): 0.4,
    ("shoes", "outerwear"): 0.65,
}

VALID_OUTFIT_COMBOS = [
    frozenset({"tops", "bottoms"}),
    frozenset({"tops", "bottoms", "shoes"}),
    frozenset({"tops", "bottoms", "outerwear"}),
    frozenset({"tops", "bottoms", "outerwear", "shoes"}),
    frozenset({"tops", "bottoms", "shoes", "bags"}),
    frozenset({"tops", "bottoms", "outerwear", "shoes", "bags"}),
]


def compatibility_score(
    candidate_embedding: np.ndarray,
    anchor_embedding: np.ndarray,
    candidate_slot: str,
    anchor_slot: str,
) -> float:
    pair = (anchor_slot, candidate_slot)
    base = COMPLEMENT_RULES.get(pair, 0.0)
    if base == 0.0:
        return 0.0

    c_emb = np.array(candidate_embedding, dtype=np.float32)
    a_emb = np.array(anchor_embedding, dtype=np.float32)

    c_norm = np.linalg.norm(c_emb)
    a_norm = np.linalg.norm(a_emb)
    if c_norm > 0 and a_norm > 0:
        sim = float(np.dot(c_emb, a_emb) / (c_norm * a_norm))
    else:
        sim = 0.0

    return base * max(sim, 0.0)


def aggregate_compatibility(
    candidate: dict, wardrobe: list[dict]
) -> float:
    if not wardrobe:
        return 0.0

    scores = []
    c_emb = candidate["embedding"]
    c_slot = candidate["slot"]

    for anchor in wardrobe:
        a_emb = anchor.get("embedding")
        a_slot = anchor.get("slot", "")
        if a_emb is None or a_slot == c_slot:
            continue
        score = compatibility_score(c_emb, a_emb, c_slot, a_slot)
        if score > 0:
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def outfit_unlock_count(
    candidate: dict, wardrobe: list[dict]
) -> int:
    def _count_outfits(slot_items: dict[str, list]) -> int:
        total = 0
        for combo in VALID_OUTFIT_COMBOS:
            if all(slot in slot_items and len(slot_items[slot]) > 0 for slot in combo):
                perms = 1
                for slot in combo:
                    perms *= len(slot_items[slot])
                total += perms
        return total

    slot_items_before: dict[str, list] = {}
    for item in wardrobe:
        slot_items_before.setdefault(item["slot"], []).append(item)

    before = _count_outfits(slot_items_before)

    slot_items_after = {k: list(v) for k, v in slot_items_before.items()}
    slot_items_after.setdefault(candidate["slot"], []).append(candidate)

    after = _count_outfits(slot_items_after)

    return max(after - before, 0)


def _multi_mode_taste_score(
    c_emb: np.ndarray,
    taste_modes: list[np.ndarray],
) -> float:
    """Best cosine similarity across all taste modes."""
    best = 0.0
    for mode in taste_modes:
        sim = float(np.dot(c_emb, mode))
        if sim > best:
            best = sim
    return best


def _anti_taste_penalty(
    c_emb: np.ndarray,
    anti_taste_vector: np.ndarray | None,
    threshold: float = 0.3,
) -> float:
    """Penalty for items that match the user's avoided aesthetics."""
    if anti_taste_vector is None or np.linalg.norm(anti_taste_vector) == 0:
        return 0.0
    sim = float(np.dot(c_emb, anti_taste_vector))
    return max(sim - threshold, 0.0)


def _negative_prototype_penalty(
    c_emb: np.ndarray,
    negative_prototype: np.ndarray | None,
    threshold: float = 0.25,
) -> float:
    """Penalty for items similar to skipped / dismissed products."""
    if negative_prototype is None or np.linalg.norm(negative_prototype) == 0:
        return 0.0
    sim = float(np.dot(c_emb, negative_prototype))
    return max(sim - threshold, 0.0)


def mmr_rerank(
    scored: list[dict], k: int = 5, lam: float = 0.7
) -> list[dict]:
    if not scored:
        return []

    selected = [scored[0]]
    remaining = list(scored[1:])

    while len(selected) < k and remaining:
        best, best_mmr = None, -float("inf")
        for cand in remaining:
            rel = cand["final_score"]
            c_emb = np.array(cand.get("embedding", []), dtype=np.float32)
            if c_emb.size == 0:
                red = 0.0
            else:
                c_norm = np.linalg.norm(c_emb)
                if c_norm > 0:
                    c_emb = c_emb / c_norm
                sims = []
                for s in selected:
                    s_raw = s.get("embedding")
                    if not s_raw:
                        continue
                    s_emb = np.array(s_raw, dtype=np.float32)
                    s_norm = np.linalg.norm(s_emb)
                    if s_norm > 0:
                        s_emb = s_emb / s_norm
                    sims.append(float(np.dot(c_emb, s_emb)))
                red = max(sims) if sims else 0.0
            mmr = lam * rel - (1 - lam) * red
            if mmr > best_mmr:
                best, best_mmr = cand, mmr
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected


def rank_candidates(
    candidates: list[dict],
    wardrobe: list[dict],
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None = None,
    trend_fingerprint: dict[str, float] | None = None,
    anti_taste_vector: np.ndarray | None = None,
    negative_prototype: np.ndarray | None = None,
) -> list[dict]:
    """
    Multi-signal ranking:
      - taste_fit:  best cosine sim across taste modes (or single vector)
      - trend_boost: weighted alignment with user's top trend archetypes
      - anti_penalty: suppression of items matching avoided aesthetics
      - utility: outfit unlock count + category compatibility
    """
    settings = get_settings()
    w_taste = settings.rank_weight_taste
    w_trend = settings.rank_weight_trend
    w_utility = settings.rank_weight_utility
    w_anti = settings.rank_weight_anti
    w_skip = settings.rank_weight_skip
    threshold = settings.confidence_threshold

    if taste_modes is None:
        taste_modes = [taste_vector]

    is_cold_start = len(wardrobe) == 0

    scored = []
    for c in candidates:
        c_emb = np.array(c["embedding"], dtype=np.float32)
        c_norm = np.linalg.norm(c_emb)
        if c_norm > 0:
            c_emb_normed = c_emb / c_norm
        else:
            c_emb_normed = c_emb

        taste_fit = _multi_mode_taste_score(c_emb_normed, taste_modes)

        t_boost = 0.0
        if trend_fingerprint:
            t_boost = trend_boost_score(c_emb_normed, trend_fingerprint)

        anti_pen = 0.0
        if anti_taste_vector is not None:
            anti_pen = _anti_taste_penalty(c_emb_normed, anti_taste_vector)

        skip_pen = _negative_prototype_penalty(c_emb_normed, negative_prototype)

        unlock = outfit_unlock_count(c, wardrobe)
        utility_score = min(unlock / 10.0, 1.0)
        compat = aggregate_compatibility(c, wardrobe)

        if is_cold_start:
            final = (
                0.65 * taste_fit
                + 0.25 * t_boost
                - 0.10 * anti_pen
                - w_skip * skip_pen
            )
        else:
            final = (
                w_taste * taste_fit
                + w_trend * t_boost
                + w_utility * (utility_score * 0.6 + compat * 0.4)
                - w_anti * anti_pen
                - w_skip * skip_pen
            )

        scored.append({
            **c,
            "final_score": float(final),
            "taste_score": float(taste_fit),
            "trend_score": float(t_boost),
            "anti_taste_penalty": float(anti_pen),
            "skip_penalty": float(skip_pen),
            "unlock_count": unlock,
            "compatibility_score": float(compat),
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    if scored and scored[0]["final_score"] < threshold:
        return []

    selected = mmr_rerank(scored, k=10, lam=0.7)

    if selected:
        taste_scores = [c["taste_score"] for c in scored]
        for item in selected:
            rank = sum(1 for t in taste_scores if t > item["taste_score"])
            item["score_context"] = {
                "taste_percentile": 1.0 - (rank / max(len(taste_scores), 1)),
            }

    return selected


def find_pairs(
    item: dict, wardrobe: list[dict], top_k: int = 5
) -> list[dict]:
    item_emb = item.get("embedding")
    item_slot = item.get("slot", "")
    if not item_emb or not item_slot:
        return []

    pairs = []
    for anchor in wardrobe:
        if anchor.get("item_id") == item.get("item_id"):
            continue
        anchor_emb = anchor.get("embedding")
        anchor_slot = anchor.get("slot", "")
        if not anchor_emb or not anchor_slot:
            continue
        score = compatibility_score(item_emb, anchor_emb, item_slot, anchor_slot)
        if score > 0:
            pairs.append({**anchor, "pair_score": score})

    pairs.sort(key=lambda x: x["pair_score"], reverse=True)
    return pairs[:top_k]
