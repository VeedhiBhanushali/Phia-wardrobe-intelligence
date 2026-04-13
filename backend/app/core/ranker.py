"""
Outfit utility ranker.

Multi-signal scoring: taste modes, trend alignment, anti-taste
gating, category compatibility, aesthetic harmony, and outfit unlock count.
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

# Style-family text prompts for projecting embeddings into style space
STYLE_FAMILY_PROMPTS = [
    "minimalist structured tailoring clean lines sharp",
    "relaxed oversized streetwear casual comfortable",
    "luxe elevated polished sophisticated refined",
    "bohemian eclectic layered artistic natural",
    "sporty athletic functional modern technical",
    "romantic feminine soft delicate graceful",
]

# Color harmony rules: pairs that look good together
_COLOR_HARMONY_PAIRS: set[frozenset[str]] = {
    frozenset({"navy", "white"}),
    frozenset({"camel", "black"}),
    frozenset({"camel", "navy"}),
    frozenset({"camel", "white"}),
    frozenset({"black", "white"}),
    frozenset({"navy", "ivory"}),
    frozenset({"grey", "black"}),
    frozenset({"grey", "navy"}),
    frozenset({"beige", "black"}),
    frozenset({"beige", "navy"}),
    frozenset({"ivory", "black"}),
    frozenset({"brown", "ivory"}),
    frozenset({"brown", "beige"}),
    frozenset({"olive", "beige"}),
    frozenset({"olive", "ivory"}),
    frozenset({"tan", "navy"}),
    frozenset({"tan", "ivory"}),
    frozenset({"burgundy", "black"}),
    frozenset({"burgundy", "ivory"}),
    frozenset({"rust", "ivory"}),
    frozenset({"rust", "beige"}),
}

# Neutrals pair with almost anything
_NEUTRAL_COLORS = {"black", "white", "grey", "ivory", "beige", "navy"}

_style_family_embeddings: np.ndarray | None = None


def _get_style_family_embeddings() -> np.ndarray:
    """Get cached style family text embeddings."""
    global _style_family_embeddings
    if _style_family_embeddings is None:
        from app.core.clip_encoder import get_encoder
        encoder = get_encoder()
        _style_family_embeddings = encoder.encode_texts(STYLE_FAMILY_PROMPTS)
    return _style_family_embeddings


def _project_to_style_space(embedding: np.ndarray) -> np.ndarray:
    """Project a CLIP embedding into low-dimensional style space."""
    style_embs = _get_style_family_embeddings()
    return (embedding @ style_embs.T).astype(np.float32)


def color_harmony_score(color_a: str, color_b: str) -> float:
    """Score color pairing: 1.0 for classic pairs, 0.5 for neutral+anything, 0.0 otherwise."""
    if not color_a or not color_b:
        return 0.0
    if color_a == color_b:
        return 0.3  # same color is okay but not ideal
    pair = frozenset({color_a, color_b})
    if pair in _COLOR_HARMONY_PAIRS:
        return 1.0
    if color_a in _NEUTRAL_COLORS or color_b in _NEUTRAL_COLORS:
        return 0.5
    return 0.0

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
    candidate_color: str = "",
    anchor_color: str = "",
) -> float:
    """
    Aesthetic harmony score: style-family coherence minus visual redundancy
    penalty, plus color harmony bonus. Replaces raw cosine similarity which
    rewarded visual sameness (e.g. leather on leather).
    """
    pair = (anchor_slot, candidate_slot)
    base = COMPLEMENT_RULES.get(pair, 0.0)
    if base == 0.0:
        return 0.0

    c_emb = np.array(candidate_embedding, dtype=np.float32)
    a_emb = np.array(anchor_embedding, dtype=np.float32)

    c_norm = np.linalg.norm(c_emb)
    a_norm = np.linalg.norm(a_emb)
    if c_norm > 0:
        c_emb = c_emb / c_norm
    if a_norm > 0:
        a_emb = a_emb / a_norm

    # Style-space coherence: project both into style families and measure agreement
    c_style = _project_to_style_space(c_emb)
    a_style = _project_to_style_space(a_emb)
    style_norm_c = np.linalg.norm(c_style)
    style_norm_a = np.linalg.norm(a_style)
    if style_norm_c > 0 and style_norm_a > 0:
        style_coherence = float(np.dot(c_style, a_style) / (style_norm_c * style_norm_a))
    else:
        style_coherence = 0.0

    # Visual redundancy penalty: raw cosine similarity > 0.82 means items look too alike
    raw_sim = float(np.dot(c_emb, a_emb))
    redundancy_penalty = max(0.0, (raw_sim - 0.82) * 3.0)  # steep penalty above threshold

    # Color harmony bonus
    color_bonus = color_harmony_score(candidate_color, anchor_color) * 0.15

    harmony = base * max(style_coherence, 0.0) - redundancy_penalty + color_bonus
    return max(harmony, 0.0)


def aggregate_compatibility(
    candidate: dict, wardrobe: list[dict]
) -> float:
    if not wardrobe:
        return 0.0

    scores = []
    c_emb = candidate["embedding"]
    c_slot = candidate["slot"]
    c_color = candidate.get("dominant_color", "")

    for anchor in wardrobe:
        a_emb = anchor.get("embedding")
        a_slot = anchor.get("slot", "")
        if a_emb is None or a_slot == c_slot:
            continue
        a_color = anchor.get("dominant_color", "")
        score = compatibility_score(
            c_emb, a_emb, c_slot, a_slot, c_color, a_color
        )
        if score > 0:
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def outfit_unlock_count(
    candidate: dict, wardrobe: list[dict]
) -> int:
    """
    Count outfit combinations this candidate enables, weighted by
    aesthetic compatibility.  Only wardrobe items with a positive
    compatibility_score to this specific candidate are counted,
    so two items in the same slot produce different unlock numbers.
    """
    c_emb = candidate.get("embedding")
    c_slot = candidate.get("slot", "")
    c_color = candidate.get("dominant_color", "")

    if not c_emb or not wardrobe:
        return 0

    compatible_counts: dict[str, int] = {}
    for item in wardrobe:
        w_slot = item.get("slot", "")
        if w_slot == c_slot:
            continue
        w_emb = item.get("embedding")
        if w_emb is None:
            continue
        score = compatibility_score(
            c_emb, w_emb, c_slot, w_slot,
            c_color, item.get("dominant_color", ""),
        )
        if score > 0:
            compatible_counts[w_slot] = compatible_counts.get(w_slot, 0) + 1

    total = 0
    for combo in VALID_OUTFIT_COMBOS:
        if c_slot not in combo:
            continue
        other_slots = [s for s in combo if s != c_slot]
        if all(s in compatible_counts for s in other_slots):
            perms = 1
            for s in other_slots:
                perms *= compatible_counts[s]
            total += perms

    return total


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


_style_axis_probes_ranker: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _get_style_axis_probes_ranker() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lazy-load cached axis probes for use inside the ranker (no encoder arg needed)."""
    global _style_axis_probes_ranker
    if _style_axis_probes_ranker is not None:
        return _style_axis_probes_ranker
    from app.core.clip_encoder import get_encoder
    from app.core.taste import _get_style_axis_probes
    encoder = get_encoder()
    _style_axis_probes_ranker = _get_style_axis_probes(encoder)
    return _style_axis_probes_ranker


def _attribute_profile_penalty(
    c_emb: np.ndarray,
    style_attributes: dict[str, float],
    confidence_gate: float = 0.35,
    per_axis_weight: float = 0.06,
    max_total_penalty: float = 0.30,
) -> float:
    """Compute a compound attribute-mismatch penalty for one candidate item.

    For each axis where the user has a clear preference (|score| > confidence_gate)
    and the item mismatches that preference, accumulates a penalty proportional to:
        - how strong the user preference is (confidence above gate)
        - how much the item mismatches (item_axis_score in the wrong direction)

    The total penalty is capped so no single item is obliterated by many weak mismatches.
    """
    if not style_attributes:
        return 0.0

    probes = _get_style_axis_probes_ranker()
    total = 0.0

    for key, user_pref in style_attributes.items():
        if abs(user_pref) < confidence_gate:
            continue  # No strong preference — don't penalise
        if key not in probes:
            continue

        pos_probe, neg_probe = probes[key]
        # Item score on this axis: positive = leans toward positive pole
        item_score = float(np.dot(c_emb, pos_probe) - np.dot(c_emb, neg_probe))

        # Mismatch: user_pref > 0 (likes positive pole) but item_score < 0, and vice versa
        mismatch = -user_pref * item_score
        if mismatch <= 0:
            continue  # Item aligns with preference — no penalty

        # Scale by confidence above gate (0..1) and item mismatch magnitude
        confidence = (abs(user_pref) - confidence_gate) / (1.0 - confidence_gate)
        total += confidence * mismatch * per_axis_weight

    return min(total, max_total_penalty)


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
    intent_vector: np.ndarray | None = None,
    intent_confidence: float = 0.0,
    intent_bias: float = 0.0,
    query_vector: np.ndarray | None = None,
    style_attributes: dict[str, float] | None = None,
) -> list[dict]:
    """
    Multi-signal ranking with explicit intent support.

    intent_bias (0.0–1.0) shifts scoring from taste-driven to query-driven:
      0.0 = pure taste/discovery (default for feed)
      0.8 = mostly query relevance (for explicit "find me a red dress")
    When intent_bias > 0, query_vector similarity dominates the score.
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

    has_session_intent = intent_vector is not None and intent_confidence > 0.3
    if has_session_intent:
        session_intent_weight = min(intent_confidence, 0.55)
        taste_scale = 1.0 - session_intent_weight
    else:
        session_intent_weight = 0.0
        taste_scale = 1.0

    # When intent_bias is active, reduce taste/trend/utility weights proportionally
    discovery_weight = 1.0 - intent_bias

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

        # Session intent fit (browsing-based)
        session_intent_fit = 0.0
        if has_session_intent:
            session_intent_fit = max(0.0, float(np.dot(c_emb_normed, intent_vector)))

        # Query intent fit (explicit search query)
        query_fit = 0.0
        if query_vector is not None:
            q = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            query_fit = max(0.0, float(np.dot(c_emb_normed, q)))

        t_boost = 0.0
        if trend_fingerprint:
            t_boost = trend_boost_score(c_emb_normed, trend_fingerprint)

        anti_pen = 0.0
        if anti_taste_vector is not None:
            anti_pen = _anti_taste_penalty(c_emb_normed, anti_taste_vector)

        skip_pen = _negative_prototype_penalty(c_emb_normed, negative_prototype)

        # Multi-axis attribute mismatch penalty
        attr_pen = _attribute_profile_penalty(c_emb_normed, style_attributes or {})

        unlock = outfit_unlock_count(c, wardrobe)
        utility_score = min(unlock / 10.0, 1.0)
        compat = aggregate_compatibility(c, wardrobe)

        if is_cold_start:
            discovery_score = (
                0.65 * taste_scale * taste_fit
                + session_intent_weight * session_intent_fit
                + 0.25 * t_boost
                - 0.10 * anti_pen
                - w_skip * skip_pen
                - attr_pen
            )
        else:
            discovery_score = (
                w_taste * taste_scale * taste_fit
                + session_intent_weight * session_intent_fit
                + w_trend * t_boost
                + w_utility * (utility_score * 0.6 + compat * 0.4)
                - w_anti * anti_pen
                - w_skip * skip_pen
                - attr_pen
            )

        final = discovery_weight * discovery_score + intent_bias * query_fit

        scored.append({
            **c,
            "final_score": float(final),
            "taste_score": float(taste_fit),
            "query_score": float(query_fit),
            "intent_score": float(session_intent_fit),
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


def rank_shopping(
    candidates: list[dict],
    query_vector: np.ndarray,
    wardrobe: list[dict] | None = None,
    taste_vector: np.ndarray | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Shopping-oriented ranking: query relevance dominates.

    Score = 0.85 * query_similarity + 0.15 * taste_fit
    No MMR diversity penalty — the user asked for something specific,
    so we return the best matches ranked tightly by relevance.
    """
    wardrobe = wardrobe or []

    q = np.array(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    scored = []
    for c in candidates:
        c_emb = np.array(c["embedding"], dtype=np.float32)
        c_norm = np.linalg.norm(c_emb)
        if c_norm > 0:
            c_emb_normed = c_emb / c_norm
        else:
            c_emb_normed = c_emb

        query_sim = max(0.0, float(np.dot(c_emb_normed, q)))

        taste_fit = 0.0
        if taste_vector is not None and np.linalg.norm(taste_vector) > 0:
            taste_fit = max(0.0, float(np.dot(c_emb_normed, taste_vector)))

        final = 0.85 * query_sim + 0.15 * taste_fit

        scored.append({
            **c,
            "final_score": float(final),
            "query_score": float(query_sim),
            "taste_score": float(taste_fit),
            "unlock_count": 0,
            "compatibility_score": 0.0,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]


def find_pairs(
    item: dict, wardrobe: list[dict], top_k: int = 5
) -> list[dict]:
    item_emb = item.get("embedding")
    item_slot = item.get("slot", "")
    item_color = item.get("dominant_color", "")
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
        anchor_color = anchor.get("dominant_color", "")
        score = compatibility_score(
            item_emb, anchor_emb, item_slot, anchor_slot,
            item_color, anchor_color,
        )
        if score > 0:
            pairs.append({**anchor, "pair_score": score})

    pairs.sort(key=lambda x: x["pair_score"], reverse=True)
    return pairs[:top_k]
