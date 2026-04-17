"""
Outfit builder.

Given wardrobe items and an occasion, assembles the best complete outfit
(top + bottom + shoes minimum, optionally outerwear for work/evening).
If a required slot is missing, finds the best catalog addition.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.clip_encoder import get_encoder
from app.core.ranker import compatibility_score
from app.core.candidates import generate_candidates
from app.core.ranker import rank_candidates

logger = logging.getLogger(__name__)

OCCASION_PROMPTS: dict[str, str] = {
    "work": "professional office business polished tailored structured elegant",
    "casual": "casual everyday relaxed comfortable laid-back denim sneakers",
    "evening": "evening dressy cocktail party elegant glamorous heels statement",
    "weekend": "weekend brunch relaxed chic comfortable effortless cool",
    "special": "special occasion formal event standout dramatic bold",
}

# Required slots per occasion
OCCASION_SLOTS: dict[str, list[str]] = {
    "work": ["tops", "bottoms", "shoes", "outerwear"],
    "casual": ["tops", "bottoms", "shoes"],
    "evening": ["tops", "bottoms", "shoes", "outerwear"],
    "weekend": ["tops", "bottoms", "shoes"],
    "special": ["tops", "bottoms", "shoes"],
}

OCCASION_MIN_RELEVANCE = 0.21

_occasion_embeddings_cache: dict[str, np.ndarray] = {}


def _get_occasion_embedding(occasion: str) -> np.ndarray:
    """Get cached CLIP text embedding for an occasion prompt."""
    if occasion not in _occasion_embeddings_cache:
        encoder = get_encoder()
        prompt = OCCASION_PROMPTS.get(occasion, OCCASION_PROMPTS["casual"])
        emb = encoder.encode_texts([prompt])[0]
        _occasion_embeddings_cache[occasion] = emb
    return _occasion_embeddings_cache[occasion]


def _occasion_relevance(item: dict, occasion: str) -> float:
    """Score how relevant a wardrobe item is for an occasion using CLIP similarity."""
    emb = item.get("embedding")
    if emb is None:
        return 0.0
    item_emb = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(item_emb)
    if norm > 0:
        item_emb = item_emb / norm

    occ_emb = _get_occasion_embedding(occasion)
    return float(np.dot(item_emb, occ_emb))


def _score_combination(items: list[dict]) -> float:
    """Score an outfit combination by mean pairwise aesthetic harmony."""
    if len(items) < 2:
        return 0.0

    scores = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = compatibility_score(
                items[i]["embedding"],
                items[j]["embedding"],
                items[i]["slot"],
                items[j]["slot"],
                items[i].get("dominant_color", ""),
                items[j].get("dominant_color", ""),
            )
            scores.append(s)

    return float(np.mean(scores)) if scores else 0.0


def assemble_outfit(
    wardrobe: list[dict],
    occasion: str,
    taste_vector: np.ndarray | None = None,
    taste_modes: list[np.ndarray] | None = None,
    trend_fingerprint: dict[str, float] | None = None,
    anti_taste_vector: np.ndarray | None = None,
    price_tier: tuple[float, float] = (40.0, 200.0),
) -> dict[str, Any]:
    """
    Assemble the best complete outfit for an occasion.

    Strategy:
    1. Score wardrobe items for occasion relevance
    2. Find the best combination across required slots
    3. If a slot is missing, find the best catalog addition

    Returns:
        dict with: wardrobe_items, catalog_addition (optional), occasion,
        title, rationale, is_complete, harmony_score
    """
    required_slots = OCCASION_SLOTS.get(occasion, ["tops", "bottoms", "shoes"])

    # Score wardrobe items for occasion relevance, reject below threshold
    by_slot: dict[str, list[dict]] = {}
    for item in wardrobe:
        slot = item.get("slot", "")
        if slot not in required_slots:
            continue
        relevance = _occasion_relevance(item, occasion)
        if relevance < OCCASION_MIN_RELEVANCE:
            continue
        by_slot.setdefault(slot, []).append({**item, "_occ_relevance": relevance})

    # Sort each slot by occasion relevance
    for slot in by_slot:
        by_slot[slot].sort(key=lambda x: x["_occ_relevance"], reverse=True)

    # Find the best combination: pick top candidate per slot, then score the combo
    best_combo: list[dict] = []
    missing_slots: list[str] = []

    for slot in required_slots:
        candidates = by_slot.get(slot, [])
        if not candidates:
            missing_slots.append(slot)
            continue
        best_combo.append(candidates[0])

    # Try alternate combinations if we have multiple options
    if len(best_combo) >= 2:
        best_score = _score_combination(best_combo)

        # Try swapping each position with the second-best option
        for slot_idx, slot in enumerate(required_slots):
            candidates = by_slot.get(slot, [])
            if len(candidates) < 2:
                continue
            test_combo = list(best_combo)
            combo_slot_idx = next(
                (i for i, item in enumerate(test_combo) if item["slot"] == slot),
                None,
            )
            if combo_slot_idx is None:
                continue
            test_combo[combo_slot_idx] = candidates[1]
            test_score = _score_combination(test_combo)
            if test_score > best_score:
                best_combo = test_combo
                best_score = test_score

    # Find catalog addition for missing slots
    catalog_addition = None
    if missing_slots and taste_vector is not None:
        try:
            for gap_slot in missing_slots:
                gap_candidates = generate_candidates(
                    taste_vector=taste_vector,
                    gap_slots=[gap_slot],
                    price_tier=price_tier,
                    top_k=20,
                    exclude_ids={item["item_id"] for item in wardrobe},
                    trend_fingerprint=trend_fingerprint,
                )
                if gap_candidates:
                    ranked = rank_candidates(
                        gap_candidates,
                        wardrobe,
                        taste_vector,
                        taste_modes=taste_modes,
                        trend_fingerprint=trend_fingerprint,
                        anti_taste_vector=anti_taste_vector,
                    )
                    if ranked:
                        catalog_addition = ranked[0]
                        best_combo.append(catalog_addition)
                        break
        except Exception:
            logger.exception("Error finding catalog addition for outfit")

    harmony_score = _score_combination(best_combo)
    is_complete = len(missing_slots) == 0 or catalog_addition is not None

    # Clean items for output (remove embeddings and internal fields)
    def _clean(item: dict) -> dict:
        return {k: v for k, v in item.items()
                if k not in ("embedding", "_occ_relevance")}

    occasion_labels = {
        "work": "Office Ready",
        "casual": "Everyday Ease",
        "evening": "Night Out",
        "weekend": "Weekend Brunch",
        "special": "Statement Look",
    }

    return {
        "wardrobe_items": [_clean(item) for item in best_combo if item != catalog_addition],
        "catalog_addition": _clean(catalog_addition) if catalog_addition else None,
        "occasion": occasion,
        "title": occasion_labels.get(occasion, occasion.title()),
        "rationale": _generate_rationale(best_combo, missing_slots, catalog_addition, occasion),
        "is_complete": is_complete,
        "harmony_score": round(harmony_score, 3),
        "missing_slots": missing_slots if not is_complete else [],
    }


def curate_outfit_from_catalog(
    occasion: str,
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None = None,
    trend_fingerprint: dict[str, float] | None = None,
    anti_taste_vector: np.ndarray | None = None,
    price_tier: tuple[float, float] = (40.0, 200.0),
    wardrobe: list[dict] | None = None,
    context_items: list[dict] | None = None,
) -> dict[str, Any]:
    """Curate a full shoppable outfit from catalog, scored as a set.

    Unlike assemble_outfit (wardrobe-first + one gap fill), this retrieves
    candidates for every required slot from the catalog, filters by occasion
    relevance, then picks the combination with the best joint harmony score.

    context_items: previously recommended items to score compatibility against
                   (e.g. for follow-up "add a blazer" requests).
    """
    required_slots = OCCASION_SLOTS.get(occasion, ["tops", "bottoms", "shoes"])
    exclude_ids = set()
    if wardrobe:
        exclude_ids = {item["item_id"] for item in wardrobe}

    occ_prompts = {
        "work": "tailored structured professional polished office elegant",
        "casual": "casual everyday relaxed comfortable denim sneakers",
        "evening": "evening dressy cocktail glamorous elegant statement",
        "weekend": "weekend brunch relaxed chic comfortable effortless",
        "special": "special occasion formal dramatic standout bold",
    }
    encoder = get_encoder()
    occ_query = occ_prompts.get(occasion, occ_prompts["casual"])
    occ_emb = encoder.encode_texts([occ_query])[0]

    # Blend occasion embedding with taste vector so results reflect the user's style
    if taste_vector is not None and np.linalg.norm(taste_vector) > 0:
        tv = np.array(taste_vector, dtype=np.float32)
        tv_norm = np.linalg.norm(tv)
        if tv_norm > 0:
            tv = tv / tv_norm
        retrieval_vec = 0.5 * occ_emb + 0.5 * tv
        r_norm = np.linalg.norm(retrieval_vec)
        if r_norm > 0:
            retrieval_vec = retrieval_vec / r_norm
    else:
        retrieval_vec = occ_emb

    slot_candidates: dict[str, list[dict]] = {}
    for slot in required_slots:
        candidates = generate_candidates(
            taste_vector=retrieval_vec,
            gap_slots=[slot],
            price_tier=price_tier,
            top_k=30,
            exclude_ids=exclude_ids,
            trend_fingerprint=trend_fingerprint,
        )
        # Filter by occasion relevance
        filtered = []
        for c in candidates:
            emb = np.array(c["embedding"], dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            rel = float(np.dot(emb, occ_emb))
            if rel >= OCCASION_MIN_RELEVANCE:
                filtered.append({**c, "_occ_rel": rel})
        filtered.sort(key=lambda x: x["_occ_rel"], reverse=True)
        slot_candidates[slot] = filtered[:8]

    # Greedy set assembly: pick the best item per slot, then try swaps
    best_combo: list[dict] = []
    missing: list[str] = []
    for slot in required_slots:
        pool = slot_candidates.get(slot, [])
        if not pool:
            missing.append(slot)
            continue
        best_combo.append(pool[0])

    # If we have context items, include them in harmony scoring
    harmony_context = list(context_items or [])
    for c in harmony_context:
        if "embedding" not in c:
            continue

    def _combo_score(combo: list[dict]) -> float:
        full = combo + [c for c in harmony_context if "embedding" in c]
        return _score_combination(full) if len(full) >= 2 else 0.0

    if len(best_combo) >= 2:
        best_score = _combo_score(best_combo)
        for slot in required_slots:
            pool = slot_candidates.get(slot, [])
            if len(pool) < 2:
                continue
            idx = next((i for i, it in enumerate(best_combo) if it["slot"] == slot), None)
            if idx is None:
                continue
            for alt in pool[1:4]:
                test = list(best_combo)
                test[idx] = alt
                s = _combo_score(test)
                if s > best_score:
                    best_combo = test
                    best_score = s

    harmony = _score_combination(best_combo)

    def _clean(item: dict) -> dict:
        return {k: v for k, v in item.items() if k not in ("embedding", "_occ_rel")}

    occasion_labels = {
        "work": "Office Ready",
        "casual": "Everyday Ease",
        "evening": "Night Out",
        "weekend": "Weekend Brunch",
        "special": "Statement Look",
    }

    return {
        "items": [_clean(it) for it in best_combo],
        "occasion": occasion,
        "title": occasion_labels.get(occasion, occasion.title()),
        "harmony_score": round(harmony, 3),
        "missing_slots": missing,
        "filled_slots": [it["slot"] for it in best_combo],
    }


def _generate_rationale(
    combo: list[dict],
    missing_slots: list[str],
    catalog_addition: dict | None,
    occasion: str,
) -> str:
    """Generate a brief rationale for the outfit."""
    if not combo:
        return f"Not enough items in your wardrobe for a {occasion} outfit."

    item_names = [item.get("title", item.get("slot", "item")) for item in combo
                  if item != catalog_addition]

    if catalog_addition and missing_slots:
        slot = missing_slots[0]
        return (
            f"Built around your {item_names[0] if item_names else 'saves'}. "
            f"Adding {catalog_addition.get('title', 'a new piece')} "
            f"completes the {slot} gap."
        )

    if len(item_names) >= 2:
        return f"Your {item_names[0]} pairs well with {item_names[1]} for {occasion}."

    return f"A solid {occasion} look from your wardrobe."
