import logging
from fastapi import APIRouter

import numpy as np

from app.db.models import FeedRequest, FeedResponse
from app.core.candidates import load_index, generate_candidates
from app.core.ranker import rank_candidates
from app.core.wardrobe import (
    build_wardrobe_embedding,
    blend_vectors,
    compute_slot_coverage,
    get_gap_slots,
    get_wardrobe_stats,
)
from app.core.orchestrator import (
    build_negative_prototype,
    build_occasion_sections_unified,
    clean_item,
)
from app.core.outfit_builder import assemble_outfit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["feed"])


def _build_taste_percentile_fn(
    taste_vector: np.ndarray,
    catalog: list[dict],
    sample_size: int = 500,
    taste_modes: list[np.ndarray] | None = None,
):
    """Build a function that converts raw taste cosine → percentile (0–1).

    Samples the catalog to compute the empirical CDF of taste scores,
    then returns a closure that maps any raw score to its percentile
    among all catalog items.  This makes displayed scores meaningful:
    95% means "better taste match than 95% of the catalog."

    When taste_modes are provided, uses multi-mode scoring (max across
    modes) to match the ranker's _multi_mode_taste_score behaviour.
    """
    modes = taste_modes if taste_modes else [taste_vector]

    rng = np.random.RandomState(0)
    indices = rng.choice(len(catalog), size=min(sample_size, len(catalog)), replace=False)
    sample_scores = []
    for idx in indices:
        emb = np.array(catalog[idx]["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        best = max(max(0.0, float(np.dot(emb, m))) for m in modes)
        sample_scores.append(best)
    sample_scores.sort()
    arr = np.array(sample_scores)

    def to_percentile(raw_score: float) -> float:
        pct = float(np.searchsorted(arr, raw_score, side="right")) / len(arr)
        return min(pct, 0.99)

    return to_percentile


def _get_wardrobe_items(item_ids: list[str]) -> list[dict]:
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        return []
    id_set = set(item_ids)
    return [item for item in catalog if item["item_id"] in id_set]


@router.post("/feed", response_model=FeedResponse)
async def discovery_feed(req: FeedRequest):
    """
    Discovery feed: returns all sections needed for the feed UI.
    Accepts optional intent vector for session-aware ranking.
    """
    taste_vector = np.array(req.taste_vector, dtype=np.float32)
    taste_modes = (
        [np.array(m, dtype=np.float32) for m in req.taste_modes]
        if req.taste_modes else None
    )
    trend_fp = req.trend_fingerprint or None
    anti_taste = (
        np.array(req.anti_taste_vector, dtype=np.float32)
        if req.anti_taste_vector else None
    )
    intent_vector = (
        np.array(req.intent_vector, dtype=np.float32)
        if req.intent_vector else None
    )
    intent_confidence = req.intent_confidence
    style_attributes = req.style_attributes or {}
    price_tier = tuple(req.price_tier) if req.price_tier else (40.0, 200.0)

    wardrobe = _get_wardrobe_items(req.wardrobe_item_ids)
    stats = get_wardrobe_stats(wardrobe)
    coverage = compute_slot_coverage(wardrobe)
    gap_slots = get_gap_slots(coverage)
    if not gap_slots:
        gap_slots = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        return FeedResponse(wardrobeStats=stats)

    wardrobe_emb = build_wardrobe_embedding(wardrobe)
    save_count = len(wardrobe)
    query_vector = blend_vectors(taste_vector, wardrobe_emb, save_count)

    to_pct = _build_taste_percentile_fn(taste_vector, catalog, taste_modes=taste_modes)

    skip_set = set(req.skipped_item_ids or [])
    negative_proto = build_negative_prototype(catalog, req.skipped_item_ids or [])
    seen_ids: set[str] = set(skip_set) | {item["item_id"] for item in wardrobe}

    # 1. Complete Your Closet — gap-targeted, top-15% gate (percentile-based)
    complete_your_closet = []
    try:
        candidates = generate_candidates(
            taste_vector=query_vector,
            gap_slots=gap_slots,
            price_tier=price_tier,
            top_k=80,
            exclude_ids=seen_ids,
            trend_fingerprint=trend_fp,
        )
        ranked = rank_candidates(
            candidates, wardrobe, taste_vector,
            taste_modes=taste_modes,
            trend_fingerprint=trend_fp,
            anti_taste_vector=anti_taste,
            negative_prototype=negative_proto,
            intent_vector=intent_vector,
            intent_confidence=intent_confidence,
            style_attributes=style_attributes,
        )
        for item in ranked:
            pct = to_pct(item["taste_score"])
            if pct < 0.85:
                continue
            seen_ids.add(item["item_id"])
            complete_your_closet.append({
                "item": clean_item(item),
                "taste_score": round(pct, 2),
                "unlock_count": item["unlock_count"],
                "explanation": f"Fills a {item['slot']} gap — unlocks {item['unlock_count']} new outfits",
            })
            if len(complete_your_closet) >= 8:
                break
    except Exception:
        logger.exception("Error generating completeYourCloset")

    # 2. Your Aesthetic — pure taste-match, works from cold start
    your_aesthetic = []
    try:
        aesthetic_candidates = generate_candidates(
            taste_vector=taste_vector,
            gap_slots=["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"],
            price_tier=price_tier,
            top_k=40,
            exclude_ids=seen_ids,
            trend_fingerprint=trend_fp,
        )
        aesthetic_ranked = rank_candidates(
            aesthetic_candidates, wardrobe, taste_vector,
            taste_modes=taste_modes,
            trend_fingerprint=trend_fp,
            anti_taste_vector=anti_taste,
            negative_prototype=negative_proto,
            style_attributes=style_attributes,
        )
        for item in aesthetic_ranked[:8]:
            seen_ids.add(item["item_id"])
            your_aesthetic.append({
                "item": clean_item(item),
                "taste_score": round(to_pct(item["taste_score"]), 2),
                "unlock_count": item.get("unlock_count", 0),
            })
    except Exception:
        logger.exception("Error generating yourAesthetic")

    # 3. Complete Your Outfits — needs 3+ saves
    complete_your_outfits = []
    if save_count >= 3:
        try:
            # Infer top 2 occasions from wardrobe
            occasions_to_try = ["work", "casual"]
            if req.occasion_vectors:
                occasions_to_try = list(req.occasion_vectors.keys())[:2]

            for occ in occasions_to_try:
                outfit = assemble_outfit(
                    wardrobe=wardrobe,
                    occasion=occ,
                    taste_vector=taste_vector,
                    taste_modes=taste_modes,
                    trend_fingerprint=trend_fp,
                    anti_taste_vector=anti_taste,
                    price_tier=price_tier,
                )
                if outfit.get("catalog_addition"):
                    seen_ids.add(outfit["catalog_addition"]["item_id"])
                complete_your_outfits.append(outfit)
        except Exception:
            logger.exception("Error generating completeYourOutfits")

    # 4. Best Prices on Your Saves — wardrobe items sorted by resale value
    best_prices_on_saves = []
    if wardrobe:
        sorted_by_price = sorted(wardrobe, key=lambda x: x.get("price", 0))
        for item in sorted_by_price[:6]:
            best_prices_on_saves.append({
                "item": clean_item(item),
                "price": item.get("price", 0),
                "savings_ratio": round(0.3 + (item.get("price", 100) / 500) * 0.2, 2),
            })

    # 5. Occasion Rows
    occasion_rows = []
    try:
        if req.occasion_vectors:
            occasion_sections = build_occasion_sections_unified(
                occasion_vectors=req.occasion_vectors,
                wardrobe=wardrobe,
                taste_vector=taste_vector,
                taste_modes=taste_modes,
                trend_fp=trend_fp,
                anti_taste=anti_taste,
                negative_prototype=negative_proto,
                gap_slots=gap_slots,
                price_tier=price_tier,
                exclude_ids=seen_ids,
                per_section=6,
                style_attributes=style_attributes,
            )
            for section in occasion_sections:
                for pick in section.get("items", []):
                    pick["taste_score"] = round(to_pct(pick["taste_score"]), 2)
            occasion_rows = occasion_sections
    except Exception:
        logger.exception("Error generating occasion rows")

    return FeedResponse(
        completeYourCloset=complete_your_closet,
        yourAesthetic=your_aesthetic,
        completeYourOutfits=complete_your_outfits,
        bestPricesOnSaves=best_prices_on_saves,
        occasionRows=occasion_rows,
        wardrobeStats=stats,
    )
