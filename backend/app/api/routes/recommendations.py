import logging
from fastapi import APIRouter, HTTPException
import numpy as np

from app.db.models import (
    WardrobeRecommendationRequest,
    WardrobeRecommendationResponse,
    GapRecommendation,
    ScoredItem,
    OccasionSection,
    OutfitBundle,
    EvaluateItemRequest,
    EvaluateItemResponse,
    EvaluateItemV2Request,
    EvaluateItemV2Response,
)
from app.core.candidates import load_index
from app.core.ranker import find_pairs
from app.core.explainer import generate_explanation
from app.core.orchestrator import run_wardrobe_orchestration
from app.core.wardrobe import get_wardrobe_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def _get_wardrobe_items(item_ids: list[str]) -> list[dict]:
    """Look up full item data from catalog by IDs."""
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        return []

    id_set = set(item_ids)
    return [item for item in catalog if item["item_id"] in id_set]


def _clean_item(item: dict) -> dict:
    return {k: v for k, v in item.items() if k != "embedding"}


@router.post("/wardrobe", response_model=WardrobeRecommendationResponse)
async def wardrobe_recommendations(req: WardrobeRecommendationRequest):
    """
    Personal-shopper orchestration: gap fill, ranked picks, occasion rows,
    outfit bundles, and a structured shopping brief.
    """
    taste_vector = np.array(req.taste_vector, dtype=np.float32)

    if len(taste_vector) == 0:
        raise HTTPException(status_code=400, detail="taste_vector is required")

    taste_modes = (
        [np.array(m, dtype=np.float32) for m in req.taste_modes]
        if req.taste_modes
        else None
    )
    trend_fp = req.trend_fingerprint or None
    anti_taste = (
        np.array(req.anti_taste_vector, dtype=np.float32)
        if req.anti_taste_vector
        else None
    )

    wardrobe = _get_wardrobe_items(req.wardrobe_item_ids)
    gap_rec = None
    top_picks: list[ScoredItem] = []
    complete_the_look = None
    occasion_sections: list[OccasionSection] = []
    outfit_suggestions: list[OutfitBundle] = []
    shopping_brief: dict = {}
    stats: dict = {}

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        logger.warning("FAISS index not built; returning empty recommendations")
        return WardrobeRecommendationResponse(
            wardrobe_stats=get_wardrobe_stats(wardrobe),
        )

    try:
        intent_vector = (
            np.array(req.intent_vector, dtype=np.float32)
            if req.intent_vector else None
        )

        orch = run_wardrobe_orchestration(
            wardrobe=wardrobe,
            taste_vector=taste_vector,
            taste_modes=taste_modes,
            occasion_vectors=req.occasion_vectors or {},
            trend_fp=trend_fp,
            anti_taste=anti_taste,
            price_tier=tuple(req.price_tier) if req.price_tier else (40.0, 200.0),
            aesthetic_label=req.aesthetic_label,
            skipped_item_ids=req.skipped_item_ids or [],
            catalog=catalog,
            intent_vector=intent_vector,
            intent_confidence=req.intent_confidence,
        )
        stats = orch["stats"]
        shopping_brief = orch["shopping_brief"]
        complete_the_look = orch["complete_the_look"]

        for bundle in orch["outfit_suggestions"]:
            outfit_suggestions.append(OutfitBundle(
                label=bundle["label"],
                items=bundle["items"],
            ))

        for sec in orch["occasion_sections"]:
            occasion_sections.append(OccasionSection(
                occasion=sec["occasion"],
                label=sec["label"],
                items=[
                    ScoredItem(
                        item=i["item"],
                        taste_score=i["taste_score"],
                        unlock_count=i["unlock_count"],
                        explanation=i["explanation"],
                    )
                    for i in sec["items"]
                ],
            ))

        ranked = orch["ranked"]
        top_trend = orch["top_trend"]
        explain_stats = {
            **stats,
            "aesthetic_label": req.aesthetic_label,
            "top_trend": top_trend,
        }

        if ranked:
            top = ranked[0]
            pairs = find_pairs(top, wardrobe)
            explanation = generate_explanation(
                top,
                explain_stats,
                pairs,
                score_context=top.get("score_context"),
            )
            gap_rec = GapRecommendation(
                item=_clean_item(top),
                unlock_count=top["unlock_count"],
                taste_score=round(top["taste_score"], 2),
                explanation=explanation,
                confidence=round(top["final_score"], 2),
            )

            for item in ranked[1:]:
                item_explanation = generate_explanation(
                    item,
                    explain_stats,
                    score_context=item.get("score_context"),
                )
                top_picks.append(ScoredItem(
                    item=_clean_item(item),
                    taste_score=round(item["taste_score"], 2),
                    unlock_count=item["unlock_count"],
                    explanation=item_explanation,
                ))

    except Exception:
        logger.exception("Error generating recommendations")

    return WardrobeRecommendationResponse(
        gap_recommendation=gap_rec,
        top_picks=top_picks,
        occasion_sections=occasion_sections,
        outfit_suggestions=outfit_suggestions,
        shopping_brief=shopping_brief,
        complete_the_look=complete_the_look,
        wardrobe_stats=stats,
    )


@router.post("/evaluate-item", response_model=EvaluateItemResponse)
async def evaluate_item(req: EvaluateItemRequest):
    """
    Evaluate how well a product fits the user's wardrobe and taste.

    Returns taste fit, outfit unlock count, pairing suggestions,
    and a template explanation.
    """
    taste_vector = np.array(req.taste_vector, dtype=np.float32)

    if not req.item_id:
        raise HTTPException(status_code=400, detail="item_id is required")

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    item = next((i for i in catalog if i["item_id"] == req.item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found in catalog")

    wardrobe = _get_wardrobe_items(req.wardrobe_item_ids)

    item_emb = np.array(item["embedding"], dtype=np.float32)
    raw_sim = float(np.dot(taste_vector, item_emb))
    raw_taste_fit = max(0.0, min(1.0, raw_sim))

    from app.core.ranker import outfit_unlock_count
    from app.api.routes.feed import _build_taste_percentile_fn

    to_pct = _build_taste_percentile_fn(taste_vector, catalog)
    taste_fit = to_pct(raw_taste_fit)

    unlock = outfit_unlock_count(item, wardrobe)

    pairs = find_pairs(item, wardrobe)

    stats = get_wardrobe_stats(wardrobe)
    explanation = generate_explanation(
        {**item, "unlock_count": unlock, "taste_score": taste_fit},
        stats,
        pairs,
    )

    return EvaluateItemResponse(
        taste_fit=round(taste_fit, 2),
        unlock_count=unlock,
        pairs_with=[_clean_item(p) for p in pairs[:5]],
        explanation=explanation,
        confidence=round(taste_fit * 0.5 + min(unlock / 10, 1) * 0.5, 2),
    )


@router.post("/evaluate-item-v2", response_model=EvaluateItemV2Response)
async def evaluate_item_v2(req: EvaluateItemV2Request):
    """
    Enhanced item evaluation with intent match and purchase confidence.
    Taste fit is percentile-normalized against the catalog.
    """
    taste_vector = np.array(req.taste_vector, dtype=np.float32)

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    item = next((i for i in catalog if i["item_id"] == req.item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found in catalog")

    wardrobe = _get_wardrobe_items(req.wardrobe_item_ids)

    item_emb = np.array(item["embedding"], dtype=np.float32)
    item_norm = np.linalg.norm(item_emb)
    if item_norm > 0:
        item_emb_normed = item_emb / item_norm
    else:
        item_emb_normed = item_emb

    raw_taste_fit = max(0.0, min(1.0, float(np.dot(taste_vector, item_emb_normed))))

    # Percentile-normalize taste fit against the catalog
    from app.api.routes.feed import _build_taste_percentile_fn
    to_pct = _build_taste_percentile_fn(taste_vector, catalog)
    taste_fit_pct = to_pct(raw_taste_fit)

    # Intent match (also percentile-normalize if available)
    intent_match = None
    if req.intent_vector and req.intent_confidence > 0.3:
        intent_vec = np.array(req.intent_vector, dtype=np.float32)
        raw_intent = max(0.0, min(1.0, float(np.dot(intent_vec, item_emb_normed))))
        intent_pct_fn = _build_taste_percentile_fn(intent_vec, catalog)
        intent_match = intent_pct_fn(raw_intent)

    from app.core.ranker import outfit_unlock_count
    unlock = outfit_unlock_count(item, wardrobe)
    pairs = find_pairs(item, wardrobe)

    # Purchase confidence: taste-dominant scoring with wardrobe-scaled denominators
    # so unlock/pairs don't saturate at small wardrobe sizes
    w_size = max(len(wardrobe), 1)
    unlock_denom = max(w_size * 2, 10)
    pairs_denom = max(w_size * 2, 8)
    score = (
        raw_taste_fit * 0.50
        + min(unlock / unlock_denom, 1.0) * 0.25
        + min(len(pairs) / pairs_denom, 1.0) * 0.25
    )
    if intent_match is not None:
        score = score * 0.7 + intent_match * 0.3

    if score >= 0.35:
        confidence = "HIGH"
    elif score >= 0.18:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    stats = get_wardrobe_stats(wardrobe)
    explanation = generate_explanation(
        {**item, "unlock_count": unlock, "taste_score": taste_fit_pct},
        stats,
        pairs,
    )

    return EvaluateItemV2Response(
        taste_fit=round(taste_fit_pct, 2),
        intent_match=round(intent_match, 2) if intent_match is not None else None,
        purchase_confidence=confidence,
        unlock_count=unlock,
        pairs_with=[_clean_item(p) for p in pairs[:5]],
        explanation=explanation,
        best_price=item.get("price", 0),
    )
