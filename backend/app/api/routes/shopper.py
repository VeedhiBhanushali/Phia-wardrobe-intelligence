"""
Goal-oriented shopper: structured plan (LLM or heuristic) + existing retrieval/ranker.
"""

import json
import logging

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db.models import ShopperPlanRequest, ShopperPlanResponse, ShopperPlan
from app.core.candidates import generate_candidates, load_index, SLOT_PRIORITY
from app.core.ranker import rank_candidates
from app.core.wardrobe import (
    blend_vectors,
    build_wardrobe_embedding,
    compute_slot_coverage,
    get_gap_slots,
)
from app.core.orchestrator import clean_item

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shopper", tags=["shopper"])


def _wardrobe_items(item_ids: list[str]) -> list[dict]:
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        return []
    id_set = set(item_ids)
    return [i for i in catalog if i["item_id"] in id_set]


def _heuristic_plan(brief: dict, user_message: str | None) -> ShopperPlan:
    occ = str(brief.get("occasion") or "casual")
    msg = (user_message or "").lower()
    if any(w in msg for w in ("work", "office", "interview", "meeting")):
        occ = "work"
    elif any(w in msg for w in ("night", "date", "party", "evening", "going out")):
        occ = "evening"
    elif any(w in msg for w in ("weekend", "brunch", "errands")):
        occ = "weekend"
    elif any(w in msg for w in ("wedding", "formal", "gala")):
        occ = "special"

    slots = brief.get("slots_to_fill") or brief.get("gap_slots")
    if not slots:
        slots = ["tops", "bottoms", "shoes"]
    ordered = [s for s in SLOT_PRIORITY if s in slots]
    for s in slots:
        if s not in ordered:
            ordered.append(s)

    max_p = brief.get("max_price")
    if max_p is not None:
        try:
            max_p = float(max_p)
        except (TypeError, ValueError):
            max_p = None

    return ShopperPlan(
        occasion=occ,
        slots_to_fill=ordered,
        tone=str(brief.get("tone") or ""),
        max_price=max_p,
    )


def _openai_plan(brief: dict, user_message: str | None, api_key: str) -> ShopperPlan | None:
    schema_hint = (
        '{"occasion":"work|casual|evening|weekend|special",'
        '"slots_to_fill":["tops","bottoms",...],'
        '"tone":"short phrase","max_price":number_or_null}'
    )
    sys = (
        "You output only valid JSON matching this shape (no markdown): "
        f"{schema_hint}. "
        "Infer occasion and clothing slots from the user message and brief."
    )
    user = json.dumps({"brief": brief, "message": user_message or ""})
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            raw = json.loads(text)
            slots = raw.get("slots_to_fill") or ["tops", "bottoms", "shoes"]
            if not isinstance(slots, list):
                slots = ["tops", "bottoms", "shoes"]
            mp = raw.get("max_price")
            if mp is not None:
                try:
                    mp = float(mp)
                except (TypeError, ValueError):
                    mp = None
            occ = str(raw.get("occasion") or "casual")
            if occ not in ("work", "casual", "evening", "weekend", "special"):
                occ = "casual"
            return ShopperPlan(
                occasion=occ,
                slots_to_fill=slots,
                tone=str(raw.get("tone") or ""),
                max_price=mp,
            )
    except Exception as e:
        logger.warning("OpenAI shopper plan failed: %s", e)
        return None


@router.post("/plan", response_model=ShopperPlanResponse)
async def shopper_plan(req: ShopperPlanRequest):
    """
    Produce a structured shopping plan and ranked catalog picks.
    Uses OpenAI when OPENAI_API_KEY is set; otherwise a deterministic heuristic.
    """
    taste = np.array(req.taste_vector, dtype=np.float32)
    if taste.size == 0:
        raise HTTPException(status_code=400, detail="taste_vector is required")

    settings = get_settings()
    plan: ShopperPlan
    if settings.openai_api_key:
        llm_plan = _openai_plan(req.user_brief_json, req.user_message, settings.openai_api_key)
        plan = llm_plan or _heuristic_plan(req.user_brief_json, req.user_message)
    else:
        plan = _heuristic_plan(req.user_brief_json, req.user_message)

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    wardrobe = _wardrobe_items(req.wardrobe_item_ids)
    wardrobe_emb = build_wardrobe_embedding(wardrobe)
    save_count = len(wardrobe)

    occ_vecs = req.occasion_vectors or {}
    occ_list = occ_vecs.get(plan.occasion)
    if occ_list:
        q = np.array(occ_list, dtype=np.float32)
    else:
        q = taste

    query = blend_vectors(q, wardrobe_emb, save_count)

    coverage = compute_slot_coverage(wardrobe)
    gap_slots = get_gap_slots(coverage)
    slots = plan.slots_to_fill or gap_slots
    if not slots:
        slots = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

    price_tier = tuple(req.price_tier) if req.price_tier else (40.0, 200.0)
    if plan.max_price is not None and plan.max_price > 0:
        price_tier = (min(price_tier[0], plan.max_price * 0.3), float(plan.max_price))

    candidates = generate_candidates(
        taste_vector=query,
        gap_slots=slots,
        price_tier=price_tier,
        top_k=48,
    )

    anti = (
        np.array(req.anti_taste_vector, dtype=np.float32)
        if req.anti_taste_vector
        else None
    )
    trend_fp = req.trend_fingerprint or None

    ranked = rank_candidates(
        candidates,
        wardrobe,
        taste,
        taste_modes=[q, taste],
        trend_fingerprint=trend_fp,
        anti_taste_vector=anti,
    )

    items = [clean_item(r) for r in ranked[:16]]
    return ShopperPlanResponse(plan=plan, items=items)
