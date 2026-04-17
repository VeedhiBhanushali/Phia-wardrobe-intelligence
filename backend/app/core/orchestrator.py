"""
Composes retrieval, ranking, occasion sections, outfit bundles, and shopping brief.

Keeps route handlers thin; all multi-step recommendation logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.candidates import generate_candidates
from app.core.ranker import rank_candidates, compatibility_score, find_pairs
from app.core.wardrobe import (
    blend_vectors,
    build_wardrobe_embedding,
    compute_slot_coverage,
    get_gap_slots,
    get_wardrobe_stats,
)
from app.core.taste import OCCASION_LABELS

logger = logging.getLogger(__name__)


def clean_item(item: dict) -> dict:
    return {k: v for k, v in item.items() if k != "embedding"}


_BAD_IMAGE_TOKENS = {"fabric", "swatch", "detail", "close-up", "closeup", "texture", "sample"}

def is_bundle_eligible(item: dict) -> bool:
    """Exclude items whose title suggests a swatch, detail shot, or non-product image."""
    title = item.get("title", "").lower()
    return not any(tok in title for tok in _BAD_IMAGE_TOKENS)


def build_negative_prototype(
    catalog: list[dict], skipped_item_ids: list[str]
) -> np.ndarray | None:
    """L2-normalized mean embedding of skipped/dismissed catalog items."""
    if not skipped_item_ids:
        return None
    id_set = set(skipped_item_ids)
    embs = []
    for item in catalog:
        if item.get("item_id") not in id_set:
            continue
        emb = item.get("embedding")
        if emb is not None:
            embs.append(np.array(emb, dtype=np.float32))
    if not embs:
        return None
    proto = np.stack(embs, axis=0).mean(axis=0)
    n = np.linalg.norm(proto)
    if n > 0:
        proto = proto / n
    return proto.astype(np.float32)


def build_shopping_brief(
    stats: dict,
    price_tier: tuple[float, float],
    trend_fingerprint: dict[str, float] | None,
    occasion_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    top_trend = ""
    if trend_fingerprint:
        top_trend = max(trend_fingerprint, key=trend_fingerprint.get)

    occ_keys = list(occasion_vectors.keys()) if occasion_vectors else []
    labels = [OCCASION_LABELS.get(k, k.title()) for k in occ_keys]

    return {
        "gap_slots": stats.get("gap_slots", []),
        "price_tier": [float(price_tier[0]), float(price_tier[1])],
        "top_trend": top_trend,
        "dominant_occasions": labels,
        "wardrobe_item_count": stats.get("total_items", 0),
        "strongest_slot": stats.get("strongest_slot"),
    }


def _slots_for_outfit_anchor(anchor_slot: str) -> list[str]:
    base = ["tops", "bottoms", "shoes"]
    if anchor_slot in base:
        return base
    if anchor_slot == "outerwear":
        return ["tops", "bottoms", "outerwear"]
    if anchor_slot == "bags":
        return ["tops", "bottoms", "bags"]
    if anchor_slot == "accessories":
        return ["tops", "bottoms", "accessories"]
    return base


def build_outfit_suggestions(
    anchor: dict | None,
    pool: list[dict],
    max_outfits: int = 3,
) -> list[dict[str, Any]]:
    """
    Greedy complementary bundles sharing the same anchor; variants use
    next-best pieces per slot for diversity.
    """
    if not anchor or not pool:
        return []

    pool = [p for p in pool if is_bundle_eligible(p)]
    if not is_bundle_eligible(anchor):
        return []

    slots = _slots_for_outfit_anchor(anchor.get("slot", ""))
    if anchor.get("slot") not in slots:
        slots = [anchor["slot"]] + [s for s in slots if s != anchor["slot"]]

    outfits: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()

    for variant in range(max_outfits):
        bundle = [anchor]
        used = {anchor["item_id"]}
        failed = False

        for sl in slots:
            if sl == anchor["slot"]:
                continue
            candidates = [
                c
                for c in pool
                if c.get("slot") == sl and c.get("item_id") not in used
            ]
            if not candidates:
                failed = True
                break

            def pair_score(c: dict) -> float:
                return sum(
                    compatibility_score(
                        c["embedding"],
                        b["embedding"],
                        sl,
                        b["slot"],
                    )
                    for b in bundle
                )

            candidates.sort(key=pair_score, reverse=True)
            if variant >= len(candidates):
                failed = True
                break
            pick = candidates[variant]
            bundle.append(pick)
            used.add(pick["item_id"])

        if failed or len(bundle) < 2:
            continue

        key = frozenset(i["item_id"] for i in bundle)
        if key in seen:
            continue
        seen.add(key)

        outfits.append({
            "label": f"Suggested look {len(outfits) + 1}",
            "items": [clean_item(i) for i in bundle],
        })

    return outfits


def _occasion_rank_modes(
    occ_vec: np.ndarray,
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None,
) -> list[np.ndarray]:
    modes: list[np.ndarray] = [occ_vec.astype(np.float32)]
    if taste_modes:
        for m in taste_modes:
            m = np.array(m, dtype=np.float32)
            if np.linalg.norm(m - occ_vec) < 1e-3:
                continue
            modes.append(m)
            if len(modes) >= 5:
                break
    else:
        tv = taste_vector.astype(np.float32)
        if np.linalg.norm(tv - occ_vec) > 1e-3:
            modes.append(tv)
    return modes


def build_occasion_sections_unified(
    occasion_vectors: dict[str, list[float]],
    wardrobe: list[dict],
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None,
    trend_fp: dict[str, float] | None,
    anti_taste: np.ndarray | None,
    negative_prototype: np.ndarray | None,
    gap_slots: list[str],
    price_tier: tuple[float, float],
    exclude_ids: set[str],
    per_section: int = 6,
    style_attributes: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """FAISS + price + ranker path aligned with main recommendations."""
    if not occasion_vectors:
        return []

    wardrobe_emb = build_wardrobe_embedding(wardrobe)
    save_count = len(wardrobe)
    sections: list[dict[str, Any]] = []
    already = set(exclude_ids)

    for occ_name, occ_list in occasion_vectors.items():
        if not occ_list:
            continue
        occ_vec = np.array(occ_list, dtype=np.float32)
        try:
            query = blend_vectors(occ_vec, wardrobe_emb, save_count)
            candidates = generate_candidates(
                taste_vector=query,
                gap_slots=gap_slots,
                price_tier=price_tier,
                top_k=50,
                exclude_ids=already,
                trend_fingerprint=trend_fp,
            )
            modes = _occasion_rank_modes(occ_vec, taste_vector, taste_modes)
            ranked = rank_candidates(
                candidates,
                wardrobe,
                taste_vector,
                taste_modes=modes,
                trend_fingerprint=trend_fp,
                anti_taste_vector=anti_taste,
                negative_prototype=negative_prototype,
                style_attributes=style_attributes or {},
            )
            items_out: list[dict] = []
            for row in ranked:
                iid = row["item_id"]
                if iid in already:
                    continue
                already.add(iid)
                items_out.append({
                    "item": clean_item(row),
                    "taste_score": round(row["taste_score"], 2),
                    "unlock_count": row["unlock_count"],
                    "explanation": f"Matches {OCCASION_LABELS.get(occ_name, occ_name)} — ranked for your board.",
                })
                if len(items_out) >= per_section:
                    break
            if items_out:
                sections.append({
                    "occasion": occ_name,
                    "label": OCCASION_LABELS.get(occ_name, occ_name.title()),
                    "items": items_out,
                })
        except Exception:
            logger.exception("occasion section %s", occ_name)

    return sections


def run_wardrobe_orchestration(
    wardrobe: list[dict],
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None,
    occasion_vectors: dict[str, list[float]],
    trend_fp: dict[str, float] | None,
    anti_taste: np.ndarray | None,
    price_tier: tuple[float, float],
    aesthetic_label: str,
    skipped_item_ids: list[str],
    catalog: list[dict],
    intent_vector: np.ndarray | None = None,
    intent_confidence: float = 0.0,
) -> dict[str, Any]:
    """
    Full pipeline: main ranked list, gap pick, top picks, occasion sections,
    complete-the-look, outfit bundles, shopping brief.
    """
    stats = get_wardrobe_stats(wardrobe)
    coverage = compute_slot_coverage(wardrobe)
    gap_slots = get_gap_slots(coverage)
    if not gap_slots:
        gap_slots = [
            "tops", "bottoms", "outerwear", "shoes", "bags", "accessories"
        ]

    wardrobe_emb = build_wardrobe_embedding(wardrobe)
    save_count = len(wardrobe)
    query_vector = blend_vectors(taste_vector, wardrobe_emb, save_count)

    skip_set = set(skipped_item_ids)
    negative_proto = build_negative_prototype(catalog, skipped_item_ids)

    candidates = generate_candidates(
        taste_vector=query_vector,
        gap_slots=gap_slots,
        price_tier=price_tier,
        top_k=50,
        exclude_ids=skip_set,
        trend_fingerprint=trend_fp,
    )

    ranked = rank_candidates(
        candidates,
        wardrobe,
        taste_vector,
        taste_modes=taste_modes,
        trend_fingerprint=trend_fp,
        anti_taste_vector=anti_taste,
        negative_prototype=negative_proto,
        intent_vector=intent_vector,
        intent_confidence=intent_confidence,
    )

    top_trend = ""
    if trend_fp:
        top_trend = max(trend_fp, key=trend_fp.get)

    gap_rec = None
    top_rows: list[dict] = []
    if ranked:
        top_rows = list(ranked)

    complete_the_look = None
    if wardrobe:
        first_item = wardrobe[0]
        pairs = find_pairs(first_item, wardrobe)
        if pairs:
            complete_the_look = {
                "anchor_item": clean_item(first_item),
                "pairs": [clean_item(p) for p in pairs[:3]],
            }

    shopping_brief = build_shopping_brief(
        stats, price_tier, trend_fp, occasion_vectors
    )

    outfit_suggestions: list[dict[str, Any]] = []
    exclude_for_occasion: set[str] = set(skip_set)
    if top_rows:
        anchor = top_rows[0]
        exclude_for_occasion.add(anchor["item_id"])
        outfit_suggestions = build_outfit_suggestions(
            anchor, top_rows[:20], max_outfits=3
        )
        for p in top_rows[1:6]:
            exclude_for_occasion.add(p["item_id"])

    occasion_sections = build_occasion_sections_unified(
        occasion_vectors=occasion_vectors,
        wardrobe=wardrobe,
        taste_vector=taste_vector,
        taste_modes=taste_modes,
        trend_fp=trend_fp,
        anti_taste=anti_taste,
        negative_prototype=negative_proto,
        gap_slots=gap_slots,
        price_tier=price_tier,
        exclude_ids=exclude_for_occasion,
    )

    return {
        "ranked": top_rows,
        "stats": stats,
        "top_trend": top_trend,
        "gap_slots": gap_slots,
        "complete_the_look": complete_the_look,
        "shopping_brief": shopping_brief,
        "outfit_suggestions": outfit_suggestions,
        "occasion_sections": occasion_sections,
        "aesthetic_label": aesthetic_label,
    }
