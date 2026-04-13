"""
Wardrobe state management.

Handles slot coverage, wardrobe embedding, gap analysis,
and cold start blending.
"""

import numpy as np

OUTFIT_SLOTS = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

SLOT_PRIORITY = {
    "bottoms": 1,
    "tops": 2,
    "shoes": 3,
    "outerwear": 4,
    "bags": 5,
    "accessories": 6,
}

CATEGORY_TO_SLOT = {
    "shirt": "tops", "blouse": "tops", "top": "tops", "tee": "tops",
    "t-shirt": "tops", "sweater": "tops", "hoodie": "tops", "tank": "tops",
    "camisole": "tops", "vest": "tops", "polo": "tops", "henley": "tops",
    "bodysuit": "tops", "shell": "tops",
    "jeans": "bottoms", "pants": "bottoms", "trousers": "bottoms",
    "skirt": "bottoms", "shorts": "bottoms", "chinos": "bottoms",
    "joggers": "bottoms", "bermudas": "bottoms",
    "coat": "outerwear", "jacket": "outerwear", "blazer": "outerwear",
    "cardigan": "outerwear", "duster": "outerwear", "puffer": "outerwear",
    "sneakers": "shoes", "boots": "shoes", "sandals": "shoes",
    "loafers": "shoes", "heels": "shoes", "pumps": "shoes",
    "flats": "shoes", "clogs": "shoes", "mules": "shoes",
    "tote": "bags", "bag": "bags", "crossbody": "bags",
    "satchel": "bags", "clutch": "bags", "purse": "bags",
    "necklace": "accessories", "earrings": "accessories", "bracelet": "accessories",
    "ring": "accessories", "scarf": "accessories", "belt": "accessories",
    "watch": "accessories", "sunglasses": "accessories", "hat": "accessories",
    "cap": "accessories", "beanie": "accessories", "gloves": "accessories",
}


def map_category_to_slot(category: str) -> str | None:
    """Map a product category to an outfit slot."""
    cat = category.lower().strip()
    if cat in OUTFIT_SLOTS:
        return cat
    return CATEGORY_TO_SLOT.get(cat)


def compute_slot_coverage(wardrobe_items: list[dict]) -> dict[str, list[dict]]:
    """Group wardrobe items by outfit slot."""
    coverage = {slot: [] for slot in OUTFIT_SLOTS}
    for item in wardrobe_items:
        slot = item.get("slot") or map_category_to_slot(item.get("category", ""))
        if slot and slot in coverage:
            coverage[slot].append(item)
    return coverage


def get_gap_slots(coverage: dict[str, list[dict]]) -> list[str]:
    """Return slots with 0 or 1 items, ordered by outfit importance."""
    gaps = [s for s in OUTFIT_SLOTS if len(coverage.get(s, [])) <= 1]
    gaps.sort(key=lambda s: SLOT_PRIORITY.get(s, 99))
    return gaps


def get_strongest_slot(coverage: dict[str, list[dict]]) -> str | None:
    """Return the slot with the most items, or None if all empty."""
    best = max(coverage, key=lambda s: len(coverage[s]))
    return best if len(coverage[best]) > 0 else None


def build_wardrobe_embedding(wardrobe_items: list[dict]) -> np.ndarray | None:
    """
    Recency-weighted mean of wardrobe item embeddings.

    More recent saves (later in list) get higher weight.
    """
    embeddings = []
    for item in wardrobe_items:
        emb = item.get("embedding")
        if emb is not None:
            embeddings.append(np.array(emb, dtype=np.float32))

    if not embeddings:
        return None

    embeddings = np.array(embeddings)
    n = len(embeddings)
    weights = np.array([1.0 / (n - i) for i in range(n)])
    weights = weights / weights.sum()

    wardrobe_vec = (embeddings * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(wardrobe_vec)
    if norm > 0:
        wardrobe_vec = wardrobe_vec / norm
    return wardrobe_vec.astype(np.float32)


def blend_vectors(
    taste_vector: np.ndarray,
    wardrobe_embedding: np.ndarray | None,
    save_count: int,
) -> np.ndarray:
    """
    Blend taste vector and wardrobe embedding based on save count.

    Conservative schedule: taste vector retains majority weight at low
    save counts to prevent cross-modal averaging from diluting signal.

    | Saves | Taste | Wardrobe |
    |-------|-------|----------|
    | 0     | 100%  | 0%       |
    | 1-4   | 85%   | 15%      |
    | 5-14  | 65%   | 35%      |
    | 15+   | 45%   | 55%      |
    """
    if wardrobe_embedding is None or save_count == 0:
        return taste_vector

    if save_count <= 4:
        taste_weight, wardrobe_weight = 0.85, 0.15
    elif save_count <= 14:
        taste_weight, wardrobe_weight = 0.65, 0.35
    else:
        taste_weight, wardrobe_weight = 0.45, 0.55

    blended = taste_weight * taste_vector + wardrobe_weight * wardrobe_embedding
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.astype(np.float32)


def get_wardrobe_stats(wardrobe_items: list[dict]) -> dict:
    """Compute summary stats for a wardrobe."""
    coverage = compute_slot_coverage(wardrobe_items)
    gap_slots = get_gap_slots(coverage)
    strongest = get_strongest_slot(coverage) if wardrobe_items else None

    return {
        "total_items": len(wardrobe_items),
        "slot_counts": {s: len(items) for s, items in coverage.items()},
        "gap_slots": gap_slots,
        "strongest_slot": strongest,
    }
