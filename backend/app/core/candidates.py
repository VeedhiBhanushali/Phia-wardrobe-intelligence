"""
Candidate generation via FAISS vector search.

Handles index building, persistence, gap-targeted retrieval,
and metadata-filtered intent-mode search.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from app.config import get_settings

_index: faiss.IndexFlatIP | None = None
_catalog_items: list[dict] = []
_catalog_summary: dict | None = None


def build_index(items: list[dict]) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from catalog item embeddings."""
    embeddings = np.array(
        [item["embedding"] for item in items], dtype=np.float32
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, items: list[dict], base_path: str | None = None):
    settings = get_settings()
    base = Path(base_path) if base_path else Path(settings.faiss_index_path).parent
    base.mkdir(parents=True, exist_ok=True)

    index_path = base / "faiss_index.bin"
    catalog_path = base / "catalog_cache.json"

    faiss.write_index(index, str(index_path))

    serializable = []
    for item in items:
        entry = {k: v for k, v in item.items() if k != "embedding"}
        entry["embedding"] = [float(x) for x in item["embedding"]]
        serializable.append(entry)

    with open(catalog_path, "w") as f:
        json.dump(serializable, f)


def load_index(base_path: str | None = None) -> tuple[faiss.IndexFlatIP, list[dict]]:
    global _index, _catalog_items, _catalog_summary

    if _index is not None:
        return _index, _catalog_items

    settings = get_settings()
    base = Path(base_path) if base_path else Path(settings.faiss_index_path).parent
    index_path = base / "faiss_index.bin"
    catalog_path = base / "catalog_cache.json"

    if not index_path.exists() or not catalog_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run the catalog builder first."
        )

    _index = faiss.read_index(str(index_path))

    with open(catalog_path) as f:
        _catalog_items = json.load(f)

    _ensure_metadata(_catalog_items)
    _catalog_summary = _build_catalog_summary(_catalog_items)

    return _index, _catalog_items


def _ensure_metadata(items: list[dict]) -> None:
    """Backfill item_type/colors/occasions for catalogs built before enrichment."""
    needs_enrichment = any("item_type" not in item for item in items)
    if not needs_enrichment:
        return
    from app.data.catalog_builder import enrich_catalog_metadata
    enrich_catalog_metadata(items)


def _build_catalog_summary(items: list[dict]) -> dict:
    """Pre-compute aggregate stats for the system prompt."""
    type_counts: dict[str, int] = {}
    color_counts: dict[str, int] = {}
    brand_set: set[str] = set()
    prices: list[float] = []

    for item in items:
        t = item.get("item_type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

        for c in item.get("colors", []):
            if c != "unknown":
                color_counts[c] = color_counts.get(c, 0) + 1

        brand = item.get("brand", "")
        if brand:
            brand_set.add(brand)

        price = item.get("price", 0)
        if price > 0:
            prices.append(price)

    return {
        "total_items": len(items),
        "item_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "colors": dict(sorted(color_counts.items(), key=lambda x: -x[1])),
        "brands": sorted(brand_set),
        "price_range": [min(prices), max(prices)] if prices else [0, 0],
    }


def get_catalog_summary() -> dict:
    """Return cached catalog summary, loading index if needed."""
    global _catalog_summary
    if _catalog_summary is None:
        load_index()
    return _catalog_summary or {}


def search(
    query_vector: np.ndarray, top_k: int = 100
) -> list[tuple[dict, float]]:
    """Search the catalog index. Returns list of (item, score) tuples."""
    index, items = load_index()

    query = query_vector.astype(np.float32).reshape(1, -1)

    if query.shape[1] != index.d:
        raise ValueError(
            f"Query vector dimension ({query.shape[1]}) doesn't match "
            f"index dimension ({index.d})"
        )

    faiss.normalize_L2(query)

    scores, indices = index.search(query, min(top_k, len(items)))

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        results.append((items[idx], float(score)))

    return results


SLOT_PRIORITY = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]


def search_with_filters(
    query_vector: np.ndarray,
    top_k: int = 10,
    slots: list[str] | None = None,
    item_type: str | None = None,
    colors: list[str] | None = None,
    exclude_ids: set[str] | None = None,
    price_tier: tuple[float, float] | None = None,
) -> tuple[list[dict], str]:
    """
    Staged intent-mode search with automatic fallback.

    Stages (stops as soon as top_k results are found):
      1. Exact: item_type + colors match (hard filter)
      2. Relaxed color: item_type match + expanded color family
      3. Type only: item_type match, any color
      4. Semantic: pure FAISS similarity, no metadata filter

    Returns (results, stage_used) so the caller can tell the user
    what was matched vs. what was relaxed.
    """
    from app.data.catalog_builder import normalize_item_type, expand_color_family

    settings = get_settings()
    tolerance = settings.price_band_tolerance
    exclude = exclude_ids or set()

    canonical_type = normalize_item_type(item_type) if item_type else None
    exact_colors = [c.lower().strip() for c in colors] if colors else None
    expanded_colors = expand_color_family(exact_colors) if exact_colors else None

    pool_size = max(top_k * 15, 300)
    results = search(query_vector, top_k=pool_size)

    def _base_filter(item: dict, score: float) -> bool:
        if item.get("item_id") in exclude:
            return False
        if slots and item.get("slot") not in slots:
            return False
        if price_tier and not _in_price_band(item, price_tier, tolerance):
            return False
        return True

    def _collect(type_filter: str | None, color_set: list[str] | None) -> list[dict]:
        found: list[dict] = []
        for item, score in results:
            if not _base_filter(item, score):
                continue
            if type_filter and item.get("item_type", "other") != type_filter:
                continue
            if color_set:
                item_colors = set(item.get("colors", []))
                if not item_colors.intersection(color_set):
                    continue
            found.append({**item, "retrieval_score": score})
            if len(found) >= top_k:
                break
        return found

    # Stage 1: exact type + exact colors
    if canonical_type and exact_colors:
        exact = _collect(canonical_type, exact_colors)
        if exact:
            return exact, "exact"

    # Stage 2: exact type + expanded color family
    if canonical_type and expanded_colors:
        relaxed_color = _collect(canonical_type, expanded_colors)
        if relaxed_color:
            return relaxed_color, "relaxed_color"

    # Stage 3: type only, drop color constraint
    if canonical_type:
        type_only = _collect(canonical_type, None)
        if type_only:
            return type_only, "type_only"

    # Stage 4: semantic-only fallback (slot filter still applies)
    semantic = _collect(None, None)
    return semantic, "semantic"


def _in_price_band(
    item: dict, price_tier: tuple[float, float], tolerance: float
) -> bool:
    price = item.get("price", 0)
    low = price_tier[0] * (1 - tolerance)
    high = price_tier[1] * (1 + tolerance)
    return low <= price <= high


def generate_candidates(
    taste_vector: np.ndarray,
    gap_slots: list[str],
    price_tier: tuple[float, float],
    top_k: int = 100,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """
    Per-slot bucketed retrieval with round-robin interleaving.

    Retrieves broadly, buckets by gap slot, then interleaves
    in SLOT_PRIORITY order so underrepresented slots aren't
    drowned out by globally-dominant categories.
    """
    settings = get_settings()
    tolerance = settings.price_band_tolerance
    top_k_per_slot = max(top_k // max(len(gap_slots), 1), 15)
    exclude = exclude_ids or set()

    results = search(taste_vector, top_k=top_k_per_slot * len(gap_slots) * 3)

    ordered_slots = [s for s in SLOT_PRIORITY if s in gap_slots]
    for s in gap_slots:
        if s not in ordered_slots:
            ordered_slots.append(s)

    by_slot: dict[str, list[dict]] = {s: [] for s in ordered_slots}
    for item, score in results:
        if item.get("item_id") in exclude:
            continue
        slot = item.get("slot", "")
        if slot not in by_slot:
            continue
        if not _in_price_band(item, price_tier, tolerance):
            continue
        if len(by_slot[slot]) < top_k_per_slot:
            by_slot[slot].append({**item, "retrieval_score": score})

    candidates: list[dict] = []
    slot_queues = [by_slot[s] for s in ordered_slots if by_slot[s]]

    for i in range(top_k_per_slot):
        for queue in slot_queues:
            if i < len(queue):
                candidates.append(queue[i])

    return candidates[: top_k_per_slot * len(gap_slots)]
