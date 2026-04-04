"""
Candidate generation via FAISS vector search.

Handles index building, persistence, and gap-targeted retrieval.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from app.config import get_settings

_index: faiss.IndexFlatIP | None = None
_catalog_items: list[dict] = []


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
    global _index, _catalog_items

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

    return _index, _catalog_items


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
