from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np

from app.core.candidates import load_index, search

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("/search")
async def search_catalog(
    q: str = Query("", description="Search query"),
    slot: str = Query("", description="Filter by outfit slot"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Search and browse the product catalog."""
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    results = catalog

    if slot:
        results = [item for item in results if item["slot"] == slot]

    if q:
        q_lower = q.lower()
        results = [
            item for item in results
            if q_lower in item.get("title", "").lower()
            or q_lower in item.get("brand", "").lower()
            or q_lower in item.get("category", "").lower()
        ]

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = results[start:end]

    return {
        "items": [{k: v for k, v in item.items() if k != "embedding"} for item in page_items],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


class TasteSearchRequest(BaseModel):
    taste_vector: list[float]
    slot: str = ""
    top_k: int = Field(default=20, ge=1, le=100)
    exclude_ids: list[str] = Field(default_factory=list)


@router.post("/taste-search")
async def taste_search(req: TasteSearchRequest):
    """
    Return catalog items ranked by CLIP cosine similarity to the taste vector.

    This is pure taste-based retrieval via FAISS — no wardrobe utility scoring,
    just "what matches this user's aesthetic the most."
    """
    taste = np.array(req.taste_vector, dtype=np.float32)
    if taste.size == 0:
        raise HTTPException(status_code=400, detail="taste_vector is required")

    try:
        results = search(taste, top_k=req.top_k * 3)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    exclude = set(req.exclude_ids)
    items = []
    for item, score in results:
        if item["item_id"] in exclude:
            continue
        if req.slot and item.get("slot") != req.slot:
            continue
        items.append({
            **{k: v for k, v in item.items() if k != "embedding"},
            "taste_score": round(float(score), 3),
        })
        if len(items) >= req.top_k:
            break

    return {"items": items, "total": len(items)}


@router.get("/item/{item_id}")
async def get_catalog_item(
    item_id: str,
    include_embedding: bool = Query(False, description="Include CLIP embedding vector"),
):
    """Get a single catalog item by ID."""
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    item = next((i for i in catalog if i["item_id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if include_embedding:
        return dict(item)

    return {k: v for k, v in item.items() if k != "embedding"}
