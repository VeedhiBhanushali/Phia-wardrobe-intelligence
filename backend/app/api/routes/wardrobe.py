from fastapi import APIRouter, HTTPException
from uuid import uuid4

from app.db.models import SaveItemRequest
from app.core.candidates import load_index

router = APIRouter(prefix="/api/wardrobe", tags=["wardrobe"])

_wardrobe_store: dict[str, list[dict]] = {}


@router.get("/{user_id}")
async def get_wardrobe(user_id: str):
    """Get a user's wardrobe (saved items)."""
    saves = _wardrobe_store.get(user_id, [])
    return {
        "user_id": user_id,
        "items": [{k: v for k, v in s.items() if k != "embedding"} for s in saves],
        "total": len(saves),
    }


@router.post("/save")
async def save_item(req: SaveItemRequest):
    """Save an item to the user's wardrobe."""
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    item = next((i for i in catalog if i["item_id"] == req.item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found in catalog")

    if req.user_id not in _wardrobe_store:
        _wardrobe_store[req.user_id] = []

    existing_ids = {s["item_id"] for s in _wardrobe_store[req.user_id]}
    if req.item_id in existing_ids:
        raise HTTPException(status_code=409, detail="Item already saved")

    save_entry = {**item, "save_id": str(uuid4())}
    _wardrobe_store[req.user_id].append(save_entry)

    return {
        "save_id": save_entry["save_id"],
        "item_id": req.item_id,
        "user_id": req.user_id,
    }


@router.delete("/save/{user_id}/{item_id}")
async def remove_save(user_id: str, item_id: str):
    """Remove an item from the user's wardrobe."""
    if user_id not in _wardrobe_store:
        raise HTTPException(status_code=404, detail="User not found")

    before = len(_wardrobe_store[user_id])
    _wardrobe_store[user_id] = [
        s for s in _wardrobe_store[user_id] if s["item_id"] != item_id
    ]

    if len(_wardrobe_store[user_id]) == before:
        raise HTTPException(status_code=404, detail="Save not found")

    return {"status": "removed", "item_id": item_id}
