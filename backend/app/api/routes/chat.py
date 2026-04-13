import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.db.models import ChatRequest
from app.core.candidates import load_index
from app.core.stylist_agent import run_stylist_chat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _get_wardrobe_items(item_ids: list[str]) -> list[dict]:
    """Look up full item data from catalog by IDs."""
    try:
        _, catalog = load_index()
    except FileNotFoundError:
        return []
    id_set = set(item_ids)
    return [item for item in catalog if item["item_id"] in id_set]


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Streaming SSE endpoint for the stylist chat.
    Each event is a JSON object with type: text|item_card|outfit_bundle|done.
    """
    wardrobe = _get_wardrobe_items(req.wardrobe_item_ids)

    taste_profile = {
        "taste_vector": req.taste_vector,
        "taste_modes": req.taste_modes,
        "occasion_vectors": req.occasion_vectors,
        "trend_fingerprint": req.trend_fingerprint,
        "anti_taste_vector": req.anti_taste_vector,
        "style_attributes": req.style_attributes,
        "price_tier": req.price_tier,
        "aesthetic_attributes": req.aesthetic_attributes,
    }

    async def event_stream():
        async for event in run_stylist_chat(req.messages, wardrobe, taste_profile):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
