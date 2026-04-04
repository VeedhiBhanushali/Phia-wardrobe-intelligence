from fastapi import APIRouter
from app.db.models import EventLogRequest
from app.core.event_store import append_event, get_events_memory

router = APIRouter(prefix="/api/events", tags=["events"])


@router.post("/log")
async def log_event(req: EventLogRequest):
    """Log recommendation / dismiss events (JSONL + optional Supabase)."""
    event = {
        "user_id": req.user_id,
        "event_type": req.event_type,
        "module": req.module,
        "item_id": req.item_id,
        "score": req.score,
        "unlock_count": req.unlock_count,
        "taste_score": req.taste_score,
        "model_version": "v0.2",
    }
    append_event(event)
    _, total = get_events_memory()
    return {"status": "logged", "total_events": total}


@router.get("/log")
async def get_events(user_id: str = "", limit: int = 100):
    """Retrieve logged events (memory + tail of JSONL on first access)."""
    events, total = get_events_memory(user_id=user_id, limit=limit)
    return {"events": events, "total": total}
