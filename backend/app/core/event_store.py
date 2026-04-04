"""
Append-only event log: JSONL on disk + optional Supabase insert.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

EVENT_LOG_PATH = Path("data/event_log.jsonl")
_loaded_from_disk = False
_memory_events: list[dict] = []


def _ensure_disk_loaded() -> None:
    global _loaded_from_disk, _memory_events
    if _loaded_from_disk:
        return
    if not EVENT_LOG_PATH.exists():
        _loaded_from_disk = True
        return
    try:
        lines = EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()[-3000:]
        for line in lines:
            try:
                _memory_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        logger.warning("Could not read event log: %s", e)
    _loaded_from_disk = True


def append_event(event: dict) -> None:
    """Persist one event to JSONL, memory buffer, and optionally Supabase."""
    global _memory_events
    _ensure_disk_loaded()

    row = {
        **event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": event.get("model_version", "v0.2"),
    }
    _memory_events.append(row)
    if len(_memory_events) > 5000:
        _memory_events = _memory_events[-4000:]

    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError as e:
        logger.warning("Could not append event log: %s", e)

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        return
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_key)
        client.table("wardrobe_events").insert({
            "user_id": row["user_id"],
            "event_type": row["event_type"],
            "module": row["module"],
            "item_id": row["item_id"],
            "score": row.get("score"),
            "unlock_count": row.get("unlock_count"),
            "taste_score": row.get("taste_score"),
            "model_version": row["model_version"],
            "created_at": row["timestamp"],
        }).execute()
    except Exception as e:
        logger.warning("Supabase wardrobe_events insert failed: %s", e)


def get_events_memory(user_id: str = "", limit: int = 100) -> tuple[list[dict], int]:
    _ensure_disk_loaded()
    events = _memory_events
    if user_id:
        events = [e for e in events if e.get("user_id") == user_id]
    return events[-limit:], len(_memory_events)
