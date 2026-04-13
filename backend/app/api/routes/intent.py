import logging
from fastapi import APIRouter

from app.db.models import IntentComputeRequest, IntentComputeResponse
from app.core.intent import compute_intent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intent", tags=["intent"])


@router.post("/compute", response_model=IntentComputeResponse)
async def compute_session_intent(req: IntentComputeRequest):
    """Compute intent vector from recently-viewed item embeddings."""
    result = compute_intent(req.viewed_embeddings)
    return IntentComputeResponse(
        intent_vector=result["intent_vector"],
        confidence=result["confidence"],
        num_views=result["num_views"],
        session_labels=result.get("session_labels", []),
    )
