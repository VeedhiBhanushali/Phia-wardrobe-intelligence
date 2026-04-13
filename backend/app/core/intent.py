"""
Session intent inference.

Computes an intent vector from recently-viewed item embeddings
and a session confidence score (mean pairwise cosine similarity).
High coherence = user is browsing with clear intent.
"""

import numpy as np
from app.core.trends import get_trend_embeddings


def compute_intent(
    viewed_embeddings: list[list[float]],
    min_views: int = 3,
    min_confidence: float = 0.3,
) -> dict:
    """
    Compute intent vector and session confidence from viewed items.

    Returns:
        dict with keys:
            - intent_vector: L2-normalized centroid (list[float]) or None
            - confidence: mean pairwise cosine similarity (float)
            - num_views: number of embeddings used
    """
    if len(viewed_embeddings) < min_views:
        return {
            "intent_vector": None,
            "confidence": 0.0,
            "num_views": len(viewed_embeddings),
        }

    embs = np.array(viewed_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embs_normed = embs / norms

    # Session confidence: mean pairwise cosine similarity
    n = len(embs_normed)
    if n < 2:
        confidence = 0.0
    else:
        sim_matrix = embs_normed @ embs_normed.T
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sim_matrix[mask]
        confidence = float(np.mean(pairwise_sims))

    if confidence < min_confidence:
        return {
            "intent_vector": None,
            "confidence": float(confidence),
            "num_views": n,
        }

    # Intent vector: L2-normalized centroid
    centroid = embs_normed.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    intent_vec = centroid.astype(np.float32)
    labels = _intent_labels(intent_vec)

    return {
        "intent_vector": intent_vec.tolist(),
        "confidence": float(confidence),
        "num_views": n,
        "session_labels": labels,
    }


def _intent_labels(intent_vector: np.ndarray, top_k: int = 2) -> list[str]:
    """Derive human-readable labels from the intent vector by matching
    against the trend lexicon.  Returns the top-k trend names whose
    similarity exceeds the mean by at least 0.5 std-devs."""
    names, embeddings = get_trend_embeddings()
    iv = intent_vector / max(float(np.linalg.norm(intent_vector)), 1e-8)
    sims = (embeddings @ iv).astype(np.float64)
    mean_s = float(sims.mean())
    std_s = float(sims.std())
    threshold = mean_s + 0.5 * std_s

    pairs = sorted(zip(names, sims.tolist()), key=lambda kv: kv[1], reverse=True)
    labels = [name for name, s in pairs if s >= threshold][:top_k]
    if not labels:
        labels = [pairs[0][0]]
    return labels


def blend_with_intent(
    taste_vector: np.ndarray,
    intent_vector: np.ndarray | None,
    intent_confidence: float,
    max_intent_weight: float = 0.55,
) -> np.ndarray:
    """
    Blend taste vector with intent vector based on confidence.

    Confidence linearly scales intent weight, capped at max_intent_weight.
    Taste always has at least (1 - max_intent_weight) influence.
    """
    if intent_vector is None or intent_confidence <= 0.3:
        return taste_vector

    intent_weight = min(intent_confidence, max_intent_weight)
    taste_weight = 1.0 - intent_weight

    tv = np.array(taste_vector, dtype=np.float32)
    iv = np.array(intent_vector, dtype=np.float32)

    blended = taste_weight * tv + intent_weight * iv
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.astype(np.float32)
