"""
Taste extraction pipeline.

Images -> FashionCLIP encode -> occasion classification ->
per-occasion taste vectors -> trend fingerprinting -> anti-taste.

Taste is not one thing. Someone's work style is different from
their casual, their going-out, their weekend.  Each occasion
carries their personal interpretation of their own aesthetic.
"""

import numpy as np
from sklearn.cluster import KMeans
from app.core.clip_encoder import get_encoder
from app.core.trends import (
    compute_trend_fingerprint,
    compute_anti_taste_vector,
)

OCCASION_CONTEXTS: dict[str, str] = {
    "work": (
        "professional office workwear business formal blazer tailored "
        "structured polished corporate elegant meeting pencil skirt "
        "button-up shirt trousers loafers"
    ),
    "casual": (
        "casual everyday relaxed comfortable streetwear laid-back denim "
        "sneakers tee hoodie baseball cap tote bag jeans simple"
    ),
    "evening": (
        "evening going out date night dressy cocktail party glamorous "
        "heels statement dress elegant satin silk jewelry clutch"
    ),
    "weekend": (
        "weekend brunch errands coffee relaxed chic comfortable "
        "effortless cool linen sneakers sundress cardigan flats"
    ),
    "special": (
        "special occasion wedding guest formal event gala standout "
        "statement piece dramatic bold print ruffles voluminous"
    ),
}

OCCASION_LABELS = {
    "work": "Your work style",
    "casual": "Your everyday",
    "evening": "Your going-out",
    "weekend": "Your weekend",
    "special": "Your standout looks",
}

STYLE_PROMPTS = {
    "silhouette": [
        ("Fitted", "fitted slim tailored clothing"),
        ("Relaxed", "relaxed loose comfortable casual clothing"),
        ("Oversized", "oversized baggy streetwear clothing"),
        ("Structured", "structured sharp tailored power dressing"),
    ],
    "color_story": [
        ("Neutral Tones", "neutral tones beige white grey cream clothing"),
        ("Bold Colors", "bold bright colorful vibrant clothing"),
        ("Monochrome", "monochrome black and white clothing"),
        ("Warm Earth Tones", "warm earth tones brown rust terracotta clothing"),
        ("Cool Tones", "cool tones blue grey silver clothing"),
    ],
    "formality": [
        ("Casual", "casual streetwear everyday relaxed clothing"),
        ("Smart Casual", "smart casual polished everyday clothing"),
        ("Business", "business professional office formal clothing"),
        ("Formal", "formal elegant evening dressy clothing"),
    ],
    "occasion": [
        ("Everyday", "everyday casual daily wear clothing"),
        ("Office", "office work professional clothing"),
        ("Evening", "evening going out party dressy clothing"),
        ("Athletic", "athletic sport activewear clothing"),
        ("Travel", "travel comfortable versatile clothing"),
    ],
}

_text_embeddings_cache: dict[str, np.ndarray] = {}
_occasion_embeddings_cache: np.ndarray | None = None


def _get_prompt_embeddings(attribute: str) -> tuple[list[str], np.ndarray]:
    if attribute not in _text_embeddings_cache:
        encoder = get_encoder()
        prompts = STYLE_PROMPTS[attribute]
        texts = [p[1] for p in prompts]
        _text_embeddings_cache[attribute] = encoder.encode_texts(texts)
    labels = [p[0] for p in STYLE_PROMPTS[attribute]]
    return labels, _text_embeddings_cache[attribute]


def _get_occasion_embeddings() -> tuple[list[str], np.ndarray]:
    global _occasion_embeddings_cache
    if _occasion_embeddings_cache is None:
        encoder = get_encoder()
        names = list(OCCASION_CONTEXTS.keys())
        prompts = list(OCCASION_CONTEXTS.values())
        _occasion_embeddings_cache = encoder.encode_texts(prompts)
    return list(OCCASION_CONTEXTS.keys()), _occasion_embeddings_cache


def classify_by_occasion(
    embeddings: np.ndarray,
) -> dict[str, list[int]]:
    """
    Assign each image embedding to the occasion context it most
    closely matches.  Returns {occasion: [image_indices]}.
    """
    names, occ_embs = _get_occasion_embeddings()
    sims = embeddings @ occ_embs.T  # (n_images, n_occasions)

    buckets: dict[str, list[int]] = {n: [] for n in names}
    for i in range(len(embeddings)):
        best = int(sims[i].argmax())
        buckets[names[best]].append(i)

    return buckets


def build_occasion_vectors(
    embeddings: np.ndarray,
    occasion_buckets: dict[str, list[int]],
    min_images: int = 2,
) -> dict[str, np.ndarray]:
    """
    Build a taste vector per occasion from the images that belong
    to that occasion.  Only occasions with >= min_images get a vector.
    """
    vectors: dict[str, np.ndarray] = {}
    for occasion, indices in occasion_buckets.items():
        if len(indices) < min_images:
            continue
        cluster = embeddings[indices]
        centroid = cluster.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        vectors[occasion] = centroid.astype(np.float32)
    return vectors


def build_taste_vector(
    embeddings: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(embeddings))
    weights = weights / weights.sum()
    taste_vector = (embeddings * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(taste_vector)
    if norm > 0:
        taste_vector = taste_vector / norm
    return taste_vector.astype(np.float32)


def extract_taste_modes(
    embeddings: np.ndarray,
    min_modes: int = 2,
    max_modes: int = 4,
) -> list[np.ndarray]:
    n = len(embeddings)
    if n < 4:
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return [centroid.astype(np.float32)]

    k = min(max_modes, max(min_modes, n // 5))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    modes: list[np.ndarray] = []
    for i in range(k):
        cluster = embeddings[labels == i]
        if len(cluster) < 2:
            continue
        centroid = cluster.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        modes.append(centroid.astype(np.float32))

    if not modes:
        fallback = embeddings.mean(axis=0)
        norm = np.linalg.norm(fallback)
        if norm > 0:
            fallback = fallback / norm
        modes = [fallback.astype(np.float32)]

    return modes


def extract_attributes(taste_vector: np.ndarray) -> dict:
    attributes = {}
    for attr in STYLE_PROMPTS:
        labels, text_embeddings = _get_prompt_embeddings(attr)
        sims = taste_vector @ text_embeddings.T
        best_idx = int(sims.argmax())
        attributes[attr] = {
            "label": labels[best_idx],
            "confidence": float(sims[best_idx]),
            "scores": {
                label: float(score) for label, score in zip(labels, sims)
            },
        }
    return attributes


def infer_price_tier(
    embeddings: np.ndarray, items: list[dict] | None = None
) -> tuple[float, float]:
    if items and any(item.get("price", 0) > 0 for item in items):
        prices = [item["price"] for item in items if item.get("price", 0) > 0]
        median = float(np.median(prices))
        std = float(np.std(prices)) if len(prices) > 1 else median * 0.3
        return (max(20, median - std), median + std)
    return (40.0, 200.0)


def extract_taste_profile(
    image_sources: list,
    source_type: str = "upload",
    existing_saves: list[dict] | None = None,
) -> dict:
    """
    Full taste extraction pipeline.

    Returns per-occasion taste vectors alongside the global taste
    vector, trend fingerprint, and anti-taste vector.
    """
    encoder = get_encoder()

    embeddings = encoder.encode_images(image_sources)

    if source_type == "pinterest":
        weights = np.full(len(embeddings), 0.6)
    else:
        weights = np.ones(len(embeddings))

    if existing_saves:
        save_embeddings = np.array(
            [s["embedding"] for s in existing_saves if "embedding" in s],
            dtype=np.float32,
        )
        if len(save_embeddings) > 0:
            save_weights = np.ones(len(save_embeddings))
            embeddings = np.concatenate([embeddings, save_embeddings], axis=0)
            weights = np.concatenate([weights, save_weights])

    taste_vector = build_taste_vector(embeddings, weights)
    taste_modes = extract_taste_modes(embeddings)

    occasion_buckets = classify_by_occasion(embeddings)
    occasion_vectors = build_occasion_vectors(embeddings, occasion_buckets)

    trend_fingerprint = compute_trend_fingerprint(taste_vector)
    anti_taste_vector = compute_anti_taste_vector(trend_fingerprint)

    attributes = extract_attributes(taste_vector)
    price_tier = infer_price_tier(embeddings, existing_saves)

    return {
        "taste_vector": taste_vector.tolist(),
        "taste_modes": [m.tolist() for m in taste_modes],
        "occasion_vectors": {k: v.tolist() for k, v in occasion_vectors.items()},
        "trend_fingerprint": trend_fingerprint,
        "anti_taste_vector": anti_taste_vector.tolist(),
        "aesthetic_attributes": attributes,
        "price_tier": list(price_tier),
    }


def update_taste_profile(
    existing_vector: np.ndarray,
    new_item_embedding: np.ndarray,
    save_count: int,
) -> np.ndarray:
    existing_weight = save_count / (save_count + 1)
    new_weight = 1.0 / (save_count + 1)

    updated = existing_weight * existing_vector + new_weight * new_item_embedding
    norm = np.linalg.norm(updated)
    if norm > 0:
        updated = updated / norm
    return updated.astype(np.float32)
