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
    top_coherent_trends,
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
    display_trends = top_coherent_trends(trend_fingerprint)

    attributes = extract_attributes(taste_vector)
    price_tier = infer_price_tier(embeddings, existing_saves)
    style_attributes = compute_style_attribute_profile(embeddings, encoder)
    style_summary = style_attribute_summary(style_attributes)

    return {
        "taste_vector": taste_vector.tolist(),
        "taste_modes": [m.tolist() for m in taste_modes],
        "occasion_vectors": {k: v.tolist() for k, v in occasion_vectors.items()},
        "trend_fingerprint": trend_fingerprint,
        "display_trends": display_trends,
        "anti_taste_vector": anti_taste_vector.tolist(),
        "aesthetic_attributes": attributes,
        "price_tier": list(price_tier),
        "style_attributes": style_attributes,
        "style_summary": style_summary,
    }


# ---------------------------------------------------------------------------
# Style Attribute Profile — 25-axis preference system
# ---------------------------------------------------------------------------
# Each axis is a (positive_pole, negative_pole, display_label, scale) tuple.
# Preference score: user images projected onto (pos_pole - neg_pole) direction.
# Result in [-1, +1]: +1 = strongly prefers positive pole, -1 = negative pole.
# scale = normalisation factor (typical inter-pole dot-product difference).
# ---------------------------------------------------------------------------

STYLE_AXES: dict[str, tuple[str, str, str, float]] = {
    # ---- Pattern ----
    "pattern_solid": (
        "solid color plain minimal clean no pattern clothing",
        "colorful patterned printed bold pattern clothing",
        "Solid / Plain",
        0.15,
    ),
    "pattern_floral": (
        "floral print botanical flower pattern clothing",
        "solid plain no print clothing",
        "Floral prints",
        0.12,
    ),
    "pattern_geometric": (
        "geometric graphic abstract print clothing",
        "solid plain no print clothing",
        "Geometric prints",
        0.12,
    ),
    "pattern_stripe": (
        "striped plaid check tartan pattern clothing",
        "solid plain no print clothing",
        "Stripes & plaid",
        0.12,
    ),
    "pattern_animal": (
        "animal print leopard zebra snake pattern clothing",
        "solid plain no print clothing",
        "Animal print",
        0.12,
    ),
    "pattern_polka": (
        "polka dot spotted dotted pattern clothing",
        "solid plain no print clothing",
        "Polka dots",
        0.10,
    ),
    # ---- Material / texture ----
    "material_silk": (
        "silk satin smooth luxurious shiny fabric clothing",
        "cotton linen matte casual fabric clothing",
        "Silk & satin",
        0.13,
    ),
    "material_knit": (
        "knit knitwear wool sweater cable cozy texture clothing",
        "smooth woven crisp fabric clothing",
        "Knits & texture",
        0.13,
    ),
    "material_leather": (
        "leather suede structured bold statement material",
        "soft fabric casual clothing",
        "Leather & suede",
        0.12,
    ),
    "material_denim": (
        "denim jeans casual indigo cotton clothing",
        "formal dressy non-denim clothing",
        "Denim",
        0.13,
    ),
    "material_linen": (
        "linen natural breathable relaxed summer fabric clothing",
        "synthetic formal structured fabric clothing",
        "Linen & natural",
        0.12,
    ),
    # ---- Branding ----
    "branding_minimal": (
        "minimal no logo clean unbranded simple clothing",
        "logo monogram branded designer label clothing",
        "No visible branding",
        0.13,
    ),
    "branding_logo": (
        "designer logo monogram branded streetwear hypebeast clothing",
        "minimal unbranded clean clothing",
        "Visible logos",
        0.13,
    ),
    # ---- Color palette ----
    "palette_neutral": (
        "neutral beige cream ivory white grey black minimal color clothing",
        "bold bright colorful vibrant clothing",
        "Neutral palette",
        0.14,
    ),
    "palette_dark": (
        "dark black navy charcoal moody dark tones clothing",
        "light pastel bright white clothing",
        "Dark tones",
        0.13,
    ),
    "palette_warm": (
        "warm tones orange rust terracotta red brown earth tones clothing",
        "cool blue grey tones clothing",
        "Warm tones",
        0.12,
    ),
    "palette_colorful": (
        "colorful vibrant bold bright multicolor rainbow clothing",
        "neutral monochrome muted clothing",
        "Bold colors",
        0.13,
    ),
    "palette_pastel": (
        "pastel soft light dusty rose lavender sage mint clothing",
        "dark saturated bold color clothing",
        "Pastels",
        0.12,
    ),
    # ---- Silhouette / fit ----
    "fit_oversized": (
        "oversized baggy relaxed loose boxy streetwear clothing",
        "fitted slim tailored structured clothing",
        "Oversized / relaxed",
        0.14,
    ),
    "fit_tailored": (
        "fitted tailored structured sharp professional clothing",
        "relaxed baggy casual loose clothing",
        "Tailored / fitted",
        0.14,
    ),
    "fit_flowy": (
        "flowy draped feminine flowing maxi midi soft clothing",
        "structured stiff rigid clothing",
        "Flowy & draped",
        0.13,
    ),
    # ---- Aesthetic era / vibe ----
    "vibe_minimal": (
        "minimal contemporary clean simple quiet luxury clothing",
        "maximalist embellished ornate busy clothing",
        "Minimalist",
        0.14,
    ),
    "vibe_vintage": (
        "vintage retro distressed worn thrifted nostalgic clothing",
        "modern fresh contemporary clean clothing",
        "Vintage & retro",
        0.13,
    ),
    "vibe_preppy": (
        "preppy academic collegiate clean-cut classic clothing",
        "street casual edgy clothing",
        "Preppy / classic",
        0.12,
    ),
    "vibe_romantic": (
        "romantic feminine soft delicate lace ruffle clothing",
        "androgynous structured minimal clothing",
        "Romantic",
        0.12,
    ),
    "vibe_sporty": (
        "sporty athletic activewear technical performance clothing",
        "dressy formal non-athletic clothing",
        "Sporty",
        0.13,
    ),
}

# Human-readable category groupings for UI display
STYLE_AXES_GROUPS: dict[str, list[str]] = {
    "Pattern": ["pattern_solid", "pattern_floral", "pattern_geometric",
                "pattern_stripe", "pattern_animal", "pattern_polka"],
    "Material": ["material_silk", "material_knit", "material_leather",
                 "material_denim", "material_linen"],
    "Branding": ["branding_minimal", "branding_logo"],
    "Color": ["palette_neutral", "palette_dark", "palette_warm",
              "palette_colorful", "palette_pastel"],
    "Fit": ["fit_oversized", "fit_tailored", "fit_flowy"],
    "Vibe": ["vibe_minimal", "vibe_vintage", "vibe_preppy",
             "vibe_romantic", "vibe_sporty"],
}

_style_axes_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _get_style_axis_probes(
    encoder,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lazy-load and cache all axis probe embeddings (one-time cost at startup)."""
    global _style_axes_cache
    if _style_axes_cache is not None:
        return _style_axes_cache

    pos_texts = [v[0] for v in STYLE_AXES.values()]
    neg_texts = [v[1] for v in STYLE_AXES.values()]
    all_texts = pos_texts + neg_texts
    all_embs = encoder.encode_texts(all_texts)

    n = len(STYLE_AXES)
    pos_embs = all_embs[:n]
    neg_embs = all_embs[n:]

    # Normalise
    for i in range(n):
        pos_embs[i] = pos_embs[i] / (np.linalg.norm(pos_embs[i]) + 1e-9)
        neg_embs[i] = neg_embs[i] / (np.linalg.norm(neg_embs[i]) + 1e-9)

    _style_axes_cache = {
        key: (pos_embs[i], neg_embs[i])
        for i, key in enumerate(STYLE_AXES.keys())
    }
    return _style_axes_cache


def compute_style_attribute_profile(
    embeddings: np.ndarray,
    encoder,
) -> dict[str, float]:
    """Compute fine-grained style preferences from a set of image embeddings.

    For each of the 25 semantic axes, measure how strongly the user's images
    lean toward each pole.  Returns a dict of axis_key -> float in [-1, +1].

      +1.0 = strongly prefers the positive pole (e.g. pattern_floral: +1 = loves florals)
      -1.0 = strongly avoids the positive pole (e.g. pattern_floral: -1 = avoids florals)
       0.0 = no detectable preference

    Only axes where the signal exceeds 0.3 are considered meaningful during ranking.
    """
    probes = _get_style_axis_probes(encoder)

    # Normalise input embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms  # (n_images, 512)

    result: dict[str, float] = {}
    for key, (pos_probe, neg_probe) in probes.items():
        scale = STYLE_AXES[key][3]
        pos_sims = normed @ pos_probe   # (n_images,)
        neg_sims = normed @ neg_probe
        mean_diff = float(np.mean(pos_sims - neg_sims))
        result[key] = float(np.clip(mean_diff / scale, -1.0, 1.0))

    return result


def style_attribute_summary(
    profile: dict[str, float],
    top_n: int = 5,
    threshold: float = 0.35,
) -> list[dict]:
    """Return the top N most distinctive preferences for UI display.

    Only includes axes where |score| > threshold (clear preference detected).
    Returns list of {key, label, score, direction} sorted by |score| desc.
    """
    entries = []
    for key, score in profile.items():
        if abs(score) < threshold:
            continue
        label = STYLE_AXES[key][2]
        entries.append({
            "key": key,
            "label": label,
            "score": round(score, 3),
            "direction": "prefers" if score > 0 else "avoids",
        })
    entries.sort(key=lambda x: abs(x["score"]), reverse=True)
    return entries[:top_n]


def update_style_attributes(
    current_attributes: dict[str, float],
    item_embedding: np.ndarray,
    encoder,
    save_count: int,
    direction: float = 1.0,
) -> dict[str, float]:
    """Update style_attributes after a save (+1) or dismiss (-1).

    Uses exponential moving average so early saves shape the profile
    strongly and later saves cause smaller incremental shifts.

    direction = +1.0 for saves (pull attributes toward item)
    direction = -1.0 for dismissals (push attributes away from item)
    """
    probes = _get_style_axis_probes(encoder)

    # Normalise item embedding
    emb = np.array(item_embedding, dtype=np.float32)
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n

    # Compute item's score on each axis
    item_scores: dict[str, float] = {}
    for key, (pos_probe, neg_probe) in probes.items():
        scale = STYLE_AXES[key][3]
        raw = float(np.dot(emb, pos_probe) - np.dot(emb, neg_probe))
        item_scores[key] = float(np.clip(raw / scale, -1.0, 1.0))

    # EMA blend: weight decays as save_count grows so the profile stabilises
    # Faster learning early (save 1-3), slower correction later (save 10+)
    learn_rate = max(0.08, 1.0 / (save_count + 1))

    updated: dict[str, float] = {}
    for key in STYLE_AXES:
        current = current_attributes.get(key, 0.0)
        signal = direction * item_scores.get(key, 0.0)
        # Only shift in the direction of the signal if it's clear (>0.3 item score)
        if abs(item_scores.get(key, 0.0)) < 0.3:
            updated[key] = current  # Item is ambiguous on this axis — no update
        else:
            blended = current + learn_rate * (signal - current)
            updated[key] = float(np.clip(blended, -1.0, 1.0))

    return updated


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
