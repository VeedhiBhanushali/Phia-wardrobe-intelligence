"""
Trend-aware scoring layer.

Maintains a curated lexicon of current TikTok / fashion aesthetics,
computes per-user trend fingerprints, and derives anti-taste vectors
for negative gating.
"""

import numpy as np
from app.core.clip_encoder import get_encoder

TREND_LEXICON: dict[str, str] = {
    "Quiet Luxury": (
        "quiet luxury old money understated elegance cashmere camel coat "
        "neutral tailored minimal high-quality fabrics"
    ),
    "Clean Girl": (
        "clean girl aesthetic minimal gold jewelry slicked hair "
        "neutral tones structured basics monochrome polished"
    ),
    "Coquette": (
        "coquette aesthetic bows ribbons lace feminine pink "
        "delicate romantic soft ballet flats satin"
    ),
    "Office Siren": (
        "office siren confident sleek workwear pencil skirt "
        "structured blazer power dressing fitted corporate chic"
    ),
    "Mob Wife": (
        "mob wife aesthetic fur coat leopard print gold jewelry "
        "bold glamorous chunky accessories dramatic maximalist"
    ),
    "Balletcore": (
        "balletcore ballet wrap top leg warmers satin skirt "
        "soft pink tulle romantic delicate dancewear inspired"
    ),
    "Y2K Street": (
        "y2k streetwear low rise jeans crop top butterfly clips "
        "metallic mini bag parachute pants colorful accessories"
    ),
    "Acubi Minimalism": (
        "acubi minimalism korean fashion clean sneakers silver "
        "micro jewelry neutral muted tones structured modern"
    ),
    "Indie Sleaze": (
        "indie sleaze band tee chunky boots fishnets leather "
        "moody concert grunge eyeliner vintage denim distressed"
    ),
    "Coastal Grandmother": (
        "coastal grandmother linen wide-leg pants straw hat "
        "cream blue stripe nautical relaxed elegant seaside"
    ),
    "Dark Academia": (
        "dark academia tweed blazer pleated skirt oxford shoes "
        "earth tones plaid knit vest literary vintage bookish"
    ),
    "Cottagecore": (
        "cottagecore floral dress puff sleeves lace collar "
        "pastoral romantic vintage prairie gingham embroidered"
    ),
    "Gorpcore": (
        "gorpcore technical outdoor fleece vest hiking boots "
        "utility pockets nylon windbreaker functional sporty"
    ),
    "Soft Girl": (
        "soft girl pastel pink lilac fluffy cardigan pearl "
        "butterfly cute kawaii plush feminine gentle dreamy"
    ),
    "Streetcore": (
        "streetwear oversized hoodie sneakers cargo pants "
        "graphic tee utility streetcore urban bold layered"
    ),
    "Minimalist Scandi": (
        "scandinavian minimalism clean lines neutral palette "
        "oversized wool coat white shirt simple elegant functional"
    ),
    "Old Money": (
        "old money preppy polo ralph lauren cable knit sweater "
        "loafers navy blazer tennis skirt heritage classic"
    ),
    "Athleisure Chic": (
        "athleisure leggings sports bra cropped hoodie sneakers "
        "matching set sleek activewear gym-to-street comfortable"
    ),
    "Romantic Maximalist": (
        "romantic maximalist ruffles voluminous sleeves bold print "
        "dramatic silhouette layered jewelry rich color velvet"
    ),
    "Avant-Garde": (
        "avant-garde deconstructed asymmetric oversized sculptural "
        "monochrome experimental architectural fashion forward"
    ),
}

_trend_embeddings_cache: np.ndarray | None = None
_trend_names_cache: list[str] | None = None


def get_trend_embeddings() -> tuple[list[str], np.ndarray]:
    """Return (trend_names, trend_embeddings_matrix).  Cached after first call."""
    global _trend_embeddings_cache, _trend_names_cache
    if _trend_embeddings_cache is not None and _trend_names_cache is not None:
        return _trend_names_cache, _trend_embeddings_cache

    encoder = get_encoder()
    names = list(TREND_LEXICON.keys())
    prompts = list(TREND_LEXICON.values())
    embeddings = encoder.encode_texts(prompts)
    _trend_names_cache = names
    _trend_embeddings_cache = embeddings
    return names, embeddings


def compute_trend_fingerprint(
    taste_vector: np.ndarray,
) -> dict[str, float]:
    """
    Cosine similarity between the user's taste vector and every trend
    archetype.  Returns a dict sorted by descending similarity.
    Used internally for ranking, anti-taste, and trend boost scoring.
    """
    names, embeddings = get_trend_embeddings()
    tv = np.array(taste_vector, dtype=np.float32)
    norm = np.linalg.norm(tv)
    if norm > 0:
        tv = tv / norm
    sims = (embeddings @ tv).tolist()
    fingerprint = dict(sorted(
        zip(names, sims), key=lambda kv: kv[1], reverse=True
    ))
    return fingerprint


def top_coherent_trends(
    trend_fingerprint: dict[str, float],
    max_trends: int = 3,
) -> dict[str, float]:
    """
    Filter a full fingerprint to only the trends with similarity at least
    1 std-dev above the mean.  Caps at *max_trends* and guarantees at
    least 1 result.  Use this for user-facing display (aesthetic card,
    profile) — not for internal ranking which should use the full fingerprint.
    """
    if not trend_fingerprint:
        return {}

    scores = list(trend_fingerprint.values())
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    threshold = mean_s + std_s

    result: dict[str, float] = {}
    for name, sim in trend_fingerprint.items():
        if sim >= threshold and len(result) < max_trends:
            result[name] = sim
        elif len(result) == 0 and sim < threshold:
            result[name] = sim
            break

    if not result:
        first = next(iter(trend_fingerprint.items()))
        result[first[0]] = first[1]

    return result


def compute_anti_taste_vector(
    trend_fingerprint: dict[str, float], bottom_k: int = 3
) -> np.ndarray:
    """
    Build an anti-taste vector from the user's lowest-scoring trend
    archetypes — the aesthetics they clearly avoid.
    """
    names, embeddings = get_trend_embeddings()
    name_to_idx = {n: i for i, n in enumerate(names)}

    sorted_trends = sorted(trend_fingerprint.items(), key=lambda kv: kv[1])
    bottom = sorted_trends[:bottom_k]

    if not bottom:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    vecs = []
    weights = []
    for name, score in bottom:
        idx = name_to_idx.get(name)
        if idx is not None:
            vecs.append(embeddings[idx])
            weights.append(max(1.0 - score, 0.1))

    if not vecs:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    anti = (np.stack(vecs) * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(anti)
    if norm > 0:
        anti = anti / norm
    return anti.astype(np.float32)


def trend_boost_score(
    candidate_embedding: np.ndarray,
    trend_fingerprint: dict[str, float],
    top_k: int = 5,
) -> float:
    """
    Score how well a candidate item aligns with the user's top-K
    trend preferences.  Returns a value in [0, 1].
    """
    names, embeddings = get_trend_embeddings()
    name_to_idx = {n: i for i, n in enumerate(names)}

    sorted_trends = sorted(
        trend_fingerprint.items(), key=lambda kv: kv[1], reverse=True
    )
    top = sorted_trends[:top_k]

    c_emb = np.array(candidate_embedding, dtype=np.float32)
    c_norm = np.linalg.norm(c_emb)
    if c_norm > 0:
        c_emb = c_emb / c_norm

    weighted_sum = 0.0
    weight_total = 0.0
    for name, user_affinity in top:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        trend_vec = embeddings[idx]
        sim = float(np.dot(c_emb, trend_vec))
        weighted_sum += max(sim, 0.0) * user_affinity
        weight_total += user_affinity

    if weight_total == 0:
        return 0.0
    return float(weighted_sum / weight_total)
