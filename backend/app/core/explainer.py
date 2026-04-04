"""
Template-based explanation generation.

No LLM calls. Selects the best template based on which scoring
signal contributed most: taste modes, trend alignment, outfit
utility, or anti-taste rejection.
"""

TEMPLATES = {
    "trend_taste": "Matches your {trend} vibe — right in line with your Pinterest board.",
    "high_utility_high_taste": "Pairs with {n} of your saves — your highest outfit unlock this week.",
    "high_utility": "Unlocks {unlock_count} new outfit combinations with your existing saves.",
    "taste_match": "Matches your {aesthetic} aesthetic — consistent with your saved palette.",
    "gap_fill": "Fills a {slot} gap in your saves — you're strong in {existing_slot} already.",
    "value": "Great value at this price point for your style.",
    "pairs_with": "Pairs with your saved {item_name} — unlocks {unlock_count} new looks.",
}


def generate_explanation(
    candidate: dict,
    wardrobe_stats: dict,
    pairs: list[dict] | None = None,
    score_context: dict | None = None,
) -> str:
    unlock = candidate.get("unlock_count", 0)
    slot = candidate.get("slot", "")
    aesthetic = wardrobe_stats.get("aesthetic_label", "")

    ctx = score_context or candidate.get("score_context") or {}
    taste_pct = ctx.get("taste_percentile", 0.5)

    trend_score = candidate.get("trend_score", 0.0)

    is_taste_match = taste_pct >= 0.75
    is_high_utility = unlock >= 2
    is_trend_aligned = trend_score > 0.35

    if is_trend_aligned and is_taste_match:
        trend_label = wardrobe_stats.get("top_trend", aesthetic or "curated")
        return TEMPLATES["trend_taste"].format(trend=trend_label)

    if pairs and len(pairs) > 0 and unlock >= 2:
        best_pair = pairs[0]
        pair_name = best_pair.get("title", best_pair.get("slot", "item"))
        return TEMPLATES["pairs_with"].format(
            item_name=pair_name, unlock_count=unlock
        )

    if is_high_utility and is_taste_match:
        wardrobe_items = wardrobe_stats.get("total_items", 0)
        n = min(unlock, wardrobe_items) if wardrobe_items > 0 else unlock
        return TEMPLATES["high_utility_high_taste"].format(n=n)

    if unlock >= 3:
        return TEMPLATES["high_utility"].format(unlock_count=unlock)

    if is_taste_match:
        label = aesthetic if aesthetic else "curated"
        return TEMPLATES["taste_match"].format(aesthetic=label)

    strongest = wardrobe_stats.get("strongest_slot") or "your wardrobe"
    return TEMPLATES["gap_fill"].format(slot=slot or "wardrobe", existing_slot=strongest)
