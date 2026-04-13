"""
Agentic stylist chat powered by Claude Sonnet.

Provides an AI stylist with live tool access to the wardrobe and catalog.
Returns inline item cards and outfit bundles inside the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

import numpy as np
from app.config import get_settings
from app.core.candidates import (
    load_index,
    generate_candidates,
    search_with_filters,
    get_catalog_summary,
)
from app.core.ranker import rank_candidates, rank_shopping, find_pairs, outfit_unlock_count
from app.core.wardrobe import (
    compute_slot_coverage,
    get_gap_slots,
    get_wardrobe_stats,
)
from app.core.outfit_builder import assemble_outfit

logger = logging.getLogger(__name__)

MAX_AGENT_TURNS = 6


def _clean_item(item: dict) -> dict:
    """Strip embeddings from items before returning to client."""
    return {k: v for k, v in item.items() if k != "embedding"}


def _format_catalog_awareness() -> str:
    """Summarize what the catalog stocks so the AI knows before searching."""
    summary = get_catalog_summary()
    if not summary:
        return "CATALOG: unavailable"

    type_lines = []
    for item_type, count in list(summary.get("item_types", {}).items())[:20]:
        type_lines.append(f"  {item_type}: {count}")

    color_lines = []
    for color, count in list(summary.get("colors", {}).items())[:15]:
        color_lines.append(f"  {color}: {count}")

    brands = summary.get("brands", [])
    brand_str = ", ".join(brands[:25])
    price_range = summary.get("price_range", [0, 0])

    return f"""CATALOG INVENTORY ({summary.get('total_items', 0)} items):
Item types stocked:
{chr(10).join(type_lines)}
Colors available:
{chr(10).join(color_lines)}
Brands: {brand_str}
Price range: ${price_range[0]:.0f}–${price_range[1]:.0f}"""


def _build_system_prompt(
    wardrobe: list[dict],
    taste_profile: dict,
) -> str:
    """Build the system prompt with wardrobe context, taste profile, and catalog awareness."""
    stats = get_wardrobe_stats(wardrobe)
    coverage = compute_slot_coverage(wardrobe)
    gaps = get_gap_slots(coverage)

    attrs = taste_profile.get("aesthetic_attributes", {})
    silhouette = attrs.get("silhouette", {}).get("label", "unknown")
    color_story = attrs.get("color_story", {}).get("label", "unknown")
    formality = attrs.get("formality", {}).get("label", "unknown")
    price_tier = taste_profile.get("price_tier", [40, 200])

    trend_fp = taste_profile.get("trend_fingerprint", {})
    top_trends = list(trend_fp.keys())[:3] if trend_fp else []

    wardrobe_summary = []
    for slot, items in coverage.items():
        if items:
            names = [item.get("title", "Unknown") for item in items[:4]]
            wardrobe_summary.append(f"  {slot} ({len(items)}): {', '.join(names)}")

    catalog_info = _format_catalog_awareness()

    return f"""You are a personal stylist. You know this person's wardrobe, their taste, and what they actually wear. You talk to them like a friend who happens to have great taste and knows exactly what to pull.

{catalog_info}

WARDROBE ({stats['total_items']} items):
{json.dumps(stats['slot_counts'])}
Gaps: {', '.join(gaps) if gaps else 'none'}
Items:
{chr(10).join(wardrobe_summary) if wardrobe_summary else '  (empty)'}

TASTE: {silhouette} / {color_story} / {formality} / ${price_tier[0]:.0f} to ${price_tier[1]:.0f}
Trends: {', '.join(top_trends) if top_trends else 'n/a'}

HOW YOU WRITE:
- Warm, direct, and specific. Like a text from a friend who knows fashion.
- Short sentences. No filler. No throat-clearing.
- Never use em dashes or long dashes. Use commas or just start a new sentence.
- Never say "Let me search", "I'll look for", "Let me check". Search silently, then talk.
- Lead with one sharp sentence of reasoning, then show the items immediately.
- Reference the person's actual pieces by name when explaining what works with what.
- One confident recommendation beats three hedged ones.
- Never invent items. Only recommend what the tools actually return."""


def _build_tools() -> list[dict]:
    """Define the tools available to the stylist agent."""
    return [
        {
            "name": "search_catalog",
            "description": (
                "Search the catalog for items. Use item_type and colors for specific requests "
                "(e.g. user asks for 'red dress' → item_type='dress', colors=['red']). "
                "Use only the query field for exploratory/vague requests."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what to search for",
                    },
                    "item_type": {
                        "type": "string",
                        "description": (
                            "Specific item type to filter by: dress, skirt, jeans, trousers, "
                            "blouse, camisole, sweater, blazer, jacket, coat, sneakers, boots, "
                            "heels, sandals, loafers, flats, bag, tote, necklace, earrings, etc."
                        ),
                    },
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Color filters: black, white, navy, blue, red, burgundy, pink, "
                            "green, yellow, orange, purple, brown, tan, grey, metallic, denim, etc."
                        ),
                    },
                    "slot": {
                        "type": "string",
                        "enum": ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"],
                        "description": "Optional: filter to a specific outfit slot",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "analyze_wardrobe",
            "description": "Analyze the user's wardrobe: gaps, coverage, utility scores, and what combinations are possible.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "build_outfit",
            "description": "Build a complete outfit for a specific occasion using wardrobe items + optional catalog addition.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "occasion": {
                        "type": "string",
                        "enum": ["work", "casual", "evening", "weekend", "special"],
                        "description": "The occasion to build the outfit for",
                    },
                },
                "required": ["occasion"],
            },
        },
        {
            "name": "score_item",
            "description": "Score a specific catalog item against the user's wardrobe and taste. Returns taste fit, outfit unlock count, and pairing suggestions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The catalog item ID to score",
                    },
                },
                "required": ["item_id"],
            },
        },
    ]


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    wardrobe: list[dict],
    taste_vector: np.ndarray,
    taste_modes: list[np.ndarray] | None,
    trend_fingerprint: dict[str, float] | None,
    anti_taste_vector: np.ndarray | None,
    price_tier: tuple[float, float],
    style_attributes: dict[str, float] | None = None,
) -> tuple[str, list[dict]]:
    """
    Execute a tool and return (result_text, items_to_emit).
    items_to_emit are full item dicts to render as cards.
    """
    items_to_emit: list[dict] = []

    if tool_name == "search_catalog":
        query = tool_input.get("query", "")
        slot_filter = tool_input.get("slot")
        item_type = tool_input.get("item_type")
        colors = tool_input.get("colors")
        top_k = tool_input.get("top_k", 5)

        from app.core.clip_encoder import get_encoder
        encoder = get_encoder()
        query_emb = encoder.encode_texts([query])[0]

        is_shopping = bool(item_type or colors)
        owned_ids = {item["item_id"] for item in wardrobe}

        if is_shopping:
            # Shopping mode: staged metadata search → shopping ranker
            slots_filter = [slot_filter] if slot_filter else None
            candidates, search_stage = search_with_filters(
                query_vector=query_emb,
                top_k=top_k * 5,
                slots=slots_filter,
                item_type=item_type,
                colors=colors,
                exclude_ids=owned_ids,
                price_tier=price_tier,
            )
            ranked = rank_shopping(
                candidates,
                query_vector=query_emb,
                wardrobe=wardrobe,
                taste_vector=taste_vector,
                top_k=top_k,
            )
        else:
            # Discovery/stylist mode: gap-filling round-robin + taste ranking
            search_stage = "discovery"
            gap_slots = [slot_filter] if slot_filter else [
                "tops", "bottoms", "outerwear", "shoes", "bags", "accessories"
            ]
            candidates = generate_candidates(
                taste_vector=query_emb,
                gap_slots=gap_slots,
                price_tier=price_tier,
                top_k=top_k * 3,
                exclude_ids=owned_ids,
            )
            if taste_modes:
                ranked = rank_candidates(
                    candidates, wardrobe, taste_vector,
                    taste_modes=taste_modes,
                    trend_fingerprint=trend_fingerprint,
                    anti_taste_vector=anti_taste_vector,
                    intent_bias=0.2,
                    query_vector=query_emb,
                    style_attributes=style_attributes or {},
                )
            else:
                ranked = candidates

        results = ranked[:top_k]
        items_to_emit = [_clean_item(r) for r in results]

        # Build result summary with fallback context
        stage_labels = {
            "exact": "exact match",
            "relaxed_color": "related colors (exact type wasn't available in the requested color)",
            "type_only": "matching type, any color (requested color not in stock)",
            "semantic": "closest visual matches (requested item type not in catalog)",
            "discovery": "discovery mode",
        }
        stage_note = stage_labels.get(search_stage, search_stage)

        header = f"[Search mode: {stage_note}]"
        if is_shopping and search_stage != "exact":
            requested = []
            if item_type:
                requested.append(f"type={item_type}")
            if colors:
                requested.append(f"colors={colors}")
            header += f"\nRequested: {', '.join(requested)}. Showing best available alternatives."

        summaries = [header]
        for r in results:
            line = f"- {r['title']} by {r['brand']} (${r['price']}, {r['slot']}"
            if r.get("item_type"):
                line += f", type={r['item_type']}"
            if r.get("colors"):
                line += f", colors={r['colors']}"
            line += f") [id: {r['item_id']}]"
            line += f"\n  scores: query={r.get('query_score', 0):.2f}, taste={r.get('taste_score', 0):.2f}"
            summaries.append(line)

        if len(summaries) <= 1:
            no_match = f"No items found for query='{query}'"
            if item_type:
                no_match += f", item_type={item_type}"
            if colors:
                no_match += f", colors={colors}"
            summary = get_catalog_summary()
            available_types = list(summary.get("item_types", {}).keys())[:10]
            available_colors = list(summary.get("colors", {}).keys())[:10]
            no_match += f"\nAvailable item types: {', '.join(available_types)}"
            no_match += f"\nAvailable colors: {', '.join(available_colors)}"
            no_match += "\nSuggest alternatives from what's actually in stock."
            return no_match, items_to_emit

        return "\n".join(summaries), items_to_emit

    elif tool_name == "analyze_wardrobe":
        stats = get_wardrobe_stats(wardrobe)
        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)

        lines = [
            f"Total items: {stats['total_items']}",
            f"Slot coverage: {json.dumps(stats['slot_counts'])}",
            f"Gaps (0-1 items): {', '.join(gaps) if gaps else 'No gaps — well rounded!'}",
            f"Strongest slot: {stats.get('strongest_slot', 'none')}",
        ]

        if wardrobe:
            lines.append("\nItems by slot:")
            for slot, items in coverage.items():
                if items:
                    for item in items:
                        unlock = outfit_unlock_count(item, wardrobe)
                        lines.append(
                            f"  [{slot}] {item.get('title', 'Unknown')} "
                            f"(${item.get('price', 0)}) — enables {unlock} outfits"
                        )

        return "\n".join(lines), []

    elif tool_name == "build_outfit":
        occasion = tool_input.get("occasion", "casual")
        result = assemble_outfit(
            wardrobe=wardrobe,
            occasion=occasion,
            taste_vector=taste_vector,
            taste_modes=taste_modes,
            trend_fingerprint=trend_fingerprint,
            anti_taste_vector=anti_taste_vector,
            price_tier=price_tier,
        )

        all_items = result["wardrobe_items"]
        if result["catalog_addition"]:
            all_items = all_items + [result["catalog_addition"]]
        items_to_emit = all_items

        lines = [
            f"Outfit: {result['title']}",
            f"Harmony score: {result['harmony_score']}",
            f"Complete: {'Yes' if result['is_complete'] else 'No'}",
            f"Rationale: {result['rationale']}",
            "\nItems:",
        ]
        for item in result["wardrobe_items"]:
            lines.append(f"  [wardrobe] {item.get('title', 'Unknown')} ({item.get('slot', '')})")
        if result["catalog_addition"]:
            add = result["catalog_addition"]
            lines.append(f"  [ADD THIS] {add.get('title', 'Unknown')} ({add.get('slot', '')}) — ${add.get('price', 0)}")

        return "\n".join(lines), items_to_emit

    elif tool_name == "score_item":
        item_id = tool_input.get("item_id", "")
        try:
            _, catalog = load_index()
        except FileNotFoundError:
            return "Catalog not loaded.", []

        item = next((i for i in catalog if i["item_id"] == item_id), None)
        if not item:
            return f"Item {item_id} not found in catalog.", []

        item_emb = np.array(item["embedding"], dtype=np.float32)
        taste_fit = max(0.0, float(np.dot(taste_vector, item_emb)))
        unlock = outfit_unlock_count(item, wardrobe)
        pairs = find_pairs(item, wardrobe)

        items_to_emit = [_clean_item(item)]

        pair_names = [p.get("title", "Unknown") for p in pairs[:3]]
        confidence = "HIGH" if taste_fit > 0.5 and unlock >= 2 else \
                     "MEDIUM" if taste_fit > 0.35 or unlock >= 1 else "LOW"

        lines = [
            f"Item: {item['title']} by {item['brand']} (${item['price']})",
            f"Taste Fit: {taste_fit:.0%}",
            f"Purchase Confidence: {confidence}",
            f"Outfit Unlocks: {unlock} new combinations",
            f"Pairs with: {', '.join(pair_names) if pair_names else 'no existing saves'}",
        ]

        return "\n".join(lines), items_to_emit

    return "Unknown tool.", []


async def run_stylist_chat(
    messages: list[dict],
    wardrobe: list[dict],
    taste_profile: dict,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agentic stylist chat loop. Yields SSE events:
    - {type: "text", content: "..."}   ← streamed token-by-token
    - {type: "item_card", item: {...}}  ← emitted after tool execution
    - {type: "outfit_bundle", items: [...], title: "...", occasion: "..."}
    - {type: "done"}
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        yield {"type": "text", "content": "Chat is not available — ANTHROPIC_API_KEY not configured."}
        yield {"type": "done"}
        return

    import anthropic

    # Use async client so we can stream tokens without blocking the event loop
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    taste_vector = np.array(taste_profile.get("taste_vector", []), dtype=np.float32)
    taste_modes = [np.array(m, dtype=np.float32) for m in taste_profile.get("taste_modes", [])] or None
    trend_fp = taste_profile.get("trend_fingerprint")
    anti_taste = np.array(taste_profile.get("anti_taste_vector", []), dtype=np.float32) if taste_profile.get("anti_taste_vector") else None
    price_tier = tuple(taste_profile.get("price_tier", [40.0, 200.0]))
    style_attrs = taste_profile.get("style_attributes", {})

    system_prompt = _build_system_prompt(wardrobe, taste_profile)
    tools = _build_tools()

    # Convert messages to Anthropic format
    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            api_messages.append({"role": role, "content": content})

    for turn in range(MAX_AGENT_TURNS):
        try:
            # Stream the response so text tokens arrive incrementally
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=system_prompt,
                tools=tools,
                messages=api_messages,
            ) as stream:
                # Yield text tokens as they arrive (empty during tool-call turns)
                async for text_chunk in stream.text_stream:
                    if text_chunk:
                        yield {"type": "text", "content": text_chunk}

                # Get the complete response for tool-use processing
                response = await stream.get_final_message()

        except Exception as e:
            logger.exception("Anthropic API error")
            err_msg = str(e)
            if "credit balance" in err_msg.lower() or "billing" in err_msg.lower():
                yield {"type": "text", "content": "The Anthropic API key needs credits to use the stylist chat. Please add credits at console.anthropic.com."}
            else:
                yield {"type": "text", "content": f"Sorry, I encountered an error: {err_msg}"}
            yield {"type": "done"}
            return

        # Process tool_use blocks from the completed response
        assistant_content = []
        has_tool_use = False

        for block in response.content:
            if block.type == "text":
                # Text was already streamed above; just track for message history
                assistant_content.append(block)

            elif block.type == "tool_use":
                has_tool_use = True
                assistant_content.append(block)

                # Run synchronous tool execution in a thread so we don't block the loop
                result_text, items = await asyncio.to_thread(
                    _execute_tool,
                    block.name,
                    block.input,
                    wardrobe,
                    taste_vector,
                    taste_modes,
                    trend_fp,
                    anti_taste,
                    price_tier,
                    style_attrs,
                )

                # Emit item cards or outfit bundles
                if block.name == "build_outfit" and items:
                    yield {
                        "type": "outfit_bundle",
                        "items": items,
                        "title": block.input.get("occasion", "Look").title(),
                        "occasion": block.input.get("occasion", "casual"),
                    }
                else:
                    for item in items:
                        yield {"type": "item_card", "item": item}

                # Append assistant turn and tool result for next iteration
                api_messages.append({"role": "assistant", "content": assistant_content})
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }],
                })
                assistant_content = []

        # If no tool was used this turn, Claude has finished responding
        if not has_tool_use:
            break

        # Keep any trailing assistant content in the message history
        if assistant_content:
            api_messages.append({"role": "assistant", "content": assistant_content})

    yield {"type": "done"}
