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
from app.core.ranker import rank_candidates, rank_shopping, find_pairs, outfit_unlock_count, compatibility_score
from app.core.wardrobe import (
    compute_slot_coverage,
    get_gap_slots,
    get_wardrobe_stats,
)
from app.core.outfit_builder import assemble_outfit, curate_outfit_from_catalog

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

TOOL ROUTING (follow strictly):
- "build me a work outfit", "complete look for X", "what should I wear to Y": use curate_outfit. This retrieves candidates per slot, filters by occasion, and scores the combination jointly so everything goes together. Never use multiple search_catalog calls to assemble an outfit.
- "find me a blazer", "show me black boots", single-item requests: use search_catalog.
- "add a blazer to that outfit", follow-ups referencing a previous outfit: use search_catalog with a specific garment query. The tool will automatically score results against the last outfit for compatibility.
- "what's in my closet", "what am I missing": use analyze_wardrobe.
- build_outfit is for wardrobe-only outfits (what can I wear from what I own).

TEXT AND CARD CONTRACT (critical):
- You may ONLY describe items that the tools actually returned. The tool output lists every item by name and brand. Do not mention any item, brand, or product that is not in that list.
- Write your explanation FIRST, then the cards appear after. Your text should reference items by the exact names from the tool results.
- If a tool returns items, always write at least 1-2 sentences explaining why these pieces work before the cards show.

HOW YOU WRITE:
- Warm, direct, and specific. Like a text from a friend who happens to have great taste.
- Short sentences. No filler. No throat-clearing.
- Never use em dashes or long dashes. Use commas or just start a new sentence.
- Never say "Let me search", "I'll look for", "Let me check". Search silently, then talk.
- Lead with one sharp sentence of reasoning, then show the items immediately.
- Reference the person's actual pieces by name when explaining what works with what.
- One confident recommendation beats three hedged ones.
- Never invent items. Only recommend what the tools actually return.

NEVER IN USER-FACING COPY:
- Never mention internal scores, ranking signals, metric names, or any decimal or percentage values from tools.
- Never quote or paraphrase numeric evaluations. Say how things look and feel in plain language.
- Tool blurbs may include qualitative hints for you only. Translate them into human language with zero numbers.

HOW YOU SEARCH:
- When searching for work or professional pieces, use specific descriptive queries: "tailored trousers straight leg" not "work pants", "structured blazer office" not "work top", "polished midi skirt pencil" not "work skirt". Occasion words alone return casual results. Describe the garment shape, fabric, and formality.
- Always prefer garment-level descriptions over occasion categories. "Silk button-down blouse" beats "nice top for work." The catalog search is visual, so the more precise and visual your query, the better the results."""


def _build_tools() -> list[dict]:
    """Define the tools available to the stylist agent."""
    return [
        {
            "name": "search_catalog",
            "description": (
                "Search the catalog for individual items. Use for single-item lookups "
                "(find a blazer, show me boots). NOT for building full outfits."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive natural language query. Use garment-level terms (tailored trousers straight leg, structured wool blazer) not vague occasion words (work pants).",
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
            "name": "curate_outfit",
            "description": (
                "Build a complete shoppable outfit for an occasion. Retrieves candidates per slot "
                "(top, bottom, shoes, outerwear), filters for occasion appropriateness, and scores "
                "the combination as a set so everything goes together. Use this for 'build me a work outfit', "
                "'what should I wear to brunch', or any full-outfit request."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "occasion": {
                        "type": "string",
                        "enum": ["work", "casual", "evening", "weekend", "special"],
                        "description": "The occasion to curate an outfit for",
                    },
                },
                "required": ["occasion"],
            },
        },
        {
            "name": "build_outfit",
            "description": "Build an outfit from the user's existing wardrobe items + one optional catalog addition. Use for 'what can I wear from my closet to work' style questions.",
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
            "name": "analyze_wardrobe",
            "description": "Analyze the user's wardrobe: gaps, coverage, and what combinations are possible.",
            "input_schema": {
                "type": "object",
                "properties": {},
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
    last_outfit_items: list[dict] | None = None,
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
                trend_fingerprint=trend_fingerprint,
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

        # If there are context items from a previous outfit, score compatibility
        if last_outfit_items and results:
            for r in results:
                r_emb = np.array(r.get("embedding", []), dtype=np.float32)
                if r_emb.size == 0:
                    r["_ctx_compat"] = 0.0
                    continue
                scores = []
                for ctx in last_outfit_items:
                    ctx_emb = ctx.get("embedding")
                    if ctx_emb is None:
                        continue
                    s = compatibility_score(
                        r_emb, np.array(ctx_emb, dtype=np.float32),
                        r.get("slot", ""), ctx.get("slot", ""),
                        r.get("dominant_color", ""), ctx.get("dominant_color", ""),
                    )
                    scores.append(s)
                r["_ctx_compat"] = float(np.mean(scores)) if scores else 0.0
            results.sort(key=lambda x: x.get("_ctx_compat", 0), reverse=True)

        items_to_emit = [_clean_item(r) for r in results]

        stage_labels = {
            "exact": "exact match",
            "relaxed_color": "related colors",
            "type_only": "matching type, any color",
            "semantic": "closest visual matches",
            "discovery": "discovery mode",
        }
        stage_note = stage_labels.get(search_stage, search_stage)

        header = f"[Search: {stage_note}]"
        summaries = [header]
        for r in results:
            line = f"- {r['title']} by {r['brand']} (${r['price']}, {r['slot']}"
            if r.get("item_type"):
                line += f", type={r['item_type']}"
            line += f") [id: {r['item_id']}]"
            if r.get("_ctx_compat", 0) > 0:
                compat = r["_ctx_compat"]
                fit_note = (
                    "Excellent visual match with the current outfit."
                    if compat >= 0.7
                    else "Works well with the current outfit."
                    if compat >= 0.5
                    else "Okay fit with the current outfit, not the strongest pairing."
                )
                line += f"\n  {fit_note}"
            summaries.append(line)

        summaries.append("\nIMPORTANT: You may ONLY describe items listed above. Do not invent or reference any other items.")

        if len(summaries) <= 2:
            no_match = f"No items found for query='{query}'"
            summary = get_catalog_summary()
            available_types = list(summary.get("item_types", {}).keys())[:10]
            no_match += f"\nAvailable item types: {', '.join(available_types)}"
            no_match += "\nSuggest alternatives from what's actually in stock."
            return no_match, items_to_emit

        return "\n".join(summaries), items_to_emit

    elif tool_name == "curate_outfit":
        occasion = tool_input.get("occasion", "casual")

        context = last_outfit_items if last_outfit_items else None
        result = curate_outfit_from_catalog(
            occasion=occasion,
            taste_vector=taste_vector,
            taste_modes=taste_modes,
            trend_fingerprint=trend_fingerprint,
            anti_taste_vector=anti_taste_vector,
            price_tier=price_tier,
            wardrobe=wardrobe,
            context_items=context,
        )

        items_to_emit = result["items"]

        hs = float(result.get("harmony_score", 0) or 0)
        cohesion = (
            "These pieces read as one cohesive outfit."
            if hs >= 0.7
            else "The combination is workable with a clear anchor piece."
            if hs >= 0.5
            else "The mix may need a swap to feel intentional."
        )

        lines = [
            f"Outfit: {result['title']}",
            cohesion,
            f"Slots filled: {', '.join(result['filled_slots'])}",
        ]
        if result["missing_slots"]:
            lines.append(f"Could not find occasion-appropriate items for: {', '.join(result['missing_slots'])}")

        lines.append("\nItems in this outfit (reference ONLY these by name):")
        for it in result["items"]:
            lines.append(f"  [{it['slot']}] {it.get('title', 'Unknown')} by {it.get('brand', '?')} (${it.get('price', 0)})")

        lines.append("\nIMPORTANT: You may ONLY describe items listed above. Do not invent or reference any other items, brands, or products.")

        return "\n".join(lines), items_to_emit

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

        hs = float(result.get("harmony_score", 0) or 0)
        cohesion = (
            "These pieces read as one cohesive outfit."
            if hs >= 0.7
            else "The combination is workable with a clear anchor piece."
            if hs >= 0.5
            else "The mix may need a swap to feel intentional."
        )
        lines = [
            f"Outfit: {result['title']}",
            cohesion,
            f"Complete: {'Yes' if result['is_complete'] else 'No'}",
            f"Rationale: {result['rationale']}",
            "\nItems (reference ONLY these by name):",
        ]
        for item in result["wardrobe_items"]:
            lines.append(f"  [wardrobe] {item.get('title', 'Unknown')} ({item.get('slot', '')})")
        if result["catalog_addition"]:
            add = result["catalog_addition"]
            lines.append(f"  [catalog] {add.get('title', 'Unknown')} ({add.get('slot', '')}) ${add.get('price', 0)}")

        lines.append("\nIMPORTANT: You may ONLY describe items listed above. Do not invent or reference any other items.")

        return "\n".join(lines), items_to_emit

    elif tool_name == "analyze_wardrobe":
        stats = get_wardrobe_stats(wardrobe)
        coverage = compute_slot_coverage(wardrobe)
        gaps = get_gap_slots(coverage)

        lines = [
            f"Total items: {stats['total_items']}",
            f"Slot coverage: {json.dumps(stats['slot_counts'])}",
            f"Gaps (0-1 items): {', '.join(gaps) if gaps else 'No gaps, well rounded!'}",
            f"Strongest slot: {stats.get('strongest_slot', 'none')}",
        ]

        if wardrobe:
            lines.append("\nItems by slot:")
            for slot, items in coverage.items():
                if items:
                    for item in items:
                        unlock = outfit_unlock_count(item, wardrobe)
                        util = (
                            "many new outfit combinations"
                            if unlock >= 4
                            else "several new outfit combinations"
                            if unlock >= 2
                            else "a few new pairings"
                            if unlock >= 1
                            else "limited new combinations with the rest of the closet"
                        )
                        lines.append(
                            f"  [{slot}] {item.get('title', 'Unknown')} "
                            f"(${item.get('price', 0)}) — {util}"
                        )

        return "\n".join(lines), []

    return "Unknown tool.", []


async def run_stylist_chat(
    messages: list[dict],
    wardrobe: list[dict],
    taste_profile: dict,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agentic stylist chat loop. Yields SSE events:
    - {type: "text", content: "..."}   <- streamed token-by-token
    - {type: "item_card", item: {...}}  <- emitted after tool execution
    - {type: "outfit_bundle", items: [...], title: "...", occasion: "..."}
    - {type: "done"}
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        yield {"type": "text", "content": "Chat is not available, ANTHROPIC_API_KEY not configured."}
        yield {"type": "done"}
        return

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    taste_vector = np.array(taste_profile.get("taste_vector", []), dtype=np.float32)
    taste_modes = [np.array(m, dtype=np.float32) for m in taste_profile.get("taste_modes", [])] or None
    trend_fp = taste_profile.get("trend_fingerprint")
    anti_taste = np.array(taste_profile.get("anti_taste_vector", []), dtype=np.float32) if taste_profile.get("anti_taste_vector") else None
    price_tier = tuple(taste_profile.get("price_tier", [40.0, 200.0]))
    style_attrs = taste_profile.get("style_attributes", {})

    system_prompt = _build_system_prompt(wardrobe, taste_profile)
    tools = _build_tools()

    # Session context: track last emitted outfit/items for follow-up compatibility
    last_outfit_items: list[dict] = []

    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            api_messages.append({"role": role, "content": content})

    for turn in range(MAX_AGENT_TURNS):
        try:
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=system_prompt,
                tools=tools,
                messages=api_messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    if text_chunk:
                        yield {"type": "text", "content": text_chunk}

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

        assistant_content = []
        has_tool_use = False
        turn_emitted_any_items = False
        turn_emitted_any_text = any(
            block.type == "text" and block.text.strip()
            for block in response.content
        )

        for block in response.content:
            if block.type == "text":
                assistant_content.append(block)

            elif block.type == "tool_use":
                has_tool_use = True
                assistant_content.append(block)

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
                    last_outfit_items,
                )

                # Update session context with emitted items (keep embeddings for follow-ups)
                if items:
                    turn_emitted_any_items = True
                    if block.name in ("curate_outfit", "build_outfit"):
                        last_outfit_items = items[:]
                    else:
                        last_outfit_items = items[:]

                if block.name in ("curate_outfit", "build_outfit") and items:
                    yield {
                        "type": "outfit_bundle",
                        "items": [_clean_item(it) for it in items],
                        "title": block.input.get("occasion", "Look").title(),
                        "occasion": block.input.get("occasion", "casual"),
                    }
                else:
                    for item in items:
                        yield {"type": "item_card", "item": _clean_item(item)}

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

        if not has_tool_use:
            # If we emitted items but the model produced no text, add a fallback
            if turn_emitted_any_items and not turn_emitted_any_text:
                yield {"type": "text", "content": "Here are some pieces I pulled together for you."}
            break

        if assistant_content:
            api_messages.append({"role": "assistant", "content": assistant_content})

    yield {"type": "done"}
