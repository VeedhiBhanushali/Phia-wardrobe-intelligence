# Wardrobe IQ — Full Technical Write-Up
**Phia × Wardrobe IQ: Building a Three-Signal Purchase Intelligence System**  
*Veedhi Bhanushali · April 2026*

---

## The Premise

Phia is a fashion price-comparison platform — it answers *"is this the best price for this item?"* That is a necessary but insufficient answer. The real question a shopper asks is: *"should I buy this, given who I am and what I already own?"*

This project is the full working proof of concept for a personalization layer that answers that harder question. It runs live, produces measurable results, and maps directly onto Phia's Series A roadmap: cold-start personalization, real-time intent inference, and purchase quality prediction.

The core idea is a **three-signal model**:

```
Taste     (persistent)  — aesthetic identity, built from Pinterest + save history
Wardrobe  (persistent)  — what you own, updated with every save  
Intent    (ephemeral)   — what you need right now, from browsing or chat
```

Every recommendation is the intersection of all three. The system knows who you are, what you have, and what you're looking for — and uses all of it simultaneously.

---

## Product Architecture

The product has four surfaces, each mapping to one or more of the three signals:

| Surface | Signal(s) | State |
|---|---|---|
| Wardrobe Builder | Wardrobe | Persistent |
| Aesthetic Builder | Taste | Persistent |
| Discovery Feed | All three combined | Persistent + session |
| Stylist Chat | Intent (explicit) | Ephemeral |

The **Wardrobe Builder** is a virtual closet — users browse the catalog by slot (Tops / Bottoms / Outerwear / Shoes / Bags / Accessories), save items, and each saved item immediately shows how many new outfit combinations it enables.

The **Aesthetic Builder** extracts a taste profile from a Pinterest board URL. One paste. ~3 seconds. The output is a named aesthetic ("Quiet Luxury", "Clean Girl", "Coquette"), dominant color swatches, a 25-axis style preference profile, and a trend fingerprint across 20 named fashion aesthetics.

The **Discovery Feed** is the main surface: horizontally scrolling sections for wardrobe gap fills, pure aesthetic matches, outfit completions, occasion-specific picks, and price signals on saved items.

The **Stylist Chat** is a full-screen agentic conversation powered by Claude. It has live tool access to the wardrobe, catalog, and taste profile — and it reasons about all three signals simultaneously to produce shoppable outfit recommendations, gap fills, and item evaluations.

---

## Tech Stack

**Backend**
- Python 3.13, FastAPI, Uvicorn
- FashionCLIP (patrickjohncyh/fashion-clip) — ViT-B/32, 512-d embeddings, fine-tuned on 800K Farfetch image-text pairs
- FAISS (IndexFlatIP) — approximate nearest neighbor search on L2-normalized embeddings
- NumPy — all vector math
- scikit-learn (KMeans) — taste mode extraction
- Anthropic Claude API (claude-sonnet-4-20250514) — agentic chat with tool use
- Playwright, httpx — catalog data ingestion (SerpAPI, HuggingFace)

**Frontend**
- Next.js 15, React, TypeScript
- Tailwind CSS — design system
- Framer Motion — streaming animations, spring transitions
- Server-Sent Events (SSE) — real-time token streaming + item card delivery

**Data**
- Catalog: 2,364 items from the `ashraq/fashion-product-images-small` HuggingFace dataset, encoded with FashionCLIP image encoder
- FAISS index persisted as `faiss_index.bin` with catalog metadata as `catalog_cache.json`
- User state: localStorage (taste profile, wardrobe items, skipped items); sent to backend per request

---

## Signal 1: Taste — Cold-Start Personalization via Visual Extraction

### The Problem

Every new Phia user sees the same generic feed. Engagement-based personalization needs purchase history that doesn't exist yet. Getting to a meaningful first impression without any interaction data is one of the hardest problems in recommender systems.

### The Approach

Users paste a Pinterest board URL. The backend scrapes the board images, runs them through **FashionCLIP** (a CLIP variant fine-tuned on 800K Farfetch fashion image-text pairs), and produces 512-dimensional L2-normalized embeddings that encode aesthetic information at a fashion-specific level — silhouettes, fabrics, construction details, color palette.

The core pipeline in `taste.py`:

1. **Global taste vector**: Weighted mean of all board image embeddings, normalized to unit length. Pinterest images get weight 0.6; wardrobe saves get weight 1.0 (stronger signal).

2. **Taste modes via KMeans**: A user's taste is not one thing — their work style differs from their weekend aesthetic. KMeans ($k = \min(4, \lfloor N/5 \rfloor)$) clusters the image embeddings and returns per-cluster centroids as taste modes. The ranker scores each candidate against all modes and takes the best, so a user with both minimalist work taste and coquette weekend taste gets appropriate recommendations for each context.

3. **Per-occasion taste vectors**: Each image is classified against 5 occasion context embeddings ("professional office workwear business formal blazer...", "casual everyday relaxed...") by cosine similarity. Images are bucketed by their best-matching occasion. Each bucket with ≥2 images gets its own normalized centroid. This lets the system know that a user's "work" aesthetic might be Sharp/Structured while their "weekend" is Coastal Grandmother.

4. **Trend fingerprinting**: The taste vector is projected against 20 named fashion aesthetics from the trend lexicon (Quiet Luxury, Clean Girl, Coquette, Office Siren, Dark Academia, Cottagecore, Gorpcore, etc.) via cosine similarity. The result is a `trend_fingerprint` dict sorted by descending similarity. The top trend drives retrieval bias; all 20 scores drive trend boost scoring at ranking time.

5. **Anti-taste vector**: The bottom-3 trend archetypes (aesthetics the user clearly avoids) are combined into a weighted mean that forms an anti-taste vector. This is used as a penalty in the ranker — items that match the user's avoided aesthetics are down-ranked.

6. **25-axis style attribute profile**: A fine-grained preference system across 25 semantic axes covering pattern (solid, floral, geometric, stripe, animal, polka dot), material (silk, knit, leather, denim, linen), branding (minimal, logo), color (neutral, dark, warm, colorful, pastel), fit (oversized, tailored, flowy), and vibe (minimal, vintage, preppy, romantic, sporty). Each axis is a (positive_pole, negative_pole) text prompt pair encoded by FashionCLIP. The user's image embeddings are projected against each axis to produce a score in [-1, +1]. Axes with |score| < 0.3 are ignored at ranking time (no detectable preference). Axes with clear preferences generate per-item mismatch penalties.

7. **Incremental profile update**: After each save, `update_taste_profile` applies an exponential moving average — the existing vector gets weight $\frac{n}{n+1}$, the new item embedding gets $\frac{1}{n+1}$. Style attributes update with a learning rate of $\max(0.08, \frac{1}{n+1})$ — faster learning early, smaller corrections once the profile stabilizes.

### Taste Extraction Code Flow

```
Pinterest URL / uploaded images
    → FashionCLIP encode_images() → (N, 512) embeddings
    → build_taste_vector(embeddings, weights) → 512-d L2-normalized centroid
    → extract_taste_modes(embeddings) → KMeans clusters → list of centroids
    → classify_by_occasion(embeddings) → bucket images by occasion
    → build_occasion_vectors(embeddings, buckets) → per-occasion taste vectors
    → compute_trend_fingerprint(taste_vector) → {trend: cosine_sim} dict
    → compute_anti_taste_vector(trend_fingerprint) → 512-d anti-taste
    → extract_attributes(taste_vector) → silhouette/color/formality/occasion labels
    → compute_style_attribute_profile(embeddings, encoder) → 25-axis float dict
    → style_attribute_summary(profile) → top-5 distinctive preferences for UI
```

### Results

| Metric | Result |
|--------|--------|
| Mean taste cosine @10 vs random baseline | **+102%** at zero saves |
| Mean taste cosine @10 vs catalog centroid | **+52%** at zero saves |
| Mean taste cosine @10 vs random (5 saves) | **+90%** |
| Mean taste cosine @10 vs random (10 saves) | **+91%** |

Personalization is meaningful before any interaction history exists. The taste signal remains stable as wardrobe context is introduced.

**Known limitation**: FashionCLIP compresses similarity scores in the minimalist/neutral/clean region of the embedding space — items within "quiet luxury" have narrower cosine distance between matches and non-matches. This causes the purchase confidence model to score ~30% on minimalist test profiles (vs ~90% on streetwear). A production system would address this with domain-specific fine-tuning or a larger catalog with finer-grained labels.

---

## Signal 2: Wardrobe — Outfit Utility Intelligence

The wardrobe is not just storage. It drives three things that change every recommendation:

### Wardrobe Embedding Blend

`wardrobe.py` builds a recency-weighted wardrobe embedding from saved items — more recent saves get higher weight ($\frac{1}{n - i}$ for item at index $i$, then normalized). This is then blended with the taste vector at a schedule based on save count:

| Saves | Taste weight | Wardrobe weight |
|-------|-------------|-----------------|
| 0     | 100%        | 0%              |
| 1–4   | 85%         | 15%             |
| 5–14  | 65%         | 35%             |
| 15+   | 45%         | 55%             |

Conservative early schedule prevents cross-modal averaging from diluting the taste signal before there are enough wardrobe items to anchor it.

### Outfit Unlock Count

Every item in the feed shows `+N outfits` — how many valid outfit combinations it enables with the user's current wardrobe. This is the core wardrobe intelligence metric.

`outfit_unlock_count(candidate, wardrobe)` computes this by:
1. For each wardrobe item in a different slot, computing `compatibility_score(candidate, wardrobe_item)`
2. Grouping compatible items by slot (e.g., 3 compatible tops, 2 compatible shoes)
3. Iterating over all valid outfit slot combinations (tops+bottoms, tops+bottoms+shoes, tops+bottoms+outerwear+shoes, etc.) that include the candidate's slot
4. Multiplying the compatible counts for each other slot in the combination
5. Summing across all combinations

Result: a candidate shoe that pairs with 3 of the user's tops and 2 of their bottoms and 1 of their bags scores 3×2×1 = 6 outfit unlocks from `tops+bottoms+shoes+bags` alone.

### Compatibility Scoring

The compatibility score between two items is not raw cosine similarity — raw cosine rewards visual sameness (e.g. leather bag + leather shoes looks too matchy). Instead:

```python
harmony = base_slot_weight * max(style_coherence, 0) - redundancy_penalty + color_bonus
```

Where:
- `base_slot_weight`: from `COMPLEMENT_RULES` — tops+bottoms get 0.9, shoes+bottoms get 0.8, accessories+tops get 0.4
- `style_coherence`: cosine similarity of both items projected into a 6-dimensional style family space (minimalist structured / relaxed oversized streetwear / luxe elevated / bohemian eclectic / sporty athletic / romantic feminine). Measures *stylistic agreement*, not visual sameness.
- `redundancy_penalty`: raw cosine similarity above 0.82 gets steeply penalized (slope of 3.0), preventing same-aesthetic-different-item situations from scoring highly
- `color_bonus`: +0.15 for classic color pairs (navy+white, camel+black, etc.), +0.075 for neutral+anything combos

### Gap Analysis

`get_gap_slots(coverage)` identifies slots with ≤1 items, ordered by outfit importance (bottoms → tops → shoes → outerwear → bags → accessories). Gap slots drive candidate generation — the system retrieves items specifically for underrepresented slots, then interleaves them in priority order.

### Negative Prototype

Dismissed/skipped items are tracked and used to build a `negative_prototype` — the L2-normalized mean of skipped item embeddings. This is used as an additional penalty in the ranker to suppress items visually similar to things the user has already passed on in this session.

---

## Signal 3: Intent — Session-Aware Browsing Inference

### The Problem

A Quiet Luxury user browsing for festival outfits should get festival recommendations, not more cashmere. Taste and intent are separate signals that most systems conflate. When a user is deep in a session looking at one type of item, the system should notice.

### The Approach

`intent.py` computes a session intent vector from recently viewed item embeddings:

1. L2-normalize each viewed item's embedding
2. Compute mean pairwise cosine similarity across all viewed items — this is the **confidence score**. High pairwise similarity = user is browsing with clear, focused intent. Low pairwise similarity = casual browsing.
3. If confidence > 0.3 (minimum threshold), compute the L2-normalized centroid of viewed embeddings as the **intent vector**
4. Label the intent by projecting the intent vector against the 20 trend archetypes and returning the top-2 above mean + 0.5 std (e.g. "Dark Academia", "Y2K Street")

At ranking time, if session intent confidence exceeds 0.3:
- `intent_weight = min(confidence, 0.55)` — capped so taste always retains at least 45% influence
- `taste_scale = 1.0 - intent_weight`
- Final score blends discovery (taste + wardrobe utility) with intent fit

In the UI, when session confidence > 0.3, a live indicator appears in the chat header — a pulsing blue dot, the inferred aesthetic labels as chips, and a confidence bar. This makes the invisible signal visible.

### Results

Intent blending adds **+13.3% P@10 lift** over taste-only ranking during focused browsing sessions. Across three simulated intent categories (black boots, linen tops, structured bags), intent-aware ranking consistently surfaces session-relevant items that taste-only misses.

---

## The Ranker — Multi-Signal Scoring Function

`rank_candidates()` in `ranker.py` is the heart of the system. Given a candidate list from FAISS, it computes a final score for each item across all active signals:

```python
# For non-cold-start:
discovery_score = (
    w_taste * taste_scale * taste_fit          # CLIP cosine to taste vector
    + session_intent_weight * intent_fit        # session browsing signal
    + w_trend * trend_boost                     # trend fingerprint alignment
    + 0.12 * style_tag_boost                    # pre-computed catalog style tags
    + w_utility * (unlock_score * 0.6 + compat * 0.4)  # outfit utility
    - w_anti * anti_taste_penalty               # anti-taste suppression
    - w_skip * skip_penalty                     # negative prototype penalty
    - attribute_mismatch_penalty                # 25-axis style preference gate
)

final = (1 - intent_bias) * discovery_score + intent_bias * query_fit
```

For cold-start (empty wardrobe), the formula simplifies — utility is 0 so the weights shift toward pure taste and trend.

After scoring, **Maximal Marginal Relevance (MMR)** reranking is applied ($\lambda = 0.7$) to balance relevance against novelty — prevents the top 10 results from all being slight variations of the same item.

### Style Tag Boost

Each catalog item is pre-tagged at index-build time with its top-3 trend affinities (CLIP cosine similarity to each trend archetype embedding). At ranking time, items tagged in the user's top trend archetypes get a boost proportional to the product of item affinity × user affinity. This is a cheaper alternative to re-projecting embeddings at query time and directly rewards catalog items that belong to the user's aesthetic subculture.

---

## Candidate Generation — FAISS + Staged Retrieval

### Discovery Mode

`generate_candidates()` does per-slot bucketed retrieval:
1. Blend the taste vector with the user's top trend embedding (25% trend, 75% taste) for style-biased retrieval
2. Run FAISS top-K search against the L2-normalized embedding index
3. Bucket results by gap slot, applying price band filter (`±tolerance` around inferred price tier)
4. Round-robin interleave slots in priority order (bottoms → tops → shoes → outerwear → bags → accessories) so underrepresented slots don't get drowned out by globally dominant categories

### Intent / Shopping Mode — 4-Stage Fallback

When the agent calls `search_catalog` with item_type and/or color filters, `search_with_filters()` runs a staged search with automatic fallback:

| Stage | Filter | Example |
|-------|--------|---------|
| 1. Exact | item_type + exact colors | "jeans" + "blue" → jeans, blue only |
| 2. Relaxed color | item_type + expanded color family | "jeans" + blue family (blue, navy, denim) |
| 3. Type only | item_type, any color | "jeans" any color |
| 4. Semantic | Pure FAISS similarity | No metadata filter |

Each stage stops if it finds ≥ top_k results. The stage used is returned to the agent as `[Search: exact match]` / `[Search: related colors]` / etc., so the LLM knows the fidelity of the results.

---

## The Orchestrator — Full Pipeline Composition

`run_wardrobe_orchestration()` composes the entire discovery feed recommendation in one call:

1. `get_wardrobe_stats()` → gap slots, slot counts, strongest slot
2. `build_wardrobe_embedding()` → recency-weighted wardrobe vector
3. `blend_vectors(taste, wardrobe, save_count)` → blended query vector
4. `build_negative_prototype(catalog, skipped_ids)` → skip penalty vector
5. `generate_candidates(blended_query, gap_slots, price_tier)` → FAISS retrieval
6. `rank_candidates(candidates, wardrobe, taste, ...)` → multi-signal ranked list + MMR
7. `build_outfit_suggestions(anchor, pool)` → greedy complementary bundles from top ranked items
8. `build_occasion_sections_unified(occasion_vectors, ...)` → per-occasion FAISS + rank pipeline
9. `build_shopping_brief(stats, price_tier, trends, occasions)` → structured purchase context

The result is a single dict consumed by the frontend to populate all feed sections simultaneously.

---

## Agentic Stylist Chat

The most technically interesting surface. The chat is a fully agentic loop powered by Claude Sonnet 4, with live tool access to the user's wardrobe, taste profile, and catalog. It reasons about all three signals simultaneously and produces mixed-content responses: streamed text + inline item cards + shoppable outfit bundles.

### System Prompt Design

The system prompt is dynamically built per request in `_build_system_prompt()`. It contains:

- **Catalog awareness**: A pre-computed inventory summary injected at top — item types with counts, available colors with counts, brands, price range. The model knows what's in stock before searching.
- **Wardrobe context**: Slot coverage (slot name + count + item names), gap slots, total item count
- **Taste profile**: Silhouette, color story, formality, price range, top 3 trends
- **Tool routing rules**: Explicit disambiguation rules for when to use each tool:
  - "Build me a work outfit" → `curate_outfit` (not multiple `search_catalog` calls)
  - "Find me a blazer" → `search_catalog`  
  - "What's in my closet" → `analyze_wardrobe`
- **Search quality rules**: When searching for work/professional pieces, use garment-level descriptions ("tailored trousers straight leg") not occasion words ("work pants") — because FashionCLIP is visual and responds to shape/fabric/formality descriptions
- **Text and card contract**: Model may only describe items the tools actually returned. No invented products.
- **Copy voice**: Short, warm, specific. Like a text from a friend with great taste. No hedging, no throat-clearing, no filler.

### Tool Definitions

Four tools, all executed synchronously via `asyncio.to_thread`:

**`search_catalog`**: Single-item lookup. Query + optional item_type, colors, slot, top_k. Routes to either `search_with_filters` (intent/shopping mode, when item_type or colors specified) or `generate_candidates` (discovery mode). When there are `last_outfit_items` in context (a previously built outfit), scores each result for compatibility against those items using `compatibility_score` and sorts accordingly — so "add a blazer to that outfit" returns blazers scored against the outfit, not the generic catalog.

**`curate_outfit`**: Full shoppable outfit from catalog. Occasion → FAISS candidates per slot (blending occasion embedding 50/50 with taste vector for retrieval) → occasion-relevance filter (CLIP cosine to occasion prompt ≥ 0.21) → greedy assembly picking the best-scoring combination across slots → swap trials to find higher-harmony alternatives. Returns harmony score, filled slots, missing slots.

**`build_outfit`**: Outfit from existing wardrobe + one catalog addition. Scores wardrobe items by occasion relevance, finds best combination across required slots, fills missing slots from catalog via ranked candidates. Returns `wardrobe_items` (owned) + `catalog_addition` (shoppable).

**`analyze_wardrobe`**: Gap analysis, slot coverage, item count, and for each item: outfit utility in human language ("many new outfit combinations" / "a few new pairings").

### Agentic Loop

```python
for turn in range(MAX_AGENT_TURNS):  # max 6 turns
    async with client.messages.stream(...) as stream:
        # Stream text tokens to client via SSE
        async for text_chunk in stream.text_stream:
            yield {"type": "text", "content": text_chunk}
        
        response = await stream.get_final_message()
    
    # Process tool calls
    for block in response.content:
        if block.type == "tool_use":
            result_text, items = await asyncio.to_thread(_execute_tool, ...)
            
            # Emit outfit bundle or individual item cards
            if block.name in ("curate_outfit", "build_outfit"):
                yield {"type": "outfit_bundle", "items": items, ...}
            else:
                for item in items:
                    yield {"type": "item_card", "item": item}
            
            # Update session context for follow-up compatibility scoring
            last_outfit_items = items[:]
            
            # Add tool result to messages and continue
    
    if not has_tool_use:
        break  # Model produced final response

yield {"type": "done"}
```

The loop is capped at 6 turns to prevent runaway tool chains. Session context (`last_outfit_items`) persists across turns so a follow-up like "add a coat to that" scores catalog results against the outfit just built.

### SSE Streaming + Mixed Content

The frontend consumes the SSE stream and builds an ordered block structure:

- `text` events → accumulated into the current text block
- `item_card` events → grouped into an `items` block (subsequent cards merge into the same strip)
- `outfit_bundle` events → become a separate `outfit_bundle` block with its own visual treatment

The result is interleaved: text → cards → more text → outfit bundle — in the order the agent produced them. This required non-trivial SSE parsing in `useStyleChat.ts` to maintain the correct interleaving as tokens stream in.

---

## Evaluation Framework

The project includes a full evaluation suite in `/scripts/`. Three scripts, one consolidated summary runner.

### Taste Model: Mean Taste Relevance @10

Methodology: for each style profile (minimalist, streetwear, elegant), at 0/5/10 saves, compare:
- **Random baseline**: mean of 50 trials of 10 random catalog items, measured by mean cosine to taste vector
- **Centroid baseline**: 10 items closest to the catalog centroid (popularity proxy)
- **Taste model**: 10 items from the full personalization pipeline

```
Saves:  0  → Random: 0.224  Centroid: 0.281  Taste: 0.452  (+102% vs random, +52% vs centroid)
Saves:  5  → ...same order of magnitude improvement maintained
Saves: 10  → ...stable improvement; wardrobe context doesn't dilute taste signal
```

### Intent: P@10 Lift

Simulated sessions: 5 items browsed from a focused query (black boots, linen tops, structured bags). Compute intent vector. Build merged candidate pool from taste + intent FAISS retrievals. Compare ranked P@10 with and without intent blending.

Result: **+13.3% mean P@10 lift** across three sessions and four wardrobe profiles.

### Purchase Confidence: Coherent vs Incoherent Ranking Accuracy

For each wardrobe profile, build two pools:
- **Coherent**: top-15 non-wardrobe items by taste cosine (items that fit the aesthetic)
- **Incoherent**: bottom-15 non-wardrobe items by taste cosine (items that clash)

Score all 30 items with the purchase confidence formula (taste fit 40% + unlock count 30% + pairing compatibility 30%). Report the fraction of (coherent, incoherent) pairs where coherent scores higher.

```
streetwear:   93.8% (vs 50% random)
smart_casual: 84.4%
minimalist:   30.2%  ← known limitation (embedding compression in neutral space)
overall:      69.5%
```

### Evaluation Summary Table

| Signal | Metric | Result |
|--------|--------|--------|
| Taste | Mean relevance @10 vs random | **+102%** at zero saves |
| Taste | Mean relevance @10 vs centroid | **+52%** at zero saves |
| Intent | P@10 lift during focused browsing | **+13.3%** over taste-only |
| Trust | Coherent > incoherent ranking accuracy | **69.5%** (vs 50% random) |

---

## System Architecture Diagram

```
User Action
    │
    ├── Paste Pinterest URL
    │       → Backend: scrape images → FashionCLIP encode → taste pipeline
    │       → Store: taste_vector, taste_modes, occasion_vectors, trend_fingerprint,
    │                anti_taste_vector, style_attributes → localStorage
    │
    ├── Save item to wardrobe
    │       → localStorage update
    │       → EMA update to taste_vector and style_attributes
    │       → Feed refresh trigger
    │
    ├── Browse catalog / view items
    │       → Session: sessionViewedItems accumulates embeddings
    │       → intent.py: compute confidence + intent_vector
    │       → UI: session indicator in chat header (>0.3 confidence)
    │
    ├── Discovery Feed load
    │       → POST /api/feed → run_wardrobe_orchestration()
    │           ├── FAISS retrieval (blended taste+wardrobe+trend query)
    │           ├── rank_candidates (taste + trend + utility + anti-taste + intent)
    │           ├── MMR reranking
    │           ├── Occasion sections (per-occasion FAISS + rank)
    │           ├── Outfit bundles (greedy complementary assembly)
    │           └── Shopping brief (gap slots, price tier, dominant occasions)
    │
    ├── Tap item → Product Detail / Evaluation Panel
    │       → POST /api/recommendations/evaluate-item-v2
    │           ├── Taste fit (percentile-normalized vs catalog)
    │           ├── Intent match (if session confidence > 0.3)
    │           ├── Purchase confidence (HIGH/MEDIUM/LOW)
    │           ├── Outfit unlock count
    │           └── Pairing suggestions from wardrobe
    │
    └── Open Chat → "Ask Phia"
            → POST /api/chat/stream → run_stylist_chat()
                ├── Build system prompt (catalog inventory + wardrobe + taste)
                ├── Agentic loop (max 6 turns)
                │   ├── Claude streams text tokens → SSE "text" events
                │   ├── Claude calls tool → execute_tool()
                │   │   ├── search_catalog → staged FAISS + rank
                │   │   ├── curate_outfit → per-slot FAISS + greedy assembly
                │   │   ├── build_outfit → wardrobe + catalog gap fill
                │   │   └── analyze_wardrobe → gap analysis + utility labels
                │   └── Tool results → item_card / outfit_bundle SSE events
                └── SSE stream consumed by useStyleChat.ts → ordered blocks
```

---

## Frontend Architecture

The frontend is a Next.js 15 app with a mobile-first design (430px max-width container, bottom navigation pattern).

### State Management

A single `useAppState()` hook manages all application state. Persistent state (taste profile, wardrobe, skipped items) lives in localStorage with safe serialization guards. Session state (viewed items, intent vector, intent confidence) is React state — ephemeral, resets on reload.

The taste profile TypeScript interface reflects the full backend output:

```typescript
interface TasteProfile {
  taste_vector: number[];
  taste_modes?: number[][];
  occasion_vectors?: Record<string, number[]>;
  trend_fingerprint?: Record<string, number>;
  anti_taste_vector?: number[];
  style_attributes?: Record<string, number>;
  aesthetic_attributes: Record<string, { label: string; confidence: number }>;
  price_tier: [number, number];
}
```

### Discovery Feed Sections

- **Complete your closet**: Gap-fill items with unlock counts, confidence gates (≥70% taste score), each card shows "+N outfits with your saves"
- **Your aesthetic**: Pure taste vector ANN results, no wardrobe filtering, works from cold start
- **Complete your outfits**: Pre-assembled outfit bundles from wardrobe + one shoppable addition each
- **Best prices on your saves**: Phia integration — price signals on saved items
- **Occasion rows**: Per-occasion FAISS sections ("Your work style", "Your everyday", "Your going-out")

### Chat UI

The `ChatDrawer` component is a full-screen animated sheet (spring-physics enter/exit via Framer Motion). It handles:
- Streamed text with a blinking cursor on the active token
- Inline item strips (horizontal scroll, 120px cards, deduplicated)
- Outfit bundles (bordered card with item grid)
- Session intent indicator (pulsing dot, trend label chips, confidence bar)
- Suggested prompts pre-populated from wardrobe state
- Preload item support: opening chat from "Ask Phia" on a product card pre-sends a context-aware query about that item

---

## Development Challenges and How We Solved Them

This section summarizes the key engineering challenges encountered during development, roughly in order.

### Challenge 1: Cold Start — Making Personalization Work at Zero Interactions

**Problem**: The first time a user opens the feed, there are no saves, no purchase history, no click events. Engagement-based recommendation systems degrade to popularity rankings at cold start.

**Solution**: Extract a 512-d visual taste vector from a Pinterest board before any interaction. FashionCLIP encodes aesthetic preferences directly from outfit inspiration images. The taste vector is meaningful from the first inference — no purchase or click history required. The +102% mean relevance lift vs random at zero saves proves this works.

### Challenge 2: Style-Family Coherence in Outfit Assembly

**Problem**: Raw cosine similarity between item embeddings rewards visual sameness — two items that look alike (e.g. black leather bag + black leather shoes) score high, but real stylists know that too much matching looks intentional in the wrong way. Early outfit scoring was returning overly matchy bundles.

**Solution**: Replaced raw cosine compatibility with a three-part score: **style-space coherence** (project both items into 6-dimensional style family space and measure agreement there), minus a **visual redundancy penalty** (steep penalty when raw cosine similarity > 0.82 — items that look too alike), plus a **color harmony bonus** (explicit color pair rules: navy+white, camel+black, etc.). This correctly rewards items that are stylistically aligned but visually distinct.

### Challenge 3: Outfit Unlock Count Showing the Same Value for Everything

**Problem**: Initially, `outfit_unlock_count` was returning 36 for every item in the feed. The UI was showing "+36 outfits" universally, which was clearly wrong and useless.

**Root cause**: The ranker was calling `outfit_unlock_count` with an empty wardrobe — it was computing theoretical maximum combinations against a zero-item wardrobe, so all items got the same number (0 compatible items = all combinations are vacuously possible under the old implementation).

**Solution**: Fixed the wardrobe pass-through in the ranker and updated `outfit_unlock_count` to return 0 when the compatibility check returns no matches (not vacuously count empty-set combinations). Added per-item unlock display only when the wardrobe has items; cold start suppresses the unlock count badge.

### Challenge 4: Chat Returning Wrong Item Types

**Problem**: When users asked for "a short dress", the AI's `search_catalog("short dress")` call was returning tops, separates, and other non-dress items. The search was ignoring the item type entirely and returning whatever was semantically adjacent to the taste vector.

**Root cause**: `generate_candidates` blends the query with the taste vector and does gap-slot-based bucketing. For a minimalist user asking for a dress, the taste vector was pulling toward structured separates (their typical aesthetic), and without hard item type filtering, "dress" in the query wasn't enough to override the taste signal.

**Solution**: Added `search_with_filters()` with 4-stage fallback (exact item_type + color → relaxed color family → type only → semantic). Updated `search_catalog` tool to route to this path when `item_type` or `colors` are specified. Updated the system prompt to tell the model to always specify `item_type` for intent queries ("jeans", "blazer", "dress"). Added the staged search result label to tool output so the model knows what was matched vs relaxed.

### Challenge 5: Chat Cards Appearing at the Bottom Instead of Inline

**Problem**: The agent would produce text, then call a tool, then produce more text — but all item cards were appearing at the bottom of the message after all text, instead of inline at the point where the tool was called.

**Root cause**: The frontend was accumulating all items into a single `items` array on the message object, so the rendering order was always: text first, then all items, then more text.

**Solution**: Replaced the flat `items` array with an ordered `blocks` array where each block has a type (`text`, `items`, `outfit_bundle`) and is emitted in the order the SSE events arrive. Text segments get merged into the current text block; `item_card` events create or extend an `items` block at that position; `outfit_bundle` events create a dedicated bundle block. The result is correct interleaving: text → cards → text → outfit bundle, exactly matching the agent's intent.

### Challenge 6: FashionCLIP Text vs Image Embedding Modality Gap

**Problem**: The catalog was initially built with text embeddings (CLIP text encoder applied to product titles). Taste vectors were built from images. These two embedding subspaces don't align cleanly — text embeddings for "navy blazer" and image embeddings of a navy blazer live in different regions of the 512-d space.

**Solution**: Rebuilt the catalog using `encode_images()` instead of `encode_texts()` for all catalog items. Used the `ashraq/fashion-product-images-small` HuggingFace dataset which includes real product images. This closes the modality gap entirely — both taste vectors and catalog embeddings are now in the image embedding subspace, making cosine similarity between them meaningful.

### Challenge 7: Minimalist Aesthetics Scoring Poorly

**Problem**: Purchase confidence accuracy was 30% for minimalist profiles vs 90%+ for streetwear. Users with clean/neutral/structured taste were getting poor confidence predictions.

**Root cause**: FashionCLIP's embedding space has higher density in the minimalist/neutral/structured region — items that a human would distinguish as "very different" (a cream blazer vs an ivory silk blouse) have high cosine similarity. The coherent-vs-incoherent cosine spread for minimalist profiles is small, making it hard for the purchase confidence model to differentiate.

**Status**: Documented as a known limitation. The technical brief is explicit about this. The solution in production would be a larger catalog with finer-grained labels in the neutral aesthetic space, or contrastive fine-tuning of the embedding model on hard negative pairs within the minimalist cluster.

---

## What Makes This Technically Impressive

1. **Three genuinely different signals with clean mathematical separation**: Taste (persistent CLIP-derived aesthetic identity), Wardrobe (combinatorial outfit utility via compatibility scoring), and Intent (ephemeral session coherence signal) — each has a principled mathematical formulation and each is separately measurable.

2. **Fashion-domain embedding model**: FashionCLIP fine-tuned on 800K Farfetch image-text pairs understands silhouette, fabric, and construction details that generic CLIP misses. This is not a generic ML project — it uses the right embedding model for the domain.

3. **25-axis fine-grained style preference system**: Beyond coarse "aesthetic label" classification, a 25-dimensional signed preference vector (positive pole vs negative pole per axis) that drives ranking penalties. An axis with |score| < 0.3 is ignored — no penalty from ambiguous signals.

4. **Occasion-decomposed taste**: Per-occasion taste vectors mean the system knows a user's "work" vs "weekend" vs "evening" aesthetic separately. A user who dresses conservatively at work and eclectically on weekends gets appropriate recommendations for each context.

5. **Compatibility scoring that prevents over-matching**: The redundancy penalty (raw cosine > 0.82) combined with style-family projection prevents the classic "all-black everything" homogeneity that plagues naive cosine-based outfit assembly.

6. **Agentic stylist with real tool architecture**: Not a chatbot with hardcoded responses. A multi-turn agentic loop where Claude decides which tools to call, in what order, with what parameters — and the tool results (real catalog items) flow back as grounded recommendations. The model cannot hallucinate products because it can only describe items the tools returned.

7. **SSE streaming with typed mixed-content blocks**: The streaming layer distinguishes text tokens, item cards, and outfit bundles as separate SSE event types and interleaves them in the correct sequence. The result is a real-time chat experience where text streams in, then an item strip appears, then text continues — matching the agent's actual reasoning order.

8. **Evaluation framework with three independently verifiable metrics**: Not just vibes. Three evaluation scripts, each with a different methodology, each measuring a different aspect of the system. The metrics map directly to Phia's stated product goals (cold-start personalization, intent disambiguation, return rate reduction).

---

## Results Summary

| Signal | Metric | Result |
|--------|--------|--------|
| Taste (cold start) | Mean relevance @10 vs random | **+102%** |
| Taste (cold start) | Mean relevance @10 vs popularity baseline | **+52%** |
| Intent | P@10 lift during focused browsing sessions | **+13.3%** |
| Wardrobe | Purchase confidence coherent-vs-incoherent accuracy | **69.5%** (vs 50% random) |
| System | Catalog items indexed | **2,364** |
| System | Style axes in preference profile | **25** |
| System | Named trend archetypes | **20** |
| System | Max agentic chat turns | **6** |
| System | Taste extraction time from Pinterest | **~3–5 seconds** |

---

## What This Is Actually Demonstrating

Three capabilities that map directly onto Phia's Series A roadmap:

**Cold-start personalization** that produces meaningful recommendations before any purchase history exists — via visual aesthetic extraction from Pinterest boards using a fashion-domain embedding model.

**Real-time intent inference** that separates "who I am" from "what I need right now" — via session browsing coherence signals that modulate ranking weights without overriding long-term taste.

**Purchase quality prediction** as a return-rate proxy — via a wardrobe-integrated confidence score that tells users not just whether they'll like something, but whether it will actually work with what they own.

The system is live, interactive, and produces measurable lift on all three axes with a demo-scale catalog.
