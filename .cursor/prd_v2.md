# Wardrobe IQ — Product Requirements Document v2
**Product:** Phia × Wardrobe IQ  
**Author:** Veeya Bhanushali  
**Date:** April 2026  
**Status:** v2 — Authoritative

---

## What This Is

A taste-aware, wardrobe-intelligent shopping layer that sits on top of Phia's catalog and price infrastructure. It answers three questions simultaneously: *Does this match who I am aesthetically? Does this fill a real gap in what I already own? Is this what I'm actually looking for right now?*

The product has four distinct surfaces, each mapping to one of three personalization signals:

| Surface | Signal | Persistence |
|---|---|---|
| Wardrobe Builder | Wardrobe — what you own | Persistent |
| Aesthetic Builder | Taste — who you are | Persistent |
| Discovery Feed | All three signals combined | Persistent + session |
| Stylist Chat | Intent — what you need now | Ephemeral (per session) |

This is not a wardrobe organizer. It is a purchase intelligence system. The wardrobe and aesthetic surfaces exist to make the feed and chat dramatically more useful than any generic shopping experience.

---

## The Problem

Phia today answers: *is this the best price for this item?*

That is a necessary but insufficient answer. The user's real question is: *should I buy this, given who I am and what I already own?*

Three gaps:

1. **No taste layer.** Phia doesn't know your aesthetic. Every user sees the same ranked feed.
2. **No wardrobe context.** Phia doesn't know what you already own. Recommendations can duplicate or clash with your existing items.
3. **No intent disambiguation.** Phia can't distinguish between your long-term style preferences and what you actually need right now. Someone who dresses minimally might be shopping for a ski trip — the system has no way to know.

Phia's Series A is explicitly funding solutions to all three. This product is the proof of concept for that roadmap.

---

## Goals

**User goal:** Build a virtual wardrobe, define your aesthetic, and get a feed that is genuinely personalized — not just "popular items in your size." Ask the chat for anything specific and get a real stylist-quality answer with actual shoppable items.

**Business goal:** Demonstrate that intent + taste + wardrobe context produces meaningfully better purchase quality than taste-only or intent-only approaches. Specifically: higher outfit utility per purchase, lower predicted return likelihood.

**Demo/internship goal:** Show Phia a working proof of concept for the personalization layer their $35M Series A is building. Show ML rigor (three-signal model, evaluation framework), product intuition (clean UX, right interaction model), and agentic AI implementation (tool-using chat that reasons about the wardrobe).

---

## Non-Goals

- Physical closet import (photos of existing clothes)
- Social features, creator-to-cart, community styling
- Real Phia backend integration (demo uses simulated catalog and saves)
- Native mobile app (web demo only)
- Trend forecasting pipeline
- Any feature not in F1–F6 below

---

## Users

Fashion-conscious Phia users, 18–35, who save items regularly and want help building a coherent wardrobe rather than collecting random pieces. Already use Pinterest for inspiration. Comfortable with AI-assisted shopping.

---

## The Three-Signal Model

Every recommendation is the intersection of three signals:

```
Taste (persistent)    — aesthetic identity, built from Pinterest + save history
Wardrobe (persistent) — what you own, updated with every save
Intent (ephemeral)    — what you need right now, expressed via chat or session behavior
```

**Taste** is who you are. Silhouette preference, color story, formality range, price tier, trend alignment. Built once at onboarding, refined incrementally. Never resets.

**Wardrobe** is what you have. Every saved item. Enables gap analysis, outfit completion, and deduplication. Without it, recommendations are taste-filtered search. With it, recommendations become genuinely useful — the system can tell you which single piece unlocks the most outfit combinations you currently can't build.

**Intent** is what you want right now. Expressed explicitly (chat message) or inferred (session browsing patterns — 5 similar items viewed = high intent signal for that category). Resets per session.

The ranker blends all three. Cold sessions with weak intent fall back on taste + wardrobe. Strong intent (explicit chat request or dense session browsing) gets weighted more heavily.

---

## Functional Requirements

### F1 — Wardrobe Builder Card

A dedicated card-surface where users build their virtual closet from Phia's catalog.

- Grid of catalog items browsable by category (All / Tops / Bottoms / Outerwear / Shoes / Bags / Accessories)
- Search bar for specific items
- Tap any item → item detail with "Add to Wardrobe" action
- Saved items appear in the wardrobe grid with an outfit utility score badge
- Each saved item shows: which other saves it pairs with, how many outfit combinations it enables
- Visual slot coverage indicator: which categories are strong vs. thin
- Wardrobe is stored in localStorage; synced to backend for recommendation computation
- Empty state is purposeful: shows the six outfit slots with prompts to fill them

### F2 — Aesthetic Builder Card

A dedicated card-surface where users define their taste via Pinterest.

- Single input: paste a Pinterest board URL
- Optional: upload up to 10 reference images for users without Pinterest
- On submit: show processing state (real CLIP embedding computation, ~3–5 seconds)
- Output: aesthetic profile rendered as:
  - Display label (inferred aesthetic name, e.g. "Quiet Luxury", "Coastal Grandmother", "Clean Streetwear")
  - Five dominant color swatches extracted from board images
  - Attribute grid: Silhouette, Color Story, Formality, Occasion
  - Trend alignment tags (top 3 named aesthetics from the trend lexicon)
  - Inferred price range
  - Confidence indicator (based on number of images processed)
- Profile can be updated at any time by submitting a new board
- If no Pinterest: profile can be built purely from wardrobe saves (requires 5+ saves)

### F3 — Discovery Feed

The main surface. Below the two builder cards. Three or more horizontally-scrolling sections, each powered by a different combination of signals.

**Section: "Complete your closet"**  
Powered by: Wardrobe gap analysis + taste filtering  
Shows: The highest-utility items per underrepresented slot. Not generic "popular bottoms" — specific items that integrate with your existing saves. Each card shows unlock count: "+6 outfits with your saves."  
Suppressed if: Wardrobe is empty (replaced with "Start saving to see personalized gaps")

**Section: "Your aesthetic"**  
Powered by: Taste vector ANN retrieval  
Shows: Items that score highest on taste compatibility. No wardrobe filtering — pure aesthetic match. Works from cold start (0 saves).  
Each card shows taste fit percentage.

**Section: "Complete your outfits"**  
Powered by: Wardrobe + taste, outfit assembly logic  
Shows: Pre-built outfit bundles from existing saves + one recommended addition each. Each bundle is a horizontal strip of 3–4 items (wardrobe items in standard styling, recommended addition visually distinguished).  
Suppressed if: Fewer than 3 wardrobe saves

**Section: "Best prices on your saves"**  
Powered by: Wardrobe + Phia price intelligence  
Shows: Saved items that currently have strong resale prices available. Highlights price drops or unusually low resale prices.  
This is the Phia integration section — price comparison is their core feature, shown here in the context of items the user actually cares about.

**Section: "For your [inferred occasions]"**  
Powered by: Taste occasion vectors  
Shows: One horizontal row per inferred occasion (Work, Weekend, Evening). Items scored for that occasion's context, filtered by taste.  
Suppressed if: Taste profile not yet built

### F4 — Stylist Chat (Intent Surface)

A full-screen chat experience. The AI stylist has live tool access to the user's wardrobe, taste profile, and catalog. It reasons about all three signals simultaneously.

**Chat behaviors:**
- Responds to natural language requests about outfit building, gap filling, occasion-specific shopping, item evaluation
- Calls tools to find real catalog items — never invents products
- Returns mixed content: text reasoning + inline item cards + outfit bundles
- Item cards in chat are shoppable: Save to Wardrobe directly from chat
- Understands wardrobe context: will not recommend items the user already has
- Respects anti-taste (inferred dislikes from low-scoring trend clusters)

**Opening state:**
- Context summary bar: "4 wardrobe items · Quiet Luxury profile · loaded"
- Three suggested prompts as tappable chips (contextually generated from wardrobe state)
- Empty chat otherwise — no artificial assistant intro message

**Suggested prompt examples (dynamic, based on wardrobe state):**
- If outerwear gap: "Find me a coat that works with my saves"
- If upcoming weekend: "Build me a weekend outfit from what I have"
- Generic fallbacks: "What's my biggest wardrobe gap?", "Help me plan outfits for the week"

**Tools available to the agent:**
- `search_catalog(query, slot, occasion, max_results)` — semantic search + taste filtering
- `analyze_wardrobe()` — gap analysis, utility scoring, coverage summary
- `build_outfit(occasion, anchor_item_id, constraints)` — greedy outfit assembly from wardrobe + one catalog addition
- `score_item(item_id)` — taste fit + outfit unlock count + intent match for a specific item

**Inline item card (chat):**
- Product image thumbnail
- Brand, title, price (new) + best resale price
- Taste fit % badge
- One-line reason: why this specific item for this specific wardrobe
- Save to Wardrobe button
- Tap to open full evaluation panel

**Inline outfit bundle (chat):**
- Horizontal scroll of 3–4 items
- Wardrobe items shown normally; recommended addition has a distinct visual treatment (bordered, labeled "Recommended addition")
- Occasion label
- "Shop the addition" CTA

### F5 — Evaluation Panel ("Should I Buy This?")

Triggered when tapping any item in the feed, a chat item card, or from the wardrobe builder.

Three score cards:
1. **Taste Fit** — visual style alignment with aesthetic profile. 0–100%. Green ≥80%, amber 60–79%, red <60%.
2. **Purchase Confidence** — wardrobe integration score. HIGH / MEDIUM / LOW. Derived from outfit unlock count + compatibility with existing saves. This is the return-rate proxy metric.
3. **Intent Match** — session context alignment. 0–100%. Only shown if session intent signal is present (>0.3 confidence). Hidden otherwise to avoid misleading cold-session scores.

Below scores:
- Best price bar: retail vs. best resale price, source, savings percentage
- "Pairs with your saves" row: thumbnails of specific wardrobe items it works with
- AI insight: one sentence specific to this item + this wardrobe. Not a template — generated by the ranker's top signal.
- Two CTAs: "Ask Phia about this" (opens chat with item pre-loaded) + "Save to wardrobe"

### F6 — Event Logging

Log every meaningful interaction for offline evaluation:
- Onboarding completion (source: pinterest / upload)
- Taste profile generation (confidence, cluster count)
- Feed section impression + item impression
- Item tap → evaluation panel open
- Save to wardrobe (from feed / panel / chat)
- Chat message sent
- Chat item card save
- Dismiss / skip

Required for the three evaluation metrics.

---

## UX Flow

```
Landing
  → Two cards side by side (desktop) or stacked (mobile):
    [Wardrobe Builder]   [Aesthetic Builder]
  
  Wardrobe Builder
    → Browse catalog by slot
    → Tap item → detail → "Add to Wardrobe"
    → Item appears in wardrobe grid with utility score
  
  Aesthetic Builder  
    → Paste Pinterest URL → "Extract my taste →"
    → Processing state (real computation)
    → Aesthetic profile renders (label, swatches, attributes, trends)
  
  Discovery Feed (below cards)
    → "Complete your closet" section
    → "Your aesthetic" section
    → "Complete your outfits" section (requires 3+ saves)
    → "Best prices on your saves" section
    → "For your [occasion]" rows
  
  Item tap → Evaluation Panel
    → Taste Fit / Purchase Confidence / Intent Match scores
    → Price comparison
    → Pairs with saves
    → AI insight
    → Save / Ask Phia
  
  Chat (full-screen, opened from nav or "Ask Phia" CTA)
    → Context bar: wardrobe + taste loaded
    → Suggested prompts
    → Conversational exchange
    → Item cards + outfit bundles inline
    → Save to wardrobe from chat
```

---

## MVP Success Criteria

1. Aesthetic profile extraction produces visually coherent results from a real Pinterest board
2. "Complete your closet" section produces non-generic, non-redundant recommendations that demonstrably fill wardrobe gaps
3. Purchase Confidence score is meaningfully higher for items that integrate well with existing saves vs. items that don't
4. Chat agent answers a specific occasion request ("dinner Friday + brunch Saturday") with a real two-occasion solution in under 10 seconds
5. Evaluation scripts show taste model outperforms popularity baseline on Precision@10
6. Evaluation scripts show Purchase Confidence score predicts simulated "return" events better than random

---

## Evaluation Metrics

**Primary**
- Taste fit Precision@10 vs. popularity baseline (at 0, 3, 5, 10 saves)
- Purchase Confidence correlation with simulated return rate (Spearman ρ)
- Cold start AUC-ROC at 0 / 3 / 5 saves

**Secondary**
- Feed section click rate per section type (which sections drive engagement)
- Chat session length (message turns)
- Save rate from chat item cards vs. feed cards
- Module suppression rate (should be <25% — high suppression means weak recommendations)

---

## Out of Scope for MVP

Physical closet scanning, social features, creator-to-cart, browser extension, freeform styling advice without catalog grounding, real Phia backend integration.
