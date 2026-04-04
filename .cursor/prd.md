# Wardrobe IQ — Product Requirements Document

**Product:** Wardrobes (Phia Extension)
**Author:** Veeya Bhanushali
**Date:** April 2026
**Status:** Draft v1

-----

## What This Is

A taste-aware outfit utility layer built on top of Phia’s existing saves and collections surface. It answers one question: **given what you’ve saved on Phia and your visual taste profile, what’s the single next piece that makes your existing saves most wearable?**

This is not a new product. It extends Phia’s existing “Should I Buy This?” moment and Wardrobe/Collections tab with personalized outfit intelligence.

-----

## The Problem

Phia today answers: *is this the best price for this item?*

What it doesn’t answer:

- Does this item fit my aesthetic?
- Does this work with what I’ve already saved?
- What single piece would make my existing saves most useful?

Users buy things that don’t work together. They return them. Phia’s brand partners absorb the cost. A system that scores purchases by outfit utility — how much they increase the wearability of existing saves — directly reduces return rates, which is Phia’s most commercially important brand partner claim.

-----

## Goals

**User goal:** Know whether a new item actually works with what you already have, and surface the one piece that would make your current saves most wearable.

**Business goal:** Deepen Phia’s personalization layer (their stated Series A investment area) and improve purchase quality (reducing returns, increasing basket value) without adding friction to the existing flow.

**Demo goal:** A working web app that shows the augmented “Should I Buy This?” panel and a Wardrobe utility view, with real taste extraction and real outfit scoring behind it.

-----

## Non-Goals

- Physical closet cataloging
- Redundancy flagging or “don’t buy this” messaging
- Social features or creator-to-cart
- Browser extension (web demo only)
- Freeform LLM-generated explanations
- Trend forecasting pipeline

-----

## Users

Fashion-conscious Phia users who save items regularly and want help building outfits, not just finding prices. 18–35, aesthetically driven, already using Pinterest or similar for inspiration.

-----

## Core Concept: Wardrobe = Phia Saves

The working wardrobe is the user’s existing Phia saves. No manual cataloging. Signal accumulates passively. Every save improves recommendation quality.

-----

## Taste Onboarding (Cold Start)

User pastes a Pinterest board URL or uploads up to 10 reference images. System extracts an aesthetic profile. No quiz. Takes under 30 seconds. This produces meaningful personalization from zero behavioral history.

-----

## Functional Requirements

### F1 — Taste Onboarding

- Accept Pinterest board URL or image upload (up to 10 images)
- Extract and display aesthetic profile: dominant color story, silhouette family, formality level, occasion range
- Profile updates incrementally with each new save
- Must work with 0 prior saves

### F2 — Wardrobe View (Collections Tab Enhancement)

- Display saved items with outfit utility scores
- Surface the single highest-utility gap recommendation: “Adding this one piece unlocks N new outfit combinations with your saves”
- “Complete the Look” for any selected saved item: show which other saves pair with it + the recommended addition
- Filter/sort by outfit utility score

### F3 — Augmented “Should I Buy This?” Panel

- When evaluating any product, show alongside existing price panel:
  - Taste fit score (% match to aesthetic profile)
  - Outfit unlock count (how many new combinations with existing saves)
  - Which specific saves it pairs with
  - Short explanation chip (template-based)
- Phia’s price comparison is unchanged

### F4 — Recommendation Engine

- Candidate generation constrained to underrepresented outfit slots in saves
- Candidates filtered to inferred price tier (±30%)
- Ranked by: taste compatibility + outfit unlock count + value score
- Module suppressed if top candidate confidence is below threshold
- Never show weak recommendations over nothing

### F5 — Explanation Layer

- Template-based only. No freeform LLM output.
- Examples:
  - “Pairs with your saved linen trouser — unlocks 7 new looks”
  - “Matches your neutral minimal aesthetic”
  - “Fills an outerwear gap in your saves”

### F6 — Event Logging

- Log: onboarding completion, taste profile generation, recommendation impression, click, save-from-recommendation
- Required for offline evaluation

-----

## UX Flow

```
Onboarding
  → Paste Pinterest URL or upload images
  → Aesthetic profile renders (color, silhouette, formality, occasion)
  → Connect to Phia saves (or simulate wardrobe for demo)

Wardrobe View
  → Grid of saves with utility scores
  → Gap recommendation surfaced at top
  → Tap any item → "Complete the Look" panel

Product Evaluation
  → Paste product URL
  → Existing: price comparison
  → New: taste fit + outfit unlocks + explanation
```

-----

## MVP Success Criteria

1. Taste extraction produces visually coherent aesthetic profile from Pinterest import
1. Outfit unlock count produces non-random, plausible recommendations across different wardrobe compositions
1. Augmented panel renders cleanly alongside simulated price data
1. Cold start (0 saves) produces usable recommendations from taste profile alone
1. Evaluation metrics show system outperforms popularity baseline on preference prediction

-----

## Metrics

**Primary**

- Taste fit prediction accuracy (held-out save prediction vs popularity baseline)
- Outfit utility calibration (correlation between predicted unlock count and actual co-save rate)
- Cold start AUC-ROC at 0 / 3 / 5 saves

**Secondary**

- Recommendation click rate in demo sessions
- Onboarding completion rate
- Module suppression rate (should be under 25%)

-----

## Out of Scope for MVP

Everything not listed in F1–F6. Ship the core loop cleanly before adding anything.