# Wardrobe IQ — Technical Brief

**Three personalization signals for purchase intelligence**
Veedhi Bhanushali · April 2026

---

## What Phia Is Building

Phia's Series A is funding three capabilities: understanding what a shopper *likes*, what they *need right now*, and whether a purchase will *stick*. Today, Phia answers "is this the best price?" — but the user's real question is "should I buy this, given who I am and what I already own?" This project is a working proof of concept for that personalization layer.

---

## What This Project Demonstrates

### 1. Taste: Cold-Start Personalization via Visual Extraction

**The problem.** Every new user sees the same feed. Engagement-based personalization requires purchase history that doesn't exist yet — Phia's stated cold-start problem.

**The approach.** We extract a 512-D taste vector from a user's Pinterest board or uploaded outfit images using FashionCLIP, a fashion-domain CLIP model. This vector encodes aesthetic preferences (silhouette, color palette, formality) without requiring any purchase or click history. Recommendations are retrieved via FAISS similarity search against the full catalog, then re-ranked using a multi-signal scoring function that incorporates taste fit, wardrobe utility (outfit unlock count), and trend alignment.

**The result.** The taste model achieves **+102% mean relevance above random** and **+52% above a catalog centroid baseline** at zero saves. Personalization is meaningful before any interaction history exists. At 5 and 10 saves, the system maintains +90% and +91% lift over random respectively, demonstrating that the taste signal remains stable as wardrobe context is introduced.

**Known limitation.** FashionCLIP's embedding space compresses similarity scores for minimalist aesthetics — items within the "neutral/clean/structured" region have narrower cosine distance between matches and non-matches. A production system would address this with a larger, better-labeled catalog and fine-tuned embeddings for under-differentiated style regions.

---

### 2. Intent: Session-Aware Signal Separation

**The problem.** A user with a Quiet Luxury profile browsing for festival outfits should get festival recommendations, not more cashmere. Taste and intent are separate signals, but most systems conflate them.

**The approach.** We maintain a session intent vector computed as the L2-normalized centroid of recently-viewed item embeddings, with a coherence-based confidence score (mean pairwise cosine similarity of viewed items). When confidence exceeds 0.3, the intent vector is blended into the ranking function with weight proportional to confidence, capped at 0.55. This allows the system to respond to focused browsing sessions without overriding long-term taste. The intent signal is surfaced live in the UI via a session indicator in the chat header, showing the inferred browsing aesthetic and confidence level.

**The result.** Intent blending adds **+13.3% P@10 lift** over taste-only ranking during focused browsing sessions. Across three simulated intent categories (black boots, linen tops, structured bags), the intent-aware model consistently surfaces session-relevant items that the taste-only model misses — because those items are pulled from a merged candidate pool that includes both taste-aligned and intent-aligned retrievals.

---

### 3. Trust: Purchase Confidence as Return Proxy

**The problem.** Phia's brand metrics target a 50% return rate reduction. Returns happen when purchased items don't integrate with the buyer's existing wardrobe — wrong aesthetic, redundant slot, poor pairing compatibility.

**The approach.** We compute a purchase confidence score for each item as a weighted combination of taste fit (cosine to user taste vector, 40%), outfit unlock count (new outfit combinations enabled by the item, 30%), and wardrobe pairing compatibility (number of existing items it pairs well with, 30%). Items are classified as HIGH / MEDIUM / LOW confidence on the product evaluation panel. High-confidence items integrate with the wardrobe along all three axes; low-confidence items don't.

**The result.** The purchase confidence model **correctly ranks stylistically coherent additions above incoherent ones in 69.5% of pairs** (vs 50% random). On the streetwear and smart casual test wardrobes, accuracy reaches 93.8% and 84.4% respectively. The minimalist profile scores 30.2% due to compressed similarity in FashionCLIP's embedding space for neutral aesthetics — a known limitation that would be addressed with domain-specific fine-tuning or a larger catalog with finer-grained labels.

---

## Summary

| Signal | Metric | Result |
|--------|--------|--------|
| Taste | Mean relevance @ 10 vs random baseline | **+102%** at zero saves |
| Taste | Mean relevance @ 10 vs centroid baseline | **+52%** at zero saves |
| Intent | P@10 lift during focused browsing | **+13.3%** over taste-only |
| Trust | Coherent vs incoherent ranking accuracy | **69.5%** (vs 50% random) |

Three signals, three direct mappings to Phia's roadmap: cold-start personalization, real-time intent inference, and purchase quality prediction. The system is live, interactive, and produces measurable lift on all three axes with a demo-scale catalog of 2,364 items.
