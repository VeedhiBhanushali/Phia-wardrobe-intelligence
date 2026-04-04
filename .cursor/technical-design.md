# Wardrobe IQ — Technical Design

**Version:** v1
**Purpose:** Implementation direction for Claude Code. Prioritizes low cost, fast iteration, strong outcomes.

-----

## Guiding Constraints

- No Phia database access. Use public product data.
- No expensive model training. Use pretrained models + lightweight fine-tuning only.
- No freeform LLM in recommendation path. Templates only.
- Minimize API costs. Batch where possible, cache aggressively.
- Every component independently testable.

-----

## Stack

|Layer                  |Choice                              |Why                                                                                                                |
|-----------------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------|
|Embedding              |CLIP ViT-B/32 (OpenAI)              |Free, fast, fashion-capable. ViT-L is better but 3x slower — not needed for MVP.                                   |
|Vector search          |FAISS (local)                       |Free, no infra, good enough at demo scale. Migrate to Pinecone if scaling.                                         |
|Clustering             |UMAP + HDBSCAN (scikit-learn)       |Standard, well-documented, runs CPU-only.                                                                          |
|Backend                |FastAPI                             |Lightweight, async, easy to structure.                                                                             |
|Frontend               |Next.js + Tailwind                  |Fast to build clean UI.                                                                                            |
|Product data           |SerpApi (Google Shopping) or Oxylabs|Cheap at demo scale. Cache all responses.                                                                          |
|DB                     |Supabase (free tier)                |Saves, sessions, taste profiles, impression logs.                                                                  |
|LLM (explanations only)|Claude Haiku                        |Cheapest per token. Used only for template selection, not generation. Or skip entirely and use hardcoded templates.|

**Do not use GPT-4 or Claude Sonnet in the hot path. All recommendation scoring is local compute.**

-----

## System Components

```
[Input Layer]
  Pinterest Scraper / Image Upload
        ↓
[Taste Extraction Pipeline]
  CLIP Encoder → Embedding Pool → UMAP+HDBSCAN → Aesthetic Profile
        ↓
[Wardrobe State]
  Phia Saves (simulated) → Wardrobe Embedding → Slot Coverage Map
        ↓
[Candidate Generation]
  Gap Analysis → Taste-Filtered ANN Retrieval → Price Band Filter
        ↓
[Outfit Utility Ranker]
  Category-Conditioned Compatibility → Outfit Unlock Count → Final Score
        ↓
[Serving Layer]
  FastAPI Orchestrator → Unified Payload → Next.js Demo UI
        ↓
[Logging]
  Supabase → Offline Evaluation Jobs
```

-----

## Component 1: Taste Extraction Pipeline

**Input:** Pinterest board URL or list of image URLs/files
**Output:** `taste_vector` (512-dim), `aesthetic_attributes` (JSON)

### Pinterest Scraper

- Use `pinterest-api` npm package or direct HTML scrape of public board
- Extract image URLs from board (cap at 50 images for cost)
- Download images, resize to 224×224, normalize

### CLIP Encoding

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32", device="cpu")

def encode_images(image_paths: list[str]) -> np.ndarray:
    images = [preprocess(Image.open(p)) for p in image_paths]
    batch = torch.stack(images)
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    return embeddings.numpy()  # shape: (N, 512)
```

### Pooling

```python
def build_taste_vector(embeddings: np.ndarray, weights=None) -> np.ndarray:
    # Recency-weighted mean. Pinterest images get weight 0.6, saves get 1.0
    if weights is None:
        weights = np.ones(len(embeddings))
    weights = weights / weights.sum()
    return (embeddings * weights[:, None]).sum(axis=0)  # (512,)
```

### Attribute Extraction

```python
# After UMAP reduction + HDBSCAN clustering on embeddings:
# For each cluster centroid, query CLIP with text prompts to extract labels

STYLE_PROMPTS = {
    "silhouette": ["fitted clothes", "oversized clothes", "relaxed fit", "structured tailoring"],
    "color_story": ["neutral tones beige white grey", "bold colors", "monochrome black", "warm earth tones"],
    "formality": ["casual streetwear", "smart casual", "business professional", "formal elegant"],
    "occasion": ["everyday casual", "office work", "evening going out", "athletic sport"]
}

def extract_attributes(taste_vector: np.ndarray) -> dict:
    attrs = {}
    for attr, prompts in STYLE_PROMPTS.items():
        text_tokens = clip.tokenize(prompts)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).numpy()
        sims = cosine_similarity([taste_vector], text_features)[0]
        attrs[attr] = prompts[sims.argmax()]
    return attrs
```

**Cost:** Zero. All local CLIP inference. ~2–4 seconds per onboarding on CPU.

-----

## Component 2: Wardrobe State

**Wardrobe = Phia saves.** For demo: pre-built wardrobe JSON sets representing different user profiles (minimalist, streetwear, business casual). Each item has: item_id, image_url, category, subcategory, price, brand, embedding.

### Slot Coverage Map

```python
OUTFIT_SLOTS = ["tops", "bottoms", "outerwear", "shoes", "bags", "accessories"]

def compute_slot_coverage(wardrobe_items: list[dict]) -> dict:
    coverage = {slot: [] for slot in OUTFIT_SLOTS}
    for item in wardrobe_items:
        slot = map_category_to_slot(item["category"])
        if slot:
            coverage[slot].append(item)
    return coverage

def get_gap_slots(coverage: dict) -> list[str]:
    # Return slots with 0 or 1 items, ordered by priority
    return [s for s in OUTFIT_SLOTS if len(coverage[s]) <= 1]
```

### Wardrobe Embedding

```python
def build_wardrobe_embedding(wardrobe_items: list[dict]) -> np.ndarray:
    embeddings = [item["embedding"] for item in wardrobe_items]
    # Recency-weighted: more recent saves weight higher
    weights = np.array([1.0 / (i + 1) for i in range(len(embeddings))])
    return build_taste_vector(np.array(embeddings), weights)
```

-----

## Component 3: Candidate Generation

**No full-catalog search. Constrained retrieval only.**

### Product Catalog

- Build a catalog of ~5,000 items using SerpApi Google Shopping queries per category
- Embed all items with CLIP on first run, store in FAISS index + Supabase
- Cache everything. Catalog doesn’t change during demo.

```python
# One-time build
def build_catalog_index(items: list[dict]) -> faiss.IndexFlatIP:
    embeddings = np.array([item["embedding"] for item in items]).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(512)
    index.add(embeddings)
    return index
```

### Gap-Targeted Retrieval

```python
def generate_candidates(
    taste_vector: np.ndarray,
    gap_slots: list[str],
    price_tier: tuple[float, float],
    catalog_index: faiss.IndexFlatIP,
    catalog_items: list[dict],
    top_k: int = 100
) -> list[dict]:
    
    query = taste_vector.astype("float32")
    faiss.normalize_L2(query.reshape(1, -1))
    scores, indices = catalog_index.search(query.reshape(1, -1), top_k * 3)
    
    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        item = catalog_items[idx]
        if item["slot"] not in gap_slots:
            continue
        if not (price_tier[0] * 0.7 <= item["price"] <= price_tier[1] * 1.3):
            continue
        item["retrieval_score"] = float(score)
        candidates.append(item)
        if len(candidates) >= top_k:
            break
    
    return candidates
```

-----

## Component 4: Outfit Utility Ranker

This is the core ML component. Two signals: compatibility score and outfit unlock count.

### Category-Conditioned Compatibility

```python
# W_AB matrices: initialized from rules, optionally fine-tuned
# Category pairs that go together get initialized with high positive weights
COMPLEMENT_RULES = {
    ("tops", "bottoms"): 0.9,
    ("tops", "shoes"): 0.7,
    ("tops", "outerwear"): 0.85,
    ("bottoms", "shoes"): 0.8,
    ("bottoms", "bags"): 0.6,
    ("outerwear", "bottoms"): 0.75,
}

def compatibility_score(
    candidate: dict,
    anchor: dict,
    W_matrices: dict  # {(cat_a, cat_b): np.ndarray shape (512, 512)}
) -> float:
    pair = (anchor["slot"], candidate["slot"])
    if pair not in COMPLEMENT_RULES:
        return 0.0
    
    base = COMPLEMENT_RULES[pair]
    
    if pair in W_matrices:
        # Learned: c_emb.T @ W_AB @ anchor_emb
        W = W_matrices[pair]
        score = candidate["embedding"] @ W @ anchor["embedding"]
    else:
        # Fallback: cosine sim scaled by rule base score
        score = cosine_similarity(
            [candidate["embedding"]], [anchor["embedding"]]
        )[0][0]
    
    return float(base * score)
```

### Outfit Unlock Count

```python
def outfit_unlock_count(
    candidate: dict,
    wardrobe: list[dict],
    min_outfit_size: int = 2
) -> int:
    """
    Count how many new complete outfit permutations become possible
    if candidate is added to wardrobe.
    A complete outfit requires at least one item per slot in a valid combination.
    """
    VALID_COMBOS = [
        {"tops", "bottoms"},
        {"tops", "bottoms", "shoes"},
        {"tops", "bottoms", "outerwear"},
        {"tops", "bottoms", "outerwear", "shoes"},
        {"tops", "shoes", "bags"},  # dress scenarios
    ]
    
    slot_items = {}
    for item in wardrobe:
        slot_items.setdefault(item["slot"], []).append(item)
    
    # Add candidate
    slot_items.setdefault(candidate["slot"], []).append(candidate)
    
    count = 0
    for combo in VALID_COMBOS:
        if all(slot in slot_items for slot in combo):
            # Count permutations across slots
            perms = 1
            for slot in combo:
                perms *= len(slot_items[slot])
            count += perms
    
    return count
```

### Final Ranking

```python
def rank_candidates(
    candidates: list[dict],
    wardrobe: list[dict],
    taste_vector: np.ndarray,
    weights: dict = None
) -> list[dict]:
    
    if weights is None:
        weights = {"taste": 0.4, "utility": 0.45, "value": 0.15}
    
    scored = []
    for c in candidates:
        taste_score = float(cosine_similarity([taste_vector], [c["embedding"]])[0][0])
        unlock_count = outfit_unlock_count(c, wardrobe)
        utility_score = min(unlock_count / 10.0, 1.0)  # normalize
        value_score = c.get("value_score", 0.5)  # from price data
        
        final = (
            weights["taste"] * taste_score +
            weights["utility"] * utility_score +
            weights["value"] * value_score
        )
        
        scored.append({**c, "final_score": final, "unlock_count": unlock_count, "taste_score": taste_score})
    
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Confidence gate: suppress if top score below threshold
    if scored and scored[0]["final_score"] < 0.35:
        return []
    
    return scored[:5]
```

-----

## Component 5: Explanation Templates

No LLM in this path. Pure template selection based on top contributing feature.

```python
TEMPLATES = {
    "high_utility_high_taste": "Pairs with {n} of your saves — your highest outfit unlock this week.",
    "high_utility": "Unlocks {n} new outfit combinations with your existing saves.",
    "taste_match": "Matches your {aesthetic} aesthetic — consistent with your saved palette.",
    "gap_fill": "Fills a {slot} gap in your saves — you're strong in {existing_slot} already.",
    "value": "Best price for this aesthetic match is resale at {discount}% below retail.",
}

def generate_explanation(candidate: dict, wardrobe_state: dict) -> str:
    unlock = candidate["unlock_count"]
    taste = candidate["taste_score"]
    slot = candidate["slot"]
    
    if unlock >= 6 and taste >= 0.7:
        return TEMPLATES["high_utility_high_taste"].format(n=unlock)
    elif unlock >= 4:
        return TEMPLATES["high_utility"].format(n=unlock)
    elif taste >= 0.8:
        return TEMPLATES["taste_match"].format(aesthetic=wardrobe_state["aesthetic_attributes"]["silhouette"])
    else:
        return TEMPLATES["gap_fill"].format(
            slot=slot,
            existing_slot=wardrobe_state["strongest_slot"]
        )
```

-----

## API Contracts

### POST /taste/extract

```json
// Request
{ "pinterest_url": "...", "images": ["base64..."] }

// Response
{
  "taste_vector": [...],
  "aesthetic_attributes": {
    "silhouette": "relaxed fit",
    "color_story": "neutral tones",
    "formality": "smart casual",
    "occasion": "everyday casual"
  },
  "price_tier": [60, 180]
}
```

### POST /recommendations/wardrobe

```json
// Request
{ "user_id": "...", "wardrobe_item_ids": [...], "taste_vector": [...] }

// Response
{
  "gap_recommendation": {
    "item": { ... },
    "unlock_count": 8,
    "taste_score": 0.86,
    "explanation": "Unlocks 8 new outfit combinations with your existing saves.",
    "confidence": 0.91
  },
  "complete_the_look": { ... },
  "model_version": "v0.1"
}
```

### POST /recommendations/evaluate-item

```json
// Request
{ "product_url": "...", "user_id": "...", "wardrobe_item_ids": [...], "taste_vector": [...] }

// Response
{
  "taste_fit": 0.84,
  "unlock_count": 5,
  "pairs_with": ["item_id_1", "item_id_3"],
  "explanation": "Pairs with your saved linen trouser — unlocks 5 new looks.",
  "confidence": 0.88
}
```

-----

## Data Models (Supabase)

```sql
-- Users and taste profiles
create table taste_profiles (
  user_id uuid primary key,
  taste_vector float8[],
  aesthetic_attributes jsonb,
  price_tier_low float,
  price_tier_high float,
  source text,  -- 'pinterest' | 'upload' | 'saves'
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Simulated wardrobe saves
create table wardrobe_saves (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references taste_profiles,
  item_id text,
  item_data jsonb,  -- full item with embedding
  saved_at timestamptz default now()
);

-- Product catalog cache
create table catalog_items (
  item_id text primary key,
  title text,
  brand text,
  category text,
  slot text,
  price float,
  image_url text,
  embedding float8[],
  source text,
  cached_at timestamptz default now()
);

-- Logging for evaluation
create table recommendation_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  event_type text,  -- 'impression' | 'click' | 'save'
  module text,
  item_id text,
  score float,
  unlock_count int,
  taste_score float,
  model_version text,
  timestamp timestamptz default now()
);
```

-----

## Evaluation Implementation

Three evaluations to run. These are what separate this from a demo.

### Eval 1: Cold Start Preference Prediction

```python
# For each simulated user profile (3 aesthetic clusters × 5 wardrobe sizes):
# Hold out 20% of saves. Score held-out items vs random negatives using taste_vector.
# Metric: Precision@10 vs popularity baseline.

def eval_cold_start(user_profiles, catalog):
    results = []
    for profile in user_profiles:
        for n_saves in [0, 3, 5, 10]:
            taste_vec = build_taste_from_n_saves(profile, n_saves)
            held_out = profile["saves"][n_saves:]
            scores = score_items_against_taste(taste_vec, held_out + sample_negatives(catalog, 100))
            p_at_10 = precision_at_k(scores, held_out, k=10)
            results.append({"n_saves": n_saves, "precision_at_10": p_at_10})
    return results
```

### Eval 2: Outfit Utility Calibration

```python
# Hypothesis: items with high predicted outfit_unlock_count appear more often
# in co-save sessions (items saved together).
# Use simulated wardrobe data with known outfit sets as ground truth.

def eval_utility_calibration(outfit_ground_truth, catalog):
    predicted_unlocks = []
    actual_co_saves = []
    for outfit in outfit_ground_truth:
        for item in outfit["items"]:
            predicted = outfit_unlock_count(item, outfit["items"])
            actual = outfit["co_save_rate"]
            predicted_unlocks.append(predicted)
            actual_co_saves.append(actual)
    return spearmanr(predicted_unlocks, actual_co_saves)
```

### Eval 3: AUC-ROC at Varying Wardrobe Sizes

```python
# For each wardrobe size, binary classification: will user save this item or not?
# Score using final_score from ranker. Compute AUC-ROC.
# Show improvement curve as wardrobe size increases.
```

-----

## Build Order

```
1. CLIP embedding pipeline (images → embeddings)
2. Pinterest scraper + taste extraction
3. Catalog builder (SerpApi queries + FAISS index)
4. Wardrobe state + slot coverage
5. Candidate generation (ANN retrieval)
6. Outfit unlock count + compatibility scoring
7. Final ranker + confidence gate
8. Explanation templates
9. FastAPI endpoints
10. Next.js demo UI (onboarding + wardrobe view + product evaluation panel)
11. Supabase logging
12. Evaluation scripts
```

Do not build 10 before 7 is producing sensible outputs. Test the ranker in a notebook first.

-----

## Cost Estimates

|Component                                    |Cost              |
|---------------------------------------------|------------------|
|CLIP inference                               |$0 — local CPU    |
|FAISS vector search                          |$0 — local        |
|SerpApi catalog build (5k items, one-time)   |~$5               |
|Supabase (free tier)                         |$0                |
|Vercel deployment                            |$0                |
|Claude Haiku (if used for template selection)|<$1 total for demo|
|**Total**                                    |**~$5–10**        |

-----

## Demo Wardrobe Profiles

Pre-build 4 simulated user wardrobes with real product data. These power the demo without requiring real users.

- **Profile A:** Minimalist neutral — 8 saves, strong in tops, missing outerwear and shoes
- **Profile B:** Streetwear — 12 saves, strong in outerwear and shoes, missing versatile bottoms
- **Profile C:** Smart casual — 6 saves, balanced but missing bags and accessories
- **Profile D:** Empty wardrobe — 0 saves, taste from Pinterest only

Each profile demonstrates a different system behavior. Profile D specifically proves cold start works.