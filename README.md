# Wardrobe IQ — Purchase Intelligence for Phia

**A three-signal personalization system that answers not just "is this the best price?" but "should I buy this, given who I am and what I already own?"**

*Built by Veedhi Bhanushali · April 2026*

---

## What It Does

Phia is a fashion price-comparison platform. This project builds the personalization layer on top of it — a working proof of concept for Phia's Series A roadmap across three signals:

| Signal | Nature | What It Captures |
|--------|--------|-----------------|
| **Taste** | Persistent | Aesthetic identity extracted from Pinterest boards via FashionCLIP |
| **Wardrobe** | Persistent | What you own, with live outfit-combination tracking |
| **Intent** | Ephemeral | What you need right now, inferred from browsing and chat |

Every recommendation is the intersection of all three simultaneously.

---

## Results

| Signal | Metric | Result |
|--------|--------|--------|
| Taste | Mean relevance @ 10 vs random baseline | **+102%** at zero saves |
| Taste | Mean relevance @ 10 vs catalog centroid | **+52%** at zero saves |
| Intent | P@10 lift during focused browsing sessions | **+13.3%** over taste-only |
| Trust | Coherent vs incoherent ranking accuracy | **69.5%** (vs 50% random) |

Meaningful personalization before any interaction history. Stable signal as wardrobe context is introduced.

---

## Product Surfaces

- **Wardrobe Builder** — virtual closet by slot (Tops / Bottoms / Shoes / etc.), with live outfit-unlock counts per save
- **Aesthetic Builder** — paste a Pinterest URL, get a named aesthetic, dominant palette, 25-axis style profile, and trend fingerprint in ~3 seconds
- **Discovery Feed** — wardrobe gap fills, aesthetic matches, outfit completions, occasion picks, and price signals — all ranked by all three signals
- **Stylist Chat** — full-screen agentic conversation powered by Claude with live tool access to wardrobe, catalog, and taste profile; streams shoppable outfit recommendations via SSE

---

## Stack

**Backend** — Python 3.13, FastAPI, FashionCLIP (ViT-B/32, 512-d, fine-tuned on 800K Farfetch pairs), FAISS (IndexFlatIP), Claude API (tool use + streaming), NumPy, scikit-learn

**Frontend** — Next.js 15, React, TypeScript, Tailwind CSS, Framer Motion, Server-Sent Events

**Data** — 2,364-item catalog from `ashraq/fashion-product-images-small`, encoded with FashionCLIP image encoder; FAISS index persisted as `faiss_index.bin`; user state in localStorage

---

## Running Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Set `ANTHROPIC_API_KEY` in `backend/.env` before starting.

---

## Docs

- [`docs/technical-brief.md`](docs/technical-brief.md) — concise signal-by-signal breakdown with metrics
- [`docs/technical-writeup.md`](docs/technical-writeup.md) — full system design, evaluation methodology, and known limitations
