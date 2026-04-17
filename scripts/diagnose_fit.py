"""
Pre-pitch diagnostic: three audits in one run.

1. Silhouette vs fit-axis agreement   — flags contradictions
2. Percentile calibration visual audit — top-10 / bottom-10 items
3. Purchase confidence tier audit      — 10 HIGH / 10 LOW with component scores

Usage:
    python scripts/diagnose_fit.py
"""

import os, sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))
os.chdir(_backend)

import numpy as np
from app.core.clip_encoder import get_encoder
from app.core.taste import (
    build_taste_vector,
    extract_attributes,
    compute_style_attribute_profile,
    style_attribute_summary,
)
from app.core.candidates import load_index
from app.core.ranker import outfit_unlock_count, find_pairs
from app.data.mock_data import DEMO_WARDROBES

STYLE_PROFILES = {
    "minimalist": "minimalist neutral clean structured tailored beige white ivory clothing",
    "streetwear": "streetwear oversized hoodie sneakers cargo pants bold urban clothing",
    "elegant":    "elegant formal polished evening cocktail heels satin silk clothing",
}

FIT_AXES = ["fit_oversized", "fit_tailored", "fit_flowy"]

SEPARATOR = "=" * 72


def _get_wardrobe_items(item_ids: list[str], catalog: list[dict]) -> list[dict]:
    id_set = set(item_ids)
    return [c for c in catalog if c["item_id"] in id_set]


def _build_taste_percentile_fn(taste_vector, catalog, sample_size=500):
    rng = np.random.RandomState(0)
    indices = rng.choice(len(catalog), size=min(sample_size, len(catalog)), replace=False)
    sample_scores = []
    for idx in indices:
        emb = np.array(catalog[idx]["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        sample_scores.append(max(0.0, float(np.dot(emb, taste_vector))))
    sample_scores.sort()
    arr = np.array(sample_scores)

    def to_percentile(raw_score: float) -> float:
        pct = float(np.searchsorted(arr, raw_score, side="right")) / len(arr)
        return min(pct, 0.99)

    return to_percentile


# ── Diagnostic 1: Silhouette vs Fit Axis ────────────────────

def diagnose_silhouette_vs_fit(encoder, catalog):
    print(f"\n{SEPARATOR}")
    print("  DIAGNOSTIC 1: Silhouette vs Fit-Axis Agreement")
    print(SEPARATOR)

    for profile_name, style_text in STYLE_PROFILES.items():
        print(f"\n── {profile_name.upper()} ──")

        text_emb = encoder.encode_texts([style_text])
        taste_vector = text_emb[0].astype(np.float32)
        n = np.linalg.norm(taste_vector)
        if n > 0:
            taste_vector = taste_vector / n

        attributes = extract_attributes(taste_vector)
        sil = attributes.get("silhouette", {})
        sil_label = sil.get("label", "?")
        sil_scores = sil.get("scores", {})

        print(f"  Silhouette label: {sil_label}")
        print(f"  Silhouette scores: {', '.join(f'{k}: {v:.3f}' for k, v in sil_scores.items())}")

        emb_2d = taste_vector.reshape(1, -1)
        style_attrs = compute_style_attribute_profile(emb_2d, encoder)

        print(f"  Fit axes:")
        for ax in FIT_AXES:
            val = style_attrs.get(ax, 0.0)
            direction = "prefers" if val > 0.35 else ("avoids" if val < -0.35 else "neutral")
            print(f"    {ax:20s} = {val:+.3f}  ({direction})")

        contradictions = []
        if sil_label in ("Relaxed", "Oversized") and style_attrs.get("fit_oversized", 0) < -0.35:
            contradictions.append(f"Silhouette={sil_label} but fit_oversized={style_attrs['fit_oversized']:.3f} (avoids)")
        if sil_label in ("Fitted", "Structured") and style_attrs.get("fit_tailored", 0) < -0.35:
            contradictions.append(f"Silhouette={sil_label} but fit_tailored={style_attrs['fit_tailored']:.3f} (avoids)")
        if sil_label in ("Relaxed", "Oversized") and style_attrs.get("fit_tailored", 0) > 0.35:
            contradictions.append(f"Silhouette={sil_label} but fit_tailored={style_attrs['fit_tailored']:.3f} (prefers)")
        if sil_label in ("Fitted", "Structured") and style_attrs.get("fit_oversized", 0) > 0.35:
            contradictions.append(f"Silhouette={sil_label} but fit_oversized={style_attrs['fit_oversized']:.3f} (prefers)")

        if contradictions:
            print(f"  ⚠ CONTRADICTIONS:")
            for c in contradictions:
                print(f"    - {c}")
        else:
            print(f"  ✓ No contradictions")


# ── Diagnostic 2: Percentile Calibration ────────────────────

def diagnose_percentile_calibration(encoder, catalog):
    print(f"\n{SEPARATOR}")
    print("  DIAGNOSTIC 2: Percentile Calibration Visual Audit")
    print(SEPARATOR)

    for profile_name, style_text in STYLE_PROFILES.items():
        print(f"\n── {profile_name.upper()} ──")

        text_emb = encoder.encode_texts([style_text])
        taste_vector = text_emb[0].astype(np.float32)
        n = np.linalg.norm(taste_vector)
        if n > 0:
            taste_vector = taste_vector / n

        to_pct = _build_taste_percentile_fn(taste_vector, catalog)

        scored = []
        for item in catalog:
            emb = np.array(item["embedding"], dtype=np.float32)
            en = np.linalg.norm(emb)
            if en > 0:
                emb = emb / en
            raw = max(0.0, float(np.dot(taste_vector, emb)))
            pct = to_pct(raw)
            scored.append((pct, raw, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"\n  TOP 10 (highest percentile — should look on-brand):")
        for i, (pct, raw, item) in enumerate(scored[:10]):
            print(f"    {i+1:2d}. {pct*100:5.1f}%  raw={raw:.3f}  {item.get('brand','?'):20s}  {item.get('name','?')[:50]}")
            img = item.get("image_url", item.get("thumbnail_url", ""))
            if img:
                print(f"        {img[:100]}")

        bottom = [s for s in scored if s[0] < 0.20]
        bottom.sort(key=lambda x: x[0])
        print(f"\n  BOTTOM 10 (lowest percentile — should look off-brand):")
        for i, (pct, raw, item) in enumerate(bottom[:10]):
            print(f"    {i+1:2d}. {pct*100:5.1f}%  raw={raw:.3f}  {item.get('brand','?'):20s}  {item.get('name','?')[:50]}")
            img = item.get("image_url", item.get("thumbnail_url", ""))
            if img:
                print(f"        {img[:100]}")


# ── Diagnostic 3: Purchase Confidence Tier ──────────────────

def diagnose_purchase_confidence(encoder, catalog):
    print(f"\n{SEPARATOR}")
    print("  DIAGNOSTIC 3: Purchase Confidence Tier Audit")
    print(SEPARATOR)

    for profile_name, style_text in STYLE_PROFILES.items():
        wardrobe_key = profile_name if profile_name in DEMO_WARDROBES else "minimalist"
        wardrobe_ids = DEMO_WARDROBES[wardrobe_key]["item_ids"]
        wardrobe = _get_wardrobe_items(wardrobe_ids, catalog)

        print(f"\n── {profile_name.upper()} (wardrobe: {wardrobe_key}, {len(wardrobe)} items) ──")

        text_emb = encoder.encode_texts([style_text])
        taste_vector = text_emb[0].astype(np.float32)
        n = np.linalg.norm(taste_vector)
        if n > 0:
            taste_vector = taste_vector / n

        scored_items = []
        for item in catalog:
            if item["item_id"] in set(wardrobe_ids):
                continue

            emb = np.array(item["embedding"], dtype=np.float32)
            en = np.linalg.norm(emb)
            if en > 0:
                emb_n = emb / en
            else:
                emb_n = emb

            raw_taste = max(0.0, min(1.0, float(np.dot(taste_vector, emb_n))))
            unlock = outfit_unlock_count(item, wardrobe)
            pairs = find_pairs(item, wardrobe)

            w_size = max(len(wardrobe), 1)
            unlock_denom = max(w_size * 2, 10)
            pairs_denom = max(w_size * 2, 8)
            score = (
                raw_taste * 0.50
                + min(unlock / unlock_denom, 1.0) * 0.25
                + min(len(pairs) / pairs_denom, 1.0) * 0.25
            )
            if score >= 0.35:
                conf = "HIGH"
            elif score >= 0.18:
                conf = "MEDIUM"
            else:
                conf = "LOW"

            scored_items.append({
                "item": item,
                "raw_taste": raw_taste,
                "unlock": unlock,
                "pairs_count": len(pairs),
                "score": score,
                "confidence": conf,
            })

        highs = sorted([s for s in scored_items if s["confidence"] == "HIGH"],
                        key=lambda x: x["score"], reverse=True)
        lows = sorted([s for s in scored_items if s["confidence"] == "LOW"],
                       key=lambda x: x["score"])

        print(f"\n  HIGH confidence ({len(highs)} total) — top 10:")
        for i, s in enumerate(highs[:10]):
            it = s["item"]
            print(f"    {i+1:2d}. score={s['score']:.3f}  taste={s['raw_taste']:.3f}  "
                  f"unlock={s['unlock']}  pairs={s['pairs_count']}  "
                  f"conf={s['confidence']}  {it.get('brand','?'):20s}  {it.get('name','?')[:45]}")

        print(f"\n  LOW confidence ({len(lows)} total) — bottom 10:")
        for i, s in enumerate(lows[:10]):
            it = s["item"]
            print(f"    {i+1:2d}. score={s['score']:.3f}  taste={s['raw_taste']:.3f}  "
                  f"unlock={s['unlock']}  pairs={s['pairs_count']}  "
                  f"conf={s['confidence']}  {it.get('brand','?'):20s}  {it.get('name','?')[:45]}")

        med_count = sum(1 for s in scored_items if s["confidence"] == "MEDIUM")
        print(f"\n  Distribution: HIGH={len(highs)} MEDIUM={med_count} LOW={len(lows)}")
        if highs:
            taste_range = [s["raw_taste"] for s in highs[:10]]
            print(f"  HIGH taste range: {min(taste_range):.3f} – {max(taste_range):.3f}")
        if lows:
            taste_range = [s["raw_taste"] for s in lows[:10]]
            print(f"  LOW taste range:  {min(taste_range):.3f} – {max(taste_range):.3f}")
        if highs and lows:
            gap = np.mean([s["raw_taste"] for s in highs[:10]]) - np.mean([s["raw_taste"] for s in lows[:10]])
            print(f"  Taste separation (HIGH top10 mean - LOW bottom10 mean): {gap:.3f}")


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading encoder and catalog...")
    encoder = get_encoder()
    _, catalog = load_index()
    print(f"Catalog: {len(catalog)} items")

    diagnose_silhouette_vs_fit(encoder, catalog)
    diagnose_percentile_calibration(encoder, catalog)
    diagnose_purchase_confidence(encoder, catalog)

    print(f"\n{SEPARATOR}")
    print("  DONE — Review above for contradictions and tier quality.")
    print(SEPARATOR)
