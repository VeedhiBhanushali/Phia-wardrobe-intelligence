"""
Catalog builder — generates embeddings for mock or real product data.

Usage:
    python -m app.data.catalog_builder --source mock
    python -m app.data.catalog_builder --source serpapi
    python -m app.data.catalog_builder --source huggingface
"""

import argparse
import json
import sys
from pathlib import Path
from io import BytesIO

import numpy as np


def build_mock_catalog():
    """Build catalog from mock data using CLIP text embeddings as proxies."""
    from app.core.clip_encoder import get_encoder
    from app.data.mock_data import generate_catalog

    encoder = get_encoder()
    catalog = generate_catalog()

    print(f"Encoding {len(catalog)} items with CLIP text embeddings...")

    descriptions = [item["clip_description"] for item in catalog]

    batch_size = 32
    all_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i : i + batch_size]
        embeddings = encoder.encode_texts(batch)
        all_embeddings.append(embeddings)
        print(f"  Encoded {min(i + batch_size, len(descriptions))}/{len(descriptions)}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    for i, item in enumerate(catalog):
        item["embedding"] = all_embeddings[i].tolist()
        item.pop("clip_description", None)

    return catalog


def build_serpapi_catalog():
    """Build catalog from SerpApi Google Shopping results with trend-enriched queries."""
    import httpx
    from app.config import get_settings

    settings = get_settings()
    if not settings.serpapi_key:
        print("ERROR: SERPAPI_KEY not set in .env")
        sys.exit(1)

    categories = {
        "tops": [
            "COS silk blouse women", "Aritzia babaton top women",
            "Massimo Dutti fitted shirt women", "Zara satin camisole women",
            "Reformation linen top women", "Theory cashmere sweater women",
            "Mango ribbed knit top women", "H&M premium cotton tee women",
            "Sandro lace trim blouse women", "Reiss structured knit top women",
        ],
        "bottoms": [
            "Agolde jeans women", "COS tailored trousers women",
            "Zara pleated midi skirt women", "Aritzia wide leg pants women",
            "Mango leather effect pants women", "Massimo Dutti linen trousers women",
            "Reformation high rise jeans women", "Reiss pencil skirt women",
            "H&M premium satin skirt women", "AllSaints cargo pants women",
        ],
        "outerwear": [
            "COS wool coat women", "Aritzia superpuff jacket women",
            "Massimo Dutti camel blazer women", "Zara oversized blazer women",
            "Mango leather jacket women", "AllSaints biker jacket women",
            "Reiss wool blend coat women", "Theory tailored blazer women",
            "H&M premium trench coat women", "Sandro tweed jacket women",
        ],
        "shoes": [
            "Aeyde ballet flat women", "Adidas Samba sneakers women",
            "Mango leather loafers women", "Zara strappy heeled sandal women",
            "Steve Madden platform boots women", "Sam Edelman pointed toe flat women",
            "COS leather ankle boots women", "New Balance 550 women",
            "Massimo Dutti kitten heel women",
        ],
        "bags": [
            "Polene leather bag women", "Mansur Gavriel bucket bag women",
            "COS quilted crossbody bag women", "Mango leather tote bag women",
            "Zara minimalist shoulder bag women", "Coach tabby bag women",
            "JW Pei shoulder bag women", "DeMellier mini bag women",
        ],
        "accessories": [
            "Mejuri gold layered necklace women", "Missoma pearl earrings women",
            "COS silk scarf women", "Mango oversized sunglasses women",
            "Monica Vinader bracelet women", "Zara chain link necklace women",
            "Mango leather belt women", "COS cashmere beanie women",
        ],
    }

    catalog = []
    item_counter = 0

    for slot, queries in categories.items():
        for query in queries:
            print(f"Fetching: {query}")
            resp = httpx.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_shopping",
                    "q": query,
                    "api_key": settings.serpapi_key,
                    "num": 30,
                },
                timeout=30,
            )
            data = resp.json()

            for result in data.get("shopping_results", []):
                catalog.append({
                    "item_id": f"{slot}_{item_counter:04d}",
                    "title": result.get("title", ""),
                    "brand": result.get("source", ""),
                    "category": query.split()[-1],
                    "slot": slot,
                    "price": _parse_price(result.get("extracted_price", 0)),
                    "image_url": result.get("thumbnail", ""),
                    "source": "serpapi",
                })
                item_counter += 1

    from app.core.clip_encoder import get_encoder
    encoder = get_encoder()

    img_dir = Path("data/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and encoding {len(catalog)} product images...")
    for i, item in enumerate(catalog):
        try:
            resp = httpx.get(item["image_url"], timeout=10, follow_redirects=True)
            resp.raise_for_status()
            img_bytes = resp.content
            embedding = encoder.encode_images([img_bytes])[0]

            img_path = img_dir / f"{item['item_id']}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            item["image_url"] = f"/static/images/{item['item_id']}.jpg"
        except Exception:
            embedding = encoder.encode_texts([f"{item['brand']} {item['title']}"])[0]
        item["embedding"] = embedding.tolist()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(catalog)}")

    return catalog


_HF_SLOT_MAP = {
    "Shirts": "tops",
    "Tshirts": "tops",
    "Tops": "tops",
    "Sweatshirts": "tops",
    "Sweaters": "tops",
    "Tunics": "tops",
    "Blouse": "tops",
    "Tank Top": "tops",
    "Kurtas": "tops",

    "Jeans": "bottoms",
    "Trousers": "bottoms",
    "Track Pants": "bottoms",
    "Shorts": "bottoms",
    "Skirts": "bottoms",
    "Leggings": "bottoms",
    "Capris": "bottoms",
    "Palazzos": "bottoms",
    "Jeggings": "bottoms",

    "Jackets": "outerwear",
    "Coats": "outerwear",
    "Blazers": "outerwear",
    "Sweaters": "outerwear",
    "Rain Jacket": "outerwear",
    "Nehru Jackets": "outerwear",
    "Waistcoat": "outerwear",

    "Casual Shoes": "shoes",
    "Sports Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Flats": "shoes",
    "Heels": "shoes",
    "Sandals": "shoes",
    "Flip Flops": "shoes",
    "Sneakers": "shoes",
    "Boots": "shoes",
    "Sports Sandals": "shoes",

    "Handbags": "bags",
    "Backpacks": "bags",
    "Clutches": "bags",
    "Messenger Bag": "bags",
    "Laptop Bag": "bags",
    "Duffel Bag": "bags",
    "Travel Accessory": "bags",
    "Trolley Bag": "bags",

    "Watches": "accessories",
    "Sunglasses": "accessories",
    "Belts": "accessories",
    "Scarves": "accessories",
    "Earrings": "accessories",
    "Necklace and Chains": "accessories",
    "Bracelet": "accessories",
    "Ring": "accessories",
    "Pendant": "accessories",
    "Jewellery Set": "accessories",
    "Hair Accessory": "accessories",
    "Caps": "accessories",
    "Hat": "accessories",
    "Ties": "accessories",
    "Mufflers": "accessories",
    "Stoles": "accessories",
    "Cufflinks": "accessories",
}

_HF_SLOT_TARGETS = {
    "tops": 100,
    "bottoms": 100,
    "outerwear": 60,
    "shoes": 80,
    "bags": 60,
    "accessories": 100,
}


def build_huggingface_catalog():
    """
    Build catalog from the ashraq/fashion-product-images-small dataset.

    Uses real product images encoded with CLIP encode_image(), closing
    the modality gap between Pinterest taste vectors and catalog embeddings.
    """
    from datasets import load_dataset
    from app.core.clip_encoder import get_encoder

    print("Loading fashion dataset from Hugging Face...")
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")

    encoder = get_encoder()
    catalog = []
    slot_counts: dict[str, int] = {s: 0 for s in _HF_SLOT_TARGETS}
    item_counter = 0
    skipped = 0

    total = len(ds)
    print(f"Scanning {total} items for slot-balanced selection...")

    for row in ds:
        article_type = row.get("articleType", "")
        slot = _HF_SLOT_MAP.get(article_type)
        if slot is None:
            continue
        if slot_counts[slot] >= _HF_SLOT_TARGETS[slot]:
            if all(slot_counts[s] >= _HF_SLOT_TARGETS[s] for s in _HF_SLOT_TARGETS):
                break
            continue

        img = row.get("image")
        if img is None:
            skipped += 1
            continue

        name = row.get("productDisplayName", "") or f"{article_type}"
        colour = row.get("baseColour", "")
        season = row.get("season", "")
        brand = row.get("brandName", "") or row.get("brand", "") or "Fashion"
        usage = row.get("usage", "")

        item_id = f"hf_{slot}_{item_counter:04d}"

        try:
            rgb = img.convert("RGB")
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            embedding = encoder.encode_images([img_bytes])[0]

            img_dir = Path("data/images")
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{item_id}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except Exception:
            skipped += 1
            continue

        price = _synthetic_price(slot, usage)

        catalog.append({
            "item_id": item_id,
            "title": name,
            "brand": brand,
            "category": article_type.lower(),
            "slot": slot,
            "price": price,
            "image_url": f"/static/images/{item_id}.jpg",
            "source": "huggingface",
            "colour": colour,
            "season": season,
            "embedding": embedding.tolist(),
        })

        slot_counts[slot] += 1
        item_counter += 1

        if item_counter % 25 == 0:
            filled = {s: slot_counts[s] for s in _HF_SLOT_TARGETS}
            print(f"  {item_counter} items selected | {filled}")

    print(f"\nFinal catalog: {len(catalog)} items ({skipped} skipped)")
    for s in _HF_SLOT_TARGETS:
        print(f"  {s}: {slot_counts[s]}/{_HF_SLOT_TARGETS[s]}")

    return catalog


def _synthetic_price(slot: str, usage: str) -> float:
    """
    Generate a realistic price for HF dataset items that lack price data.
    Uses slot-based ranges with slight variation from usage context.
    """
    import random
    base_ranges = {
        "tops": (28, 120),
        "bottoms": (35, 150),
        "outerwear": (80, 300),
        "shoes": (45, 200),
        "bags": (40, 250),
        "accessories": (20, 120),
    }
    low, high = base_ranges.get(slot, (30, 150))
    if usage and "formal" in usage.lower():
        low = int(low * 1.3)
        high = int(high * 1.3)
    elif usage and "sports" in usage.lower():
        low = int(low * 0.7)
        high = int(high * 0.9)
    return round(random.uniform(low, high), 2)


def _parse_price(price) -> float:
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        cleaned = price.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Build product catalog")
    parser.add_argument(
        "--source",
        choices=["mock", "serpapi", "huggingface"],
        default="huggingface",
        help="Data source for catalog items",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for index files",
    )
    args = parser.parse_args()

    if args.source == "mock":
        catalog = build_mock_catalog()
    elif args.source == "serpapi":
        catalog = build_serpapi_catalog()
    else:
        catalog = build_huggingface_catalog()

    from app.core.candidates import build_index, save_index

    print(f"Building FAISS index for {len(catalog)} items...")
    index = build_index(catalog)
    save_index(index, catalog, base_path=args.output_dir)
    print(f"Saved index and catalog to {args.output_dir}/")
    print(f"  Items: {len(catalog)}")
    print(f"  Slots: {set(item['slot'] for item in catalog)}")


if __name__ == "__main__":
    main()
