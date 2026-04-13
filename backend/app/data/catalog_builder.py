"""
Catalog builder — generates embeddings for mock or real product data.

Usage:
    python -m app.data.catalog_builder --source mock
    python -m app.data.catalog_builder --source serpapi
    python -m app.data.catalog_builder --source huggingface
"""

import argparse
import json
import re
import sys
from pathlib import Path
from io import BytesIO

import numpy as np


# ── Metadata enrichment ──────────────────────────────────────────────

_ITEM_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("dress",       re.compile(r"\bdress(es)?\b", re.I)),
    ("jumpsuit",    re.compile(r"\bjumpsuit\b", re.I)),
    ("romper",      re.compile(r"\bromper\b", re.I)),
    ("bodysuit",    re.compile(r"\bbodysuit\b", re.I)),
    ("blouse",      re.compile(r"\bblouse\b", re.I)),
    ("shirt",       re.compile(r"\b(shirt|button[- ]?down|oxford)\b", re.I)),
    ("camisole",    re.compile(r"\b(cami(sole)?|slip top)\b", re.I)),
    ("tank",        re.compile(r"\btank(\s?top)?\b", re.I)),
    ("tee",         re.compile(r"\b(tee|t-shirt|tshirt)\b", re.I)),
    ("crop top",    re.compile(r"\bcrop\s?top\b", re.I)),
    ("corset",      re.compile(r"\bcorset\b", re.I)),
    ("sweater",     re.compile(r"\b(sweater|knit|pullover|crewneck|turtleneck|cardigan)\b", re.I)),
    ("hoodie",      re.compile(r"\b(hoodie|sweatshirt)\b", re.I)),
    ("polo",        re.compile(r"\bpolo\b", re.I)),
    ("vest",        re.compile(r"\bvest\b", re.I)),
    ("top",         re.compile(r"\btop\b", re.I)),
    ("jeans",       re.compile(r"\b(jeans|denim\s+pant)\b", re.I)),
    ("trousers",    re.compile(r"\b(trousers|pants|chinos|joggers|palazzo|bermuda)\b", re.I)),
    ("skirt",       re.compile(r"\b(skirt|skort)\b", re.I)),
    ("shorts",      re.compile(r"\bshorts\b", re.I)),
    ("leggings",    re.compile(r"\b(leggings|tights)\b", re.I)),
    ("coat",        re.compile(r"\b(coat|trench|overcoat|duster)\b", re.I)),
    ("jacket",      re.compile(r"\b(jacket|bomber|puffer|biker|trucker)\b", re.I)),
    ("blazer",      re.compile(r"\bblazer\b", re.I)),
    ("sneakers",    re.compile(r"\b(sneaker|trainer|samba|new balance|nb\s?\d)\b", re.I)),
    ("boots",       re.compile(r"\bboots?\b", re.I)),
    ("heels",       re.compile(r"\b(heel|pump|stiletto|kitten heel)\b", re.I)),
    ("sandals",     re.compile(r"\b(sandal|slide|flip\s?flop|mule)\b", re.I)),
    ("loafers",     re.compile(r"\b(loafer|oxford shoe|derby)\b", re.I)),
    ("flats",       re.compile(r"\b(flat|ballet)\b", re.I)),
    ("clogs",       re.compile(r"\b(clog|birkenstock)\b", re.I)),
    ("tote",        re.compile(r"\btote\b", re.I)),
    ("crossbody",   re.compile(r"\bcrossbody\b", re.I)),
    ("clutch",      re.compile(r"\bclutch\b", re.I)),
    ("backpack",    re.compile(r"\bbackpack\b", re.I)),
    ("bag",         re.compile(r"\b(bag|satchel|purse|hobo|bucket bag|shoulder bag)\b", re.I)),
    ("necklace",    re.compile(r"\bnecklace\b", re.I)),
    ("earrings",    re.compile(r"\bearring\b", re.I)),
    ("bracelet",    re.compile(r"\b(bracelet|bangle|cuff)\b", re.I)),
    ("ring",        re.compile(r"\bring\b", re.I)),
    ("watch",       re.compile(r"\bwatch\b", re.I)),
    ("sunglasses",  re.compile(r"\b(sunglasses|shades)\b", re.I)),
    ("belt",        re.compile(r"\bbelt\b", re.I)),
    ("scarf",       re.compile(r"\b(scarf|stole|muffler)\b", re.I)),
    ("hat",         re.compile(r"\b(hat|cap|beanie|beret)\b", re.I)),
    ("gloves",      re.compile(r"\bgloves?\b", re.I)),
]

# Canonical item_type synonyms: query alias → catalog canonical label.
# When the LLM or user says "gown" we match catalog items tagged "dress".
ITEM_TYPE_SYNONYMS: dict[str, str] = {
    "gown": "dress", "mini dress": "dress", "midi dress": "dress",
    "maxi dress": "dress", "cocktail dress": "dress", "slip dress": "dress",
    "sundress": "dress", "wrap dress": "dress", "bodycon": "dress",
    "sheath": "dress", "a-line dress": "dress",
    "jean": "jeans", "denim": "jeans",
    "pant": "trousers", "trouser": "trousers", "chino": "trousers",
    "jogger": "trousers", "palazzo": "trousers",
    "tshirt": "tee", "t-shirt": "tee",
    "cami": "camisole",
    "pullover": "sweater", "knit": "sweater", "crewneck": "sweater",
    "turtleneck": "sweater", "cardigan": "sweater",
    "sweatshirt": "hoodie",
    "puffer": "jacket", "bomber": "jacket",
    "trench": "coat", "overcoat": "coat", "duster": "coat",
    "trainer": "sneakers", "sneaker": "sneakers",
    "boot": "boots", "ankle boot": "boots", "knee boot": "boots",
    "pump": "heels", "stiletto": "heels",
    "sandal": "sandals", "slide": "sandals", "mule": "sandals",
    "loafer": "loafers",
    "ballet flat": "flats",
    "purse": "bag", "handbag": "bag", "satchel": "bag",
    "shoulder bag": "bag", "bucket bag": "bag", "hobo": "bag",
    "earring": "earrings",
    "bangle": "bracelet", "cuff": "bracelet",
    "beanie": "hat", "cap": "hat", "beret": "hat",
    "stole": "scarf", "muffler": "scarf",
    "shade": "sunglasses",
    "glove": "gloves",
}

_COLOR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("black",       re.compile(r"\bblack\b", re.I)),
    ("white",       re.compile(r"\b(white|ivory|cream|ecru|off[- ]?white)\b", re.I)),
    ("navy",        re.compile(r"\bnavy\b", re.I)),
    ("blue",        re.compile(r"\bblue\b", re.I)),
    ("red",         re.compile(r"\b(red|scarlet|crimson|cherry)\b", re.I)),
    ("burgundy",    re.compile(r"\b(burgundy|maroon|wine|oxblood|merlot)\b", re.I)),
    ("pink",        re.compile(r"\b(pink|rose|blush|fuchsia|magenta|dusty rose)\b", re.I)),
    ("green",       re.compile(r"\b(green|sage|olive|emerald|forest|khaki|moss)\b", re.I)),
    ("yellow",      re.compile(r"\b(yellow|mustard|gold(en)?|lemon)\b", re.I)),
    ("orange",      re.compile(r"\b(orange|rust|terracotta|burnt|tangerine)\b", re.I)),
    ("purple",      re.compile(r"\b(purple|violet|plum|lavender|lilac|mauve)\b", re.I)),
    ("brown",       re.compile(r"\b(brown|chocolate|espresso|cocoa|coffee|cognac|caramel)\b", re.I)),
    ("tan",         re.compile(r"\b(tan|camel|beige|khaki|nude|sand|stone|taupe|oat)\b", re.I)),
    ("grey",        re.compile(r"\b(gr[ae]y|charcoal|silver|slate)\b", re.I)),
    ("metallic",    re.compile(r"\b(metallic|gold|silver|shimmer|sequin|glitter)\b", re.I)),
    ("denim",       re.compile(r"\bdenim\b", re.I)),
    ("floral",      re.compile(r"\bfloral\b", re.I)),
    ("print",       re.compile(r"\b(print|pattern|stripe|plaid|check|houndstooth|leopard|polka)\b", re.I)),
]

# Color family groups: when a user searches for "red", also match items
# tagged as "burgundy" or "orange" (warm reds).  Used for relaxed fallback.
COLOR_FAMILIES: dict[str, list[str]] = {
    "red":      ["red", "burgundy", "orange"],
    "burgundy": ["burgundy", "red"],
    "pink":     ["pink", "red"],
    "blue":     ["blue", "navy", "denim"],
    "navy":     ["navy", "blue", "denim"],
    "green":    ["green"],
    "orange":   ["orange", "red"],
    "purple":   ["purple", "pink"],
    "brown":    ["brown", "tan"],
    "tan":      ["tan", "brown"],
    "white":    ["white"],
    "black":    ["black"],
    "grey":     ["grey", "metallic"],
    "yellow":   ["yellow", "orange"],
    "metallic": ["metallic", "grey", "yellow"],
    "denim":    ["denim", "blue", "navy"],
}

_OCCASION_KEYWORDS: dict[str, list[str]] = {
    "casual":   ["casual", "everyday", "relaxed", "weekend", "comfort"],
    "work":     ["office", "work", "professional", "formal", "tailored", "business"],
    "evening":  ["evening", "date", "night", "cocktail", "party", "club", "sexy"],
    "special":  ["wedding", "gala", "special", "event", "prom", "graduation"],
    "active":   ["sport", "athletic", "gym", "yoga", "running", "activewear"],
    "vacation": ["beach", "resort", "vacation", "travel", "summer", "tropical"],
}


def normalize_item_type(raw: str) -> str:
    """Resolve a query item_type through the synonym table to canonical form."""
    key = raw.lower().strip()
    return ITEM_TYPE_SYNONYMS.get(key, key)


def expand_color_family(colors: list[str]) -> list[str]:
    """Expand a list of query colors to include related color families."""
    expanded: set[str] = set()
    for c in colors:
        c_lower = c.lower().strip()
        expanded.add(c_lower)
        for related in COLOR_FAMILIES.get(c_lower, []):
            expanded.add(related)
    return list(expanded)


def _detect_item_type(text: str) -> str:
    """First-match item type from title/category text."""
    for label, pat in _ITEM_TYPE_PATTERNS:
        if pat.search(text):
            return label
    return "other"


def _detect_colors(text: str) -> list[str]:
    """All matching color families from title/description text."""
    found: list[str] = []
    for label, pat in _COLOR_PATTERNS:
        if pat.search(text):
            found.append(label)
    return found or ["unknown"]


def _detect_occasions(text: str) -> list[str]:
    text_lower = text.lower()
    found: list[str] = []
    for occasion, keywords in _OCCASION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(occasion)
    return found or ["casual"]


def enrich_catalog_metadata(catalog: list[dict]) -> list[dict]:
    """Add item_type, colors, and occasions to every catalog item in-place."""
    for item in catalog:
        search_text = " ".join(filter(None, [
            item.get("title", ""),
            item.get("category", ""),
            item.get("colour", ""),
            item.get("dominant_color", ""),
            item.get("clip_description", ""),
        ]))

        if not item.get("item_type"):
            item["item_type"] = _detect_item_type(search_text)

        if not item.get("colors"):
            explicit_color = item.get("colour") or item.get("dominant_color")
            detected = _detect_colors(search_text)
            if explicit_color and explicit_color.lower() not in detected:
                detected.insert(0, explicit_color.lower())
            item["colors"] = detected

        if not item.get("occasions"):
            item["occasions"] = _detect_occasions(search_text)

    return catalog


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
            "Zara satin camisole women", "Reformation linen top women",
            "Mango ribbed knit top women", "H&M premium cotton tee women",
            "Edikted corset top women", "Princess Polly crop top women",
            "White Fox tube top women", "Skims long sleeve top women",
            "Alo Yoga ribbed tank top women", "Abercrombie bodysuit women",
        ],
        "bottoms": [
            "Agolde jeans women", "COS tailored trousers women",
            "Zara pleated midi skirt women", "Aritzia wide leg pants women",
            "Mango leather effect pants women", "Reformation high rise jeans women",
            "H&M premium satin skirt women", "AllSaints cargo pants women",
            "Princess Polly mini skirt women", "Edikted low rise pants women",
            "White Fox cargo pants women", "Abercrombie linen pants women",
        ],
        "outerwear": [
            "COS wool coat women", "Aritzia superpuff jacket women",
            "Zara oversized blazer women", "Mango leather jacket women",
            "AllSaints biker jacket women", "H&M premium trench coat women",
            "Princess Polly puffer jacket women", "Meshki blazer women",
            "Edikted faux fur jacket women", "Alo Yoga hoodie women",
            "Abercrombie bomber jacket women",
        ],
        "shoes": [
            "Adidas Samba sneakers women", "Mango leather loafers women",
            "Zara strappy heeled sandal women", "Steve Madden platform boots women",
            "Sam Edelman pointed toe flat women", "COS leather ankle boots women",
            "New Balance 550 women", "Princess Polly platform heels women",
            "Meshki heeled sandals women",
        ],
        "bags": [
            "Polene leather bag women", "Mansur Gavriel bucket bag women",
            "COS quilted crossbody bag women", "Mango leather tote bag women",
            "Zara minimalist shoulder bag women", "Coach tabby bag women",
            "JW Pei shoulder bag women", "Meshki mini bag women",
        ],
        "accessories": [
            "Mejuri gold layered necklace women", "Missoma pearl earrings women",
            "COS silk scarf women", "Mango oversized sunglasses women",
            "Zara chain link necklace women", "Mango leather belt women",
            "Skims accessories women", "Oh Polly jewelry women",
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

    print("Enriching catalog metadata (item_type, colors, occasions)...")
    enrich_catalog_metadata(catalog)

    type_counts: dict[str, int] = {}
    for item in catalog:
        t = item.get("item_type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Item types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

    print(f"Building FAISS index for {len(catalog)} items...")
    index = build_index(catalog)
    save_index(index, catalog, base_path=args.output_dir)
    print(f"Saved index and catalog to {args.output_dir}/")
    print(f"  Items: {len(catalog)}")
    print(f"  Slots: {set(item['slot'] for item in catalog)}")


if __name__ == "__main__":
    main()
