"""
Mock product catalog for Wardrobe IQ demo.

Uses text-based CLIP embeddings as a proxy when real product images aren't
available. Each item is described with enough detail that CLIP text embeddings
produce meaningful similarity neighborhoods.
"""

MOCK_CATALOG: list[dict] = []

_BRANDS = {
    "tops": ["COS", "Everlane", "ZARA", "Aritzia", "Reformation", "& Other Stories", "Mango", "Uniqlo"],
    "bottoms": ["AGOLDE", "Citizens of Humanity", "ZARA", "Aritzia", "Levi's", "COS", "Mango", "Reformation"],
    "outerwear": ["COS", "Aritzia", "AllSaints", "ZARA", "Mango", "Arket", "Toteme", "The Frankie Shop"],
    "shoes": ["Adidas", "New Balance", "Mango", "ZARA", "Steve Madden", "Sam Edelman", "Veja", "Birkenstock"],
    "bags": ["Mango", "ZARA", "COS", "A.P.C.", "Polene", "Coach", "Tory Burch", "Jacquemus"],
    "accessories": ["Mejuri", "Jenny Bird", "Mango", "ZARA", "COS", "Arket", "& Other Stories", "Monica Vinader"],
}

_ITEMS = {
    "tops": [
        ("White Linen Button-Down Shirt", 68, "white linen button-down shirt relaxed fit casual", "white"),
        ("Black Ribbed Tank Top", 25, "black ribbed tank top fitted minimal", "black"),
        ("Cream Cashmere Crewneck Sweater", 148, "cream cashmere crewneck sweater soft neutral", "ivory"),
        ("Navy Breton Stripe Tee", 45, "navy and white breton stripe t-shirt classic", "navy"),
        ("Olive Silk Camisole", 78, "olive green silk camisole elegant minimal", "olive"),
        ("Grey Oversized Hoodie", 55, "grey oversized hoodie streetwear comfortable", "grey"),
        ("Beige Turtleneck Knit", 89, "beige turtleneck knit sweater warm neutral", "beige"),
        ("White Cropped Tee", 28, "white cropped t-shirt casual fitted", "white"),
        ("Black Blazer Top", 95, "black structured blazer top formal sharp", "black"),
        ("Dusty Rose Wrap Blouse", 72, "dusty rose pink wrap blouse feminine elegant", "rust"),
        ("Camel V-Neck Sweater", 85, "camel v-neck sweater classic neutral", "camel"),
        ("White Oversized Oxford", 58, "white oversized oxford shirt preppy relaxed", "white"),
        ("Black Mock Neck Top", 42, "black mock neck long sleeve top minimal", "black"),
        ("Sage Linen Camp Shirt", 62, "sage green linen camp collar shirt relaxed summer", "olive"),
        ("Ivory Satin Blouse", 88, "ivory satin blouse elegant evening", "ivory"),
        ("Charcoal Merino Polo", 75, "charcoal merino wool polo smart casual", "grey"),
        ("Stone Cotton Henley", 38, "stone cotton henley casual everyday", "beige"),
        ("Off-White Peplum Top", 52, "off-white peplum top feminine structured", "ivory"),
        ("Black Bodysuit", 35, "black sleek bodysuit minimal layering", "black"),
        ("Terracotta Linen Tank", 44, "terracotta linen tank top earthy summer", "rust"),
        ("Cream Cable Knit Vest", 68, "cream cable knit sweater vest layering classic", "ivory"),
        ("Lavender Poplin Shirt", 56, "lavender poplin button-up shirt fresh spring", "ivory"),
        ("Chocolate Mesh Top", 48, "chocolate brown mesh top sheer evening", "brown"),
        ("White Smocked Top", 46, "white smocked top feminine casual summer", "white"),
        ("Navy Silk Shell", 82, "navy silk shell top elegant minimal office", "navy"),
    ],
    "bottoms": [
        ("High-Rise Straight Jeans Medium Wash", 128, "high-rise straight leg jeans medium wash denim", "navy"),
        ("Black Wide-Leg Trousers", 95, "black wide-leg tailored trousers formal elegant", "black"),
        ("Cream Linen Pants", 78, "cream linen wide-leg pants relaxed summer", "ivory"),
        ("Dark Wash Slim Jeans", 118, "dark wash slim fit jeans classic minimal", "navy"),
        ("Olive Cargo Pants", 85, "olive cargo pants relaxed streetwear", "olive"),
        ("Camel Pleated Trousers", 92, "camel pleated trousers smart casual elegant", "camel"),
        ("Grey Tailored Shorts", 65, "grey tailored shorts above knee smart casual", "grey"),
        ("White Midi Skirt", 72, "white midi skirt minimal clean", "white"),
        ("Black Leather Pants", 168, "black leather pants sleek edgy evening", "black"),
        ("Beige Chinos", 68, "beige chinos relaxed classic preppy", "beige"),
        ("Denim Midi Skirt", 75, "denim midi skirt casual everyday", "navy"),
        ("Navy Wool Trousers", 110, "navy wool tailored trousers office formal", "navy"),
        ("Brown Corduroy Pants", 82, "brown corduroy wide-leg pants retro warm", "brown"),
        ("Black Bike Shorts", 32, "black bike shorts athletic minimal", "black"),
        ("Stone Linen Shorts", 55, "stone linen shorts summer casual relaxed", "beige"),
        ("Charcoal Joggers", 58, "charcoal cotton joggers athleisure comfortable", "grey"),
        ("Ivory Satin Skirt", 88, "ivory satin midi skirt elegant evening", "ivory"),
        ("Light Wash Barrel Jeans", 135, "light wash barrel leg jeans trendy relaxed", "beige"),
        ("Black Pleated Midi Skirt", 78, "black pleated midi skirt elegant versatile", "black"),
        ("Tan Suede Mini Skirt", 92, "tan suede mini skirt western earthy", "tan"),
        ("Ecru Wide-Leg Trousers", 85, "ecru wide-leg flowing trousers minimal summer", "ivory"),
        ("Vintage Blue Mom Jeans", 108, "vintage blue mom jeans relaxed retro", "navy"),
        ("Black Tailored Bermudas", 72, "black tailored bermuda shorts smart casual", "black"),
        ("Moss Satin Pants", 95, "moss green satin pants elegant evening", "olive"),
        ("Grey Wool Flannel Trousers", 125, "grey wool flannel trousers professional classic", "grey"),
    ],
    "outerwear": [
        ("Camel Wool Coat", 245, "camel wool long coat classic elegant winter", "camel"),
        ("Black Leather Jacket", 295, "black leather biker jacket edgy cool", "black"),
        ("Cream Puffer Jacket", 165, "cream puffer jacket warm casual winter", "ivory"),
        ("Navy Trench Coat", 198, "navy trench coat classic timeless", "navy"),
        ("Olive Bomber Jacket", 135, "olive bomber jacket casual streetwear", "olive"),
        ("Beige Linen Blazer", 142, "beige linen blazer relaxed summer smart casual", "beige"),
        ("Black Oversized Blazer", 175, "black oversized blazer structured power dressing", "black"),
        ("Grey Wool Cardigan", 128, "grey wool long cardigan layering cozy", "grey"),
        ("Denim Trucker Jacket", 115, "blue denim trucker jacket casual classic", "navy"),
        ("Taupe Suede Jacket", 225, "taupe suede jacket western luxe", "tan"),
        ("Ivory Shearling Coat", 285, "ivory shearling coat winter warm luxe", "ivory"),
        ("Black Rain Jacket", 95, "black minimal rain jacket utility", "black"),
        ("Khaki Field Jacket", 148, "khaki field jacket safari utility outdoor", "beige"),
        ("Charcoal Double-Breasted Coat", 268, "charcoal double-breasted wool coat formal", "grey"),
        ("White Cropped Denim Jacket", 88, "white cropped denim jacket casual summer", "white"),
        ("Brown Quilted Vest", 108, "brown quilted vest layering outdoor", "brown"),
        ("Dusty Pink Blazer", 155, "dusty pink oversized blazer feminine modern", "rust"),
        ("Sage Linen Duster", 128, "sage green linen duster coat relaxed summer", "olive"),
        ("Houndstooth Coat", 215, "houndstooth pattern coat classic preppy", "grey"),
        ("Black Puffer Vest", 95, "black puffer vest minimal winter layering", "black"),
    ],
    "shoes": [
        ("White Leather Sneakers", 98, "white leather minimal sneakers clean everyday", "white"),
        ("Black Ankle Boots", 165, "black leather ankle boots pointed sleek", "black"),
        ("Beige Suede Loafers", 125, "beige suede loafers classic smart casual", "beige"),
        ("White Canvas Sneakers", 65, "white canvas low-top sneakers casual", "white"),
        ("Black Strappy Heeled Sandals", 118, "black strappy heeled sandals elegant evening", "black"),
        ("Brown Leather Sandals", 85, "brown leather flat sandals minimal summer", "brown"),
        ("New Balance 550", 110, "new balance 550 retro sneakers beige white", "beige"),
        ("Black Chelsea Boots", 178, "black leather chelsea boots minimal sleek", "black"),
        ("Cream Kitten Heels", 95, "cream kitten heel pumps elegant minimal", "ivory"),
        ("Birkenstock Boston Clogs", 155, "birkenstock boston suede clogs taupe casual", "tan"),
        ("Black Ballet Flats", 88, "black leather ballet flats classic feminine", "black"),
        ("Tan Knee-High Boots", 215, "tan leather knee-high boots fall equestrian", "tan"),
        ("Grey New Balance 2002R", 130, "grey new balance 2002r running sneakers", "grey"),
        ("Nude Pointed Pumps", 135, "nude pointed toe pumps formal classic", "beige"),
        ("White Platform Sneakers", 108, "white platform sneakers chunky modern", "white"),
        ("Black Mule Sandals", 92, "black leather mule sandals minimal summer", "black"),
        ("Cognac Ankle Boots", 155, "cognac leather ankle boots western warm", "brown"),
        ("Silver Metallic Sandals", 88, "silver metallic strappy sandals evening party", "grey"),
        ("Adidas Samba", 100, "adidas samba og sneakers black white retro", "black"),
        ("Cream Suede Mules", 112, "cream suede backless mules elegant relaxed", "ivory"),
    ],
    "bags": [
        ("Black Leather Tote", 195, "black leather structured tote bag everyday work", "black"),
        ("Tan Crossbody Bag", 128, "tan leather crossbody bag small casual", "tan"),
        ("Cream Canvas Tote", 65, "cream canvas tote bag casual everyday large", "ivory"),
        ("Black Mini Bag", 148, "black leather mini bag evening elegant", "black"),
        ("Brown Leather Satchel", 175, "brown leather satchel bag classic structured", "brown"),
        ("Beige Woven Bag", 88, "beige woven straw bag summer casual", "beige"),
        ("Black Nylon Shoulder Bag", 72, "black nylon shoulder bag minimal sporty", "black"),
        ("Olive Leather Bucket Bag", 155, "olive green leather bucket bag casual", "olive"),
        ("Camel Leather Clutch", 95, "camel leather clutch evening minimal", "camel"),
        ("White Quilted Bag", 168, "white quilted leather bag elegant classic", "white"),
        ("Black Belt Bag", 78, "black leather belt bag hands-free minimal", "black"),
        ("Cognac Hobo Bag", 185, "cognac leather hobo bag relaxed everyday", "brown"),
        ("Navy Structured Bag", 142, "navy leather structured top-handle bag professional", "navy"),
        ("Cream Leather Shoulder Bag", 138, "cream leather shoulder bag minimal chic", "ivory"),
        ("Tortoise Resin Bag", 115, "tortoise resin structured bag statement unique", "brown"),
    ],
    "accessories": [
        ("Gold Chain Necklace", 68, "gold chain necklace minimal delicate everyday", "camel"),
        ("Tortoise Sunglasses", 85, "tortoiseshell sunglasses classic retro", "brown"),
        ("Black Leather Belt", 55, "black leather belt classic minimal", "black"),
        ("Silk Scarf Floral", 72, "silk scarf floral print elegant", "ivory"),
        ("Gold Hoop Earrings", 48, "gold hoop earrings medium classic", "camel"),
        ("Cashmere Beanie Grey", 45, "grey cashmere beanie winter minimal", "grey"),
        ("Brown Leather Watch", 158, "brown leather strap watch classic", "brown"),
        ("Pearl Stud Earrings", 42, "pearl stud earrings classic elegant", "ivory"),
        ("Wool Scarf Camel", 65, "camel wool scarf winter warm", "camel"),
        ("Silver Cuff Bracelet", 75, "silver cuff bracelet modern minimal", "grey"),
        ("Black Cat-Eye Sunglasses", 92, "black cat-eye sunglasses retro chic", "black"),
        ("Gold Signet Ring", 55, "gold signet ring minimal everyday", "camel"),
        ("Beige Baseball Cap", 32, "beige cotton baseball cap casual sporty", "beige"),
        ("Layered Gold Necklace", 78, "layered gold chain necklace minimal modern", "camel"),
        ("Tan Leather Gloves", 68, "tan leather gloves classic winter", "tan"),
    ],
}


COLOR_VOCABULARY = [
    "black", "white", "navy", "camel", "beige", "grey", "ivory",
    "olive", "burgundy", "rust", "brown", "tan",
]


def generate_catalog() -> list[dict]:
    """Generate the full mock catalog with structured data."""
    catalog = []
    item_counter = 0

    for slot, items in _ITEMS.items():
        brands = _BRANDS[slot]
        for i, (title, price, description, dominant_color) in enumerate(items):
            brand = brands[i % len(brands)]
            catalog.append({
                "item_id": f"{slot}_{item_counter:04d}",
                "title": title,
                "brand": brand,
                "category": title.split()[-1].lower() if len(title.split()) > 1 else slot,
                "slot": slot,
                "price": float(price),
                "dominant_color": dominant_color,
                "image_url": f"https://placehold.co/400x500/f5f5f0/1a1a1a?text={title.replace(' ', '+')}",
                "clip_description": description,
                "source": "mock",
            })
            item_counter += 1

    return catalog


DEMO_WARDROBES = {
    "minimalist": {
        "name": "Minimalist",
        "description": "8 saves, strong tops/outerwear, missing shoes and bottoms.",
        "item_ids": [
            "tops_0000", "tops_0002", "tops_0006", "tops_0012",
            "outerwear_0050", "outerwear_0056",
            "bags_0090",
            "accessories_0120",
        ],
    },
    "streetwear": {
        "name": "Streetwear",
        "description": "12 saves, strong outerwear/shoes, missing versatile bottoms.",
        "item_ids": [
            "tops_0005", "tops_0007", "tops_0019",
            "outerwear_0054", "outerwear_0058", "outerwear_0064",
            "shoes_0071", "shoes_0076", "shoes_0082", "shoes_0088",
            "bags_0096",
            "accessories_0123",
        ],
    },
    "smart_casual": {
        "name": "Smart Casual",
        "description": "6 saves, balanced, missing bags/accessories.",
        "item_ids": [
            "tops_0000", "tops_0008", "tops_0015",
            "bottoms_0026", "bottoms_0030",
            "shoes_0072",
        ],
    },
    "cold_start": {
        "name": "Cold Start",
        "description": "0 saves, taste profile only (proves cold start works).",
        "item_ids": [],
    },
}
