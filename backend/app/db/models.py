from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4


class TasteProfile(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    taste_vector: list[float] = Field(default_factory=list)
    aesthetic_attributes: dict = Field(default_factory=dict)
    price_tier_low: float = 0.0
    price_tier_high: float = 0.0
    source: str = "upload"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WardrobeSave(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    item_id: str
    item_data: dict = Field(default_factory=dict)
    saved_at: datetime = Field(default_factory=datetime.now)


class CatalogItem(BaseModel):
    item_id: str
    title: str
    brand: str
    category: str
    slot: str
    price: float
    image_url: str
    embedding: list[float] = Field(default_factory=list)
    source: str = "mock"
    cached_at: datetime = Field(default_factory=datetime.now)


class RecommendationEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    event_type: str  # impression | click | save
    module: str
    item_id: str
    score: float = 0.0
    unlock_count: int = 0
    taste_score: float = 0.0
    model_version: str = "v0.1"
    timestamp: datetime = Field(default_factory=datetime.now)


# --- API Request/Response Models ---

class TasteExtractRequest(BaseModel):
    pinterest_url: str | None = None
    images: list[str] | None = None  # base64-encoded images

class TasteExtractResponse(BaseModel):
    user_id: str
    taste_vector: list[float]
    aesthetic_attributes: dict
    price_tier: list[float]

class WardrobeRecommendationRequest(BaseModel):
    user_id: str
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    taste_vector: list[float] = Field(default_factory=list)
    taste_modes: list[list[float]] = Field(default_factory=list)
    occasion_vectors: dict[str, list[float]] = Field(default_factory=dict)
    trend_fingerprint: dict[str, float] = Field(default_factory=dict)
    anti_taste_vector: list[float] = Field(default_factory=list)
    price_tier: list[float] = Field(default_factory=lambda: [40.0, 200.0])
    aesthetic_label: str = ""
    skipped_item_ids: list[str] = Field(default_factory=list)
    intent_vector: list[float] | None = None
    intent_confidence: float = 0.0

class GapRecommendation(BaseModel):
    item: dict
    unlock_count: int
    taste_score: float
    explanation: str
    confidence: float

class ScoredItem(BaseModel):
    item: dict
    taste_score: float
    unlock_count: int = 0
    explanation: str = ""

class OccasionSection(BaseModel):
    occasion: str
    label: str
    items: list[ScoredItem] = Field(default_factory=list)


class OutfitBundle(BaseModel):
    label: str
    items: list[dict] = Field(default_factory=list)


class WardrobeRecommendationResponse(BaseModel):
    gap_recommendation: GapRecommendation | None = None
    top_picks: list[ScoredItem] = Field(default_factory=list)
    occasion_sections: list[OccasionSection] = Field(default_factory=list)
    outfit_suggestions: list[OutfitBundle] = Field(default_factory=list)
    shopping_brief: dict = Field(default_factory=dict)
    complete_the_look: dict | None = None
    wardrobe_stats: dict = Field(default_factory=dict)
    model_version: str = "v0.2"

class EvaluateItemRequest(BaseModel):
    product_url: str | None = None
    item_id: str | None = None
    user_id: str
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    taste_vector: list[float] = Field(default_factory=list)

class EvaluateItemResponse(BaseModel):
    taste_fit: float
    unlock_count: int
    pairs_with: list[dict]
    explanation: str
    confidence: float

class SaveItemRequest(BaseModel):
    user_id: str
    item_id: str

class TasteUpdateRequest(BaseModel):
    user_id: str
    taste_vector: list[float]
    item_id: str
    save_count: int
    style_attributes: dict[str, float] = Field(default_factory=dict)

class TasteUpdateResponse(BaseModel):
    taste_vector: list[float]
    trend_fingerprint: dict[str, float] = Field(default_factory=dict)
    display_trends: dict[str, float] = Field(default_factory=dict)
    aesthetic_attributes: dict
    price_tier: list[float]
    style_attributes: dict[str, float] = Field(default_factory=dict)
    style_summary: list[dict] = Field(default_factory=list)

class TasteDismissRequest(BaseModel):
    item_id: str
    style_attributes: dict[str, float] = Field(default_factory=dict)
    dismiss_count: int = 1

class TasteDismissResponse(BaseModel):
    style_attributes: dict[str, float]
    style_summary: list[dict]

class EventLogRequest(BaseModel):
    user_id: str
    event_type: str  # impression | click | save | dismiss | skip
    module: str
    item_id: str
    score: float = 0.0
    unlock_count: int = 0
    taste_score: float = 0.0


class ShopperPlan(BaseModel):
    occasion: str = "casual"
    slots_to_fill: list[str] = Field(default_factory=list)
    tone: str = ""
    max_price: float | None = None


class ShopperPlanRequest(BaseModel):
    user_brief_json: dict = Field(default_factory=dict)
    user_message: str | None = None
    taste_vector: list[float] = Field(default_factory=list)
    occasion_vectors: dict[str, list[float]] = Field(default_factory=dict)
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    trend_fingerprint: dict[str, float] = Field(default_factory=dict)
    anti_taste_vector: list[float] = Field(default_factory=list)
    price_tier: list[float] = Field(default_factory=lambda: [40.0, 200.0])


class ShopperPlanResponse(BaseModel):
    plan: ShopperPlan
    items: list[dict] = Field(default_factory=list)


# --- Intent Models ---

class IntentComputeRequest(BaseModel):
    viewed_embeddings: list[list[float]] = Field(default_factory=list)


class IntentComputeResponse(BaseModel):
    intent_vector: list[float] | None = None
    confidence: float = 0.0
    num_views: int = 0
    session_labels: list[str] = Field(default_factory=list)


# --- Feed Models ---

class FeedRequest(BaseModel):
    user_id: str
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    taste_vector: list[float] = Field(default_factory=list)
    taste_modes: list[list[float]] = Field(default_factory=list)
    occasion_vectors: dict[str, list[float]] = Field(default_factory=dict)
    trend_fingerprint: dict[str, float] = Field(default_factory=dict)
    anti_taste_vector: list[float] = Field(default_factory=list)
    price_tier: list[float] = Field(default_factory=lambda: [40.0, 200.0])
    aesthetic_label: str = ""
    skipped_item_ids: list[str] = Field(default_factory=list)
    intent_vector: list[float] | None = None
    intent_confidence: float = 0.0
    style_attributes: dict[str, float] = Field(default_factory=dict)


class FeedSection(BaseModel):
    section_type: str
    title: str
    items: list[dict] = Field(default_factory=list)


class FeedResponse(BaseModel):
    completeYourCloset: list[dict] = Field(default_factory=list)
    yourAesthetic: list[dict] = Field(default_factory=list)
    completeYourOutfits: list[dict] = Field(default_factory=list)
    bestPricesOnSaves: list[dict] = Field(default_factory=list)
    occasionRows: list[dict] = Field(default_factory=list)
    wardrobeStats: dict = Field(default_factory=dict)


# --- Chat Models ---

class ChatRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    taste_vector: list[float] = Field(default_factory=list)
    taste_modes: list[list[float]] = Field(default_factory=list)
    occasion_vectors: dict[str, list[float]] = Field(default_factory=dict)
    trend_fingerprint: dict[str, float] = Field(default_factory=dict)
    anti_taste_vector: list[float] = Field(default_factory=list)
    style_attributes: dict[str, float] = Field(default_factory=dict)
    price_tier: list[float] = Field(default_factory=lambda: [40.0, 200.0])
    aesthetic_attributes: dict = Field(default_factory=dict)


# --- Evaluate Item V2 (with intent + purchase confidence) ---

class EvaluateItemV2Request(BaseModel):
    item_id: str
    user_id: str
    wardrobe_item_ids: list[str] = Field(default_factory=list)
    taste_vector: list[float] = Field(default_factory=list)
    intent_vector: list[float] | None = None
    intent_confidence: float = 0.0


class EvaluateItemV2Response(BaseModel):
    taste_fit: float
    intent_match: float | None = None
    purchase_confidence: str = "LOW"  # HIGH / MEDIUM / LOW
    unlock_count: int = 0
    pairs_with: list[dict] = Field(default_factory=list)
    explanation: str = ""
    best_price: float = 0.0
