from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from io import BytesIO
from uuid import uuid4
import numpy as np

from app.core.taste import extract_taste_profile, update_taste_profile, extract_attributes
from app.core.trends import compute_trend_fingerprint
from app.core.candidates import load_index
from app.data.pinterest import scrape_board
from app.db.models import TasteUpdateRequest, TasteUpdateResponse

router = APIRouter(prefix="/api/taste", tags=["taste"])


@router.post("/extract")
async def extract_taste(
    pinterest_url: str = Form(None),
    images: list[UploadFile] = File(None),
):
    """
    Extract a taste profile from Pinterest board URL or uploaded images.

    Accepts multipart form with either:
      - pinterest_url: public Pinterest board URL
      - images: up to 10 image files
    """
    image_data: list[bytes] = []

    if pinterest_url:
        try:
            image_data = await scrape_board(pinterest_url)
            source_type = "pinterest"
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif images:
        if len(images) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed.")

        for upload in images:
            content = await upload.read()
            try:
                img = Image.open(BytesIO(content))
                img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG")
                image_data.append(buf.getvalue())
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not process image: {upload.filename}",
                )
        source_type = "upload"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a Pinterest URL or upload images.",
        )

    profile = extract_taste_profile(image_data, source_type=source_type)
    user_id = str(uuid4())

    return {
        "user_id": user_id,
        "taste_vector": profile["taste_vector"],
        "taste_modes": profile["taste_modes"],
        "occasion_vectors": profile["occasion_vectors"],
        "trend_fingerprint": profile["trend_fingerprint"],
        "anti_taste_vector": profile["anti_taste_vector"],
        "aesthetic_attributes": profile["aesthetic_attributes"],
        "price_tier": profile["price_tier"],
    }


@router.post("/update", response_model=TasteUpdateResponse)
async def update_taste(req: TasteUpdateRequest):
    """
    Update the taste vector via EMA when a user saves an item.

    The frontend sends the current taste vector, the saved item's id,
    and the total save count so the blending weight decays over time.
    """
    taste_vector = np.array(req.taste_vector, dtype=np.float32)
    if len(taste_vector) == 0:
        raise HTTPException(status_code=400, detail="taste_vector is required")

    try:
        _, catalog = load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Catalog not built yet")

    item = next((i for i in catalog if i["item_id"] == req.item_id), None)
    if not item or "embedding" not in item:
        raise HTTPException(status_code=404, detail="Item not found in catalog")

    item_embedding = np.array(item["embedding"], dtype=np.float32)

    new_vector = update_taste_profile(
        existing_vector=taste_vector,
        new_item_embedding=item_embedding,
        save_count=req.save_count,
    )

    attrs = extract_attributes(new_vector)
    trend_fp = compute_trend_fingerprint(new_vector)
    price_tier = [float(item.get("price", 40.0) * 0.6), float(item.get("price", 200.0) * 1.4)]

    return TasteUpdateResponse(
        taste_vector=new_vector.tolist(),
        trend_fingerprint=trend_fp,
        aesthetic_attributes=attrs,
        price_tier=price_tier,
    )
