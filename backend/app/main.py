import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import get_settings
from app.api.routes import taste, recommendations, catalog, wardrobe, events, shopper

settings = get_settings()

app = FastAPI(
    title="Wardrobe IQ API",
    description="Taste-aware outfit utility engine for Phia",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(taste.router)
app.include_router(recommendations.router)
app.include_router(catalog.router)
app.include_router(wardrobe.router)
app.include_router(events.router)
app.include_router(shopper.router)


images_dir = Path("data/images")
if images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="product-images")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
