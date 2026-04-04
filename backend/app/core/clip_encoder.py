"""
Fashion-specific CLIP encoder using FashionCLIP 2.0.

Fine-tuned from CLIP ViT-B/32 on 800K Farfetch fashion product
image-text pairs.  Understands silhouettes, fabrics, construction
details, and style vocabulary far better than generic CLIP.
Produces 512-d L2-normalised embeddings.
"""

import numpy as np
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor
from app.config import get_settings

_encoder: "CLIPEncoder | None" = None

_MODEL_ID = "patrickjohncyh/fashion-clip"


class CLIPEncoder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(_MODEL_ID).to(device)
        self.processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim  # 512

    def _load_image(self, source: str | bytes | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            return Image.open(BytesIO(source)).convert("RGB")
        return Image.open(source).convert("RGB")

    def encode_images(
        self, sources: list[str | bytes | Image.Image]
    ) -> np.ndarray:
        images = [self._load_image(s) for s in sources]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            features = self.model.visual_projection(vision_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            text_out = self.model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            features = self.model.text_projection(text_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def similarity(
        self, image_features: np.ndarray, text_features: np.ndarray
    ) -> np.ndarray:
        return image_features @ text_features.T


def get_encoder() -> CLIPEncoder:
    global _encoder
    if _encoder is None:
        settings = get_settings()
        _encoder = CLIPEncoder(device=settings.clip_device)
    return _encoder
