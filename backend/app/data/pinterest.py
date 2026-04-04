"""
Pinterest board scraper.

Uses pinterest-export (Playwright headless browser) to extract images
from public Pinterest boards. Falls back gracefully on failure.
"""

import re
import logging
import httpx
from io import BytesIO
from PIL import Image
from pinterest_export.scraper import scrape_board as _scrape_pins

logger = logging.getLogger(__name__)


def _upscale_url(url: str) -> str:
    """Upgrade Pinterest thumbnail URLs to higher resolution.

    Pinterest serves images at /236x/ by default from board views.
    Replace with /736x/ for better quality while keeping download fast.
    """
    return re.sub(r"/\d+x/", "/736x/", url)


def _normalize_board_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    url = url.rstrip("/")
    if not re.search(r"pinterest\.(com|co\.\w+|ca|fr|de|it|es|jp)", url):
        raise ValueError("That doesn't look like a Pinterest URL.")
    return url


async def scrape_board(board_url: str, max_images: int = 25) -> list[bytes]:
    """
    Scrape image data from a public Pinterest board URL.

    Uses a headless browser to render Pinterest's JS-heavy pages,
    then downloads and preprocesses each image for CLIP.

    Returns list of JPEG image bytes sized to 224x224.
    Raises ValueError on failure.
    """
    clean_url = _normalize_board_url(board_url)

    try:
        pins = await _scrape_pins(clean_url, limit=max_images)
    except Exception as e:
        logger.error("Pinterest scrape failed for %s: %s", clean_url, e)
        raise ValueError(
            f"Couldn't read that Pinterest board: {e}. "
            "Make sure the board is public and the URL is correct."
        )

    if not pins:
        raise ValueError(
            "No pins found on this board. "
            "The board may be empty, private, or the URL may be incorrect."
        )

    image_urls = [_upscale_url(pin.image_url) for pin in pins if pin.image_url]
    logger.info("Found %d pin image URLs from %s", len(image_urls), clean_url)

    images: list[bytes] = []
    async with httpx.AsyncClient(timeout=15) as client:
        for url in image_urls:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue
                img = Image.open(BytesIO(resp.content))
                img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG")
                images.append(buf.getvalue())
            except Exception:
                continue

            if len(images) >= max_images:
                break

    if not images:
        raise ValueError(
            "Found pins but couldn't download any images. "
            "Try again or upload images directly."
        )

    logger.info("Successfully downloaded %d images from Pinterest", len(images))
    return images
