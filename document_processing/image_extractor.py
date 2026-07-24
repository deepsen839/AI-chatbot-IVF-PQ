"""
Enterprise Image Extractor

Features
--------
✓ Extract embedded images
✓ Skip tiny icons
✓ Remove duplicates
✓ Save deterministic filenames
✓ Image metadata
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
from PIL import Image

from config import (
    IMAGE_DIR,
    IMAGE_FORMAT,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)

from core.logger import get_logger
from core.utils import ensure_directory

logger = get_logger(__name__)

class ImageExtractor:

    def __init__(self):

        ensure_directory(
            IMAGE_DIR
        )

        self.hash_cache = set()

        # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    def extract_images(
        self,
        page: fitz.Page,
    ):

        return page.get_images(
            full=True
        )

        # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    def image_count(
        self,
        page: fitz.Page,
    ):

        return len(
            self.extract_images(
                page
            )
        )

        # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    def image_hash(
        self,
        image_bytes: bytes,
    ):

        return hashlib.sha256(
            image_bytes
        ).hexdigest()

        # --------------------------------------------------------
    # Duplicate
    # --------------------------------------------------------

    def is_duplicate(
        self,
        image_bytes: bytes,
    ):

        image_hash = self.image_hash(
            image_bytes
        )

        if image_hash in self.hash_cache:

            return True

        self.hash_cache.add(
            image_hash
        )

        return False

        # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_image(
        self,
        image: Image.Image,
        filename: str,
    ) -> Path:

        path = (
            IMAGE_DIR /
            filename
        )

        image.save(
            path,
            IMAGE_FORMAT.upper(),
        )

        return path

        # --------------------------------------------------------
    # Pixmap
    # --------------------------------------------------------

    def pixmap_to_image(
        self,
        pixmap: fitz.Pixmap,
    ) -> Image.Image:

        if pixmap.alpha:

            pixmap = fitz.Pixmap(
                fitz.csRGB,
                pixmap,
            )

        image = Image.frombytes(

            "RGB",

            [pixmap.width, pixmap.height],

            pixmap.samples,
        )

        return image


        # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    def valid_size(
        self,
        image: Image.Image,
    ):

        return (

            image.width >=
            MIN_IMAGE_WIDTH

            and

            image.height >=
            MIN_IMAGE_HEIGHT

        )

        # --------------------------------------------------------
    # Extract Single
    # --------------------------------------------------------

    def extract_image(
        self,
        document: fitz.Document,
        xref: int,
    ):

        base = document.extract_image(
            xref
        )

        return (

            base["image"],

            base["ext"],
        )

        # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def metadata(
        self,
        image: Image.Image,
        path: Path,
    ):

        return {

            "path": str(path),

            "width": image.width,

            "height": image.height,

            "mode": image.mode,

            "size": path.stat().st_size,
        }