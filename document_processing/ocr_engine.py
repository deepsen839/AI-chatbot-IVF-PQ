"""
Enterprise OCR Engine

Features
--------
✓ PaddleOCR GPU support
✓ High-resolution page rendering
✓ OCR result caching
✓ Native + OCR text merging
✓ Memory efficient
✓ Large PDF support
"""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image
from paddleocr import PaddleOCR

from config import (
    OCR_LANGUAGE,
    OCR_DPI,
    OCR_USE_GPU,
    OCR_CACHE_DIR,
)

from core.logger import get_logger
from core.utils import ensure_directory

logger = get_logger(__name__)
import hashlib
from typing import Generator

from core.constants import PageType

class OCREngine:

    """
    Enterprise PaddleOCR Engine
    """

    def __init__(self):

        ensure_directory(
            OCR_CACHE_DIR
        )

        logger.info(
            "Initializing PaddleOCR..."
        )

        self.ocr = PaddleOCR(
            lang=OCR_LANGUAGE,
            device="gpu",
        )

        logger.info(
            "PaddleOCR initialized."
        )


        # -------------------------------------------------------
    # Render Page
    # -------------------------------------------------------

    def render_page(
        self,
        page: fitz.Page,
    ) -> Image.Image:

        zoom = OCR_DPI / 72.0

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.frombytes(

            "RGB",

            [pix.width, pix.height],

            pix.samples,
        )

        return image

        # -------------------------------------------------------
    # Cache Image
    # -------------------------------------------------------

    def save_render(
        self,
        image: Image.Image,
        filename: str,
    ) -> Path:

        path = (
            OCR_CACHE_DIR /
            filename
        )

        image.save(
            path,
            "PNG",
        )

        return path

        # -------------------------------------------------------
    # OCR
    # -------------------------------------------------------

    def image_to_text(
        self,
        image_path: Path,
    ) -> str:

        try:

            result = self.ocr.predict(
                str(image_path)
            )

        except AttributeError:
            # PaddleOCR 2.x compatibility
            result = self.ocr.ocr(
                str(image_path),
                cls=True,
            )

        lines = []

        # PaddleOCR 3.x result
        if result and hasattr(result[0], "res"):
            for block in result[0].res.get("rec_texts", []):
                lines.append(block)

        # PaddleOCR 2.x result
        elif result:
            for line in result[0]:
                if line:
                    lines.append(line[1][0])

        return "\n".join(lines)

        # -------------------------------------------------------
    # OCR Page
    # -------------------------------------------------------

    def page_to_text(
        self,
        page: fitz.Page,
        cache_name: str,
    ) -> str:

        image = self.render_page(
            page
        )

        image_path = self.save_render(
            image,
            cache_name,
        )

        return self.image_to_text(
            image_path
        )


        # -------------------------------------------------------
    # Cache Exists
    # -------------------------------------------------------

    def cached(
        self,
        filename: str,
    ):

        return (
            OCR_CACHE_DIR /
            filename
        ).exists()


        # -------------------------------------------------------
    # Load Cache
    # -------------------------------------------------------

    def load_cache(
        self,
        filename: str,
    ) -> str:

        path = (
            OCR_CACHE_DIR /
            filename
        )

        txt = path.with_suffix(
            ".txt"
        )

        if txt.exists():

            return txt.read_text(
                encoding="utf-8"
            )

        return ""

        # -------------------------------------------------------
    # Save Cache
    # -------------------------------------------------------

    def save_cache(
        self,
        filename: str,
        text: str,
    ):

        path = (
            OCR_CACHE_DIR /
            filename
        ).with_suffix(".txt")

        path.write_text(
            text,
            encoding="utf-8",
        )

        # -------------------------------------------------------
    # Cache Key
    # -------------------------------------------------------

    def cache_key(
        self,
        file_hash: str,
        page_number: int,
    ) -> str:

        return hashlib.sha256(
            f"{file_hash}:{page_number}".encode("utf-8")
        ).hexdigest()


        # -------------------------------------------------------
    # OCR With Cache
    # -------------------------------------------------------

    def ocr_page(
        self,
        page: fitz.Page,
        file_hash: str,
        page_number: int,
    ) -> str:

        key = self.cache_key(
            file_hash,
            page_number,
        )

        if self.load_cache(key):

            logger.debug(
                "OCR cache hit: %s",
                page_number,
            )

            return self.load_cache(key)

        logger.info(
            "Running OCR on page %d",
            page_number,
        )

        text = self.page_to_text(
            page,
            key + ".png",
        )

        self.save_cache(
            key,
            text,
        )

        return text


        # -------------------------------------------------------
    # Merge
    # -------------------------------------------------------

    def merge_text(
        self,
        native_text: str,
        ocr_text: str,
    ) -> str:
        """
        Merge native PDF text and OCR text while
        removing duplicate lines.
        """

        merged = []

        seen = set()

        for line in (
            native_text.splitlines()
            + ocr_text.splitlines()
        ):

            cleaned = line.strip()

            if not cleaned:

                continue

            key = cleaned.lower()

            if key in seen:

                continue

            seen.add(key)

            merged.append(cleaned)

        return "\n".join(merged)


        # -------------------------------------------------------
    # Process Page
    # -------------------------------------------------------

    def process_page(
        self,
        page: fitz.Page,
        page_type: PageType,
        native_text: str,
        file_hash: str,
        page_number: int,
    ) -> str:

        if page_type == PageType.TEXT:

            return native_text

        ocr_text = self.ocr_page(
            page,
            file_hash,
            page_number,
        )

        if page_type == PageType.SCANNED:

            return ocr_text

        if page_type == PageType.MIXED:

            return self.merge_text(
                native_text,
                ocr_text,
            )

        return ""


        # -------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------

    def process_pages(
        self,
        pages,
        pdf_loader,
        file_hash: str,
    ) -> Generator[dict, None, None]:

        for page_number, page in pages:

            page_type = pdf_loader.classify_page(
                page
            )

            native_text = pdf_loader.extract_native_text(
                page
            )

            text = self.process_page(
                page=page,
                page_type=page_type,
                native_text=native_text,
                file_hash=file_hash,
                page_number=page_number,
            )

            yield {
                "page_number": page_number,
                "page_type": page_type,
                "text": text,
            }


        # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def cleanup_render(
        self,
        filename: str,
    ):

        image_path = (
            OCR_CACHE_DIR /
            filename
        )

        if image_path.exists():

            image_path.unlink()


        # -------------------------------------------------------
    # Close
    # -------------------------------------------------------

    def close(self):

        logger.info(
            "OCR Engine closed."
        )