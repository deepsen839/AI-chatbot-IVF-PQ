"""
Enterprise Text Extractor

Features
--------
✓ Reading order preservation
✓ Block extraction
✓ Word extraction
✓ Table-friendly extraction
✓ Metadata generation
✓ Memory efficient
"""

from __future__ import annotations

from typing import List

import fitz

from core.logger import get_logger

logger = get_logger(__name__)

class TextExtractor:

    """
    Enterprise Text Extraction
    """

    def __init__(self):

        pass

        # ---------------------------------------------------------
    # Plain Text
    # ---------------------------------------------------------

    def extract_text(
        self,
        page: fitz.Page,
    ) -> str:

        try:

            text = page.get_text(
                "text",
                sort=True,
            )

            return text.strip()

        except Exception as e:

            logger.exception(e)

            return ""

        # ---------------------------------------------------------
    # Blocks
    # ---------------------------------------------------------

    def extract_blocks(
        self,
        page: fitz.Page,
    ):

        try:

            return page.get_text(
                "blocks",
                sort=True,
            )

        except Exception as e:

            logger.exception(e)

            return []

        # ---------------------------------------------------------
    # Words
    # ---------------------------------------------------------

    def extract_words(
        self,
        page: fitz.Page,
    ):

        try:

            return page.get_text(
                "words",
                sort=True,
            )

        except Exception as e:

            logger.exception(e)

            return []

        # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------

    def extract_dict(
        self,
        page: fitz.Page,
    ):

        try:

            return page.get_text(
                "dict",
                sort=True,
            )

        except Exception as e:

            logger.exception(e)

            return {}

        # ---------------------------------------------------------
    # HTML
    # ---------------------------------------------------------

    def extract_html(
        self,
        page,
    ):

        return page.get_text(
            "html",
        )

        # ---------------------------------------------------------
    # Markdown
    # ---------------------------------------------------------

    def extract_markdown(
        self,
        page,
    ):

        try:

            return page.get_text(
                "markdown"
            )

        except Exception:

            return self.extract_text(
                page
            )

        # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def metadata(
        self,
        page,
    ):

        text = self.extract_text(
            page
        )

        words = self.extract_words(
            page
        )

        return {

            "characters": len(text),

            "words": len(words),

            "blocks": len(
                self.extract_blocks(
                    page
                )
            ),

            "rotation":
                page.rotation,

            "width":
                page.rect.width,

            "height":
                page.rect.height,
        }

        # ---------------------------------------------------------
    # Text Exists
    # ---------------------------------------------------------

    def has_text(
        self,
        page,
    ):

        return (
            len(
                self.extract_text(
                    page
                )
            ) > 0
        )