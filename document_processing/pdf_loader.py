"""
Enterprise PDF Loader

Features
--------
✓ Large PDF support
✓ Streaming page processing
✓ Native PDF
✓ Scanned PDF
✓ Mixed PDF
✓ Image extraction
✓ Memory efficient
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from core.constants import PageType
from config import OCR_MIN_TEXT_THRESHOLD
import fitz

from config import (
    DOCUMENTS_DIR,
    PDF_BATCH_PAGES,
)

from core.logger import get_logger
from core.utils import (
    calculate_file_hash,
    ensure_directory,
    get_file_size,
    sanitize_filename,
)

logger = get_logger(__name__)


class PDFLoader:

    """
    Enterprise PDF Loader
    """

    def __init__(self):

        ensure_directory(
            DOCUMENTS_DIR
        )

        # ----------------------------------------------------------
    # Open
    # ----------------------------------------------------------

    def open(
        self,
        pdf_path: str | Path,
    ) -> fitz.Document:

        pdf_path = Path(pdf_path)

        logger.info(
            "Opening %s",
            pdf_path.name,
        )

        return fitz.open(pdf_path)

        # ----------------------------------------------------------
    # Pages
    # ----------------------------------------------------------

    def total_pages(
        self,
        document: fitz.Document,
    ) -> int:

        return document.page_count

        # ----------------------------------------------------------
    # Hash
    # ----------------------------------------------------------

    def file_hash(
        self,
        pdf_path,
    ):

        return calculate_file_hash(
            pdf_path
        )

        # ----------------------------------------------------------
    # Size
    # ----------------------------------------------------------

    def file_size(
        self,
        pdf_path,
    ):

        return get_file_size(
            pdf_path
        )

        # ----------------------------------------------------------
    # Store
    # ----------------------------------------------------------

    def store_pdf(
        self,
        uploaded_file,
    ) -> Path:

        filename = sanitize_filename(
            uploaded_file.name
        )

        destination = (
            DOCUMENTS_DIR /
            filename
        )

        with open(
            destination,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        logger.info(
            "Stored %s",
            filename,
        )

        return destination

        # ----------------------------------------------------------
    # Streaming
    # ----------------------------------------------------------

    def stream_pages(
        self,
        document: fitz.Document,
    ) -> Generator:

        for page_number in range(
            document.page_count
        ):

            yield (
                page_number,
                document.load_page(
                    page_number
                ),
            )

        # ----------------------------------------------------------
    # Batch
    # ----------------------------------------------------------

    def stream_page_batches(
        self,
        document: fitz.Document,
    ):

        total = document.page_count

        start = 0

        while start < total:

            end = min(
                start + PDF_BATCH_PAGES,
                total,
            )

            pages = []

            for page_no in range(
                start,
                end,
            ):

                pages.append(
                    (
                        page_no,
                        document.load_page(
                            page_no
                        ),
                    )
                )

            yield pages

            pages.clear()

            start = end


        # ----------------------------------------------------------
    # Close
    # ----------------------------------------------------------

    def close(
        self,
        document,
    ):

        document.close()

        # ----------------------------------------------------------
    # Native Text
    # ----------------------------------------------------------

    def extract_native_text(
        self,
        page: fitz.Page,
    ) -> str:
        """
        Extract text directly from PDF.
        """

        try:

            text = page.get_text(
                "text"
            )

            return text.strip()

        except Exception as e:

            logger.exception(e)

            return ""


        # ----------------------------------------------------------
    # Page Classification
    # ----------------------------------------------------------

    def classify_page(
        self,
        page: fitz.Page,
    ) -> PageType:
        """
        Detect page type.

        TEXT
        SCANNED
        MIXED
        EMPTY
        """

        text = self.extract_native_text(
            page
        )

        images = page.get_images(
            full=True
        )

        text_length = len(text)

        if (
            text_length == 0
            and len(images) == 0
        ):
            return PageType.EMPTY

        if (
            text_length >=
            OCR_MIN_TEXT_THRESHOLD
            and len(images) == 0
        ):
            return PageType.TEXT

        if (
            text_length <
            OCR_MIN_TEXT_THRESHOLD
            and len(images) > 0
        ):
            return PageType.SCANNED

        return PageType.MIXED

        # ----------------------------------------------------------
    # OCR Decision
    # ----------------------------------------------------------

    def requires_ocr(
        self,
        page: fitz.Page,
    ) -> bool:

        page_type = self.classify_page(
            page
        )

        return page_type in {

            PageType.SCANNED,

            PageType.MIXED,

        }

        # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    def page_metadata(
        self,
        page_number: int,
        page: fitz.Page,
        file_hash: str,
        file_name: str,
    ):

        return {

            "page": page_number,

            "file_hash": file_hash,

            "file_name": file_name,

            "page_type":
                self.classify_page(
                    page
                ).value,

            "rotation":
                page.rotation,

            "width":
                page.rect.width,

            "height":
                page.rect.height,
        }

        # ----------------------------------------------------------
    # Analyze
    # ----------------------------------------------------------

    def analyze_document(
        self,
        document: fitz.Document,
    ):

        statistics = {

            "pages": 0,

            "text": 0,

            "scanned": 0,

            "mixed": 0,

            "empty": 0,
        }

        for _, page in self.stream_pages(
            document
        ):

            statistics["pages"] += 1

            page_type = self.classify_page(
                page
            )

            if page_type == PageType.TEXT:

                statistics["text"] += 1

            elif page_type == PageType.SCANNED:

                statistics["scanned"] += 1

            elif page_type == PageType.MIXED:

                statistics["mixed"] += 1

            else:

                statistics["empty"] += 1

        return statistics

        # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def print_summary(
        self,
        stats: dict,
    ):

        logger.info(

            "Pages=%d | Text=%d | "
            "Scanned=%d | Mixed=%d | "
            "Empty=%d",

            stats["pages"],

            stats["text"],

            stats["scanned"],

            stats["mixed"],

            stats["empty"],
        )