"""
Enterprise Document Chunker

Features
--------
✓ Structure-aware chunking
✓ Paragraph preservation
✓ Configurable chunk size
✓ Configurable overlap
✓ Metadata-rich chunks
✓ Streaming support
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List
import re
from typing import Iterable
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from core.logger import get_logger

logger = get_logger(__name__)

@dataclass(slots=True)
class DocumentChunk:
    """
    A single chunk ready for embedding.
    """

    chunk_id: str

    text: str

    file_hash: str

    file_name: str

    page_number: int

    page_type: str

    chunk_index: int

    metadata: dict


class DocumentChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

        # --------------------------------------------------------
    # Split Text
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Enhanced Split
    # --------------------------------------------------------

    def split_text(
        self,
        text: str,
    ) -> list[str]:

        if not text.strip():
            return []

        text = self.merge_headings(text)

        chunks = self.splitter.split_text(
            text
        )

        return self.clean_chunks(
            chunks
        )
        # --------------------------------------------------------
    # Page Chunking
    # --------------------------------------------------------

    def chunk_page(
        self,
        page_text: str,
        page_metadata: dict,
    ) -> Generator[DocumentChunk, None, None]:

        chunks = self.split_text(
            page_text
        )

        for index, chunk in enumerate(chunks):

            metadata = page_metadata.copy()

            metadata.update({

                "character_count": len(chunk),

                "estimated_tokens": self.estimate_tokens(chunk),

                "has_table": "|" in chunk,

                "has_number": bool(
                    re.search(r"\d", chunk)
                ),

                "has_url": (
                    "http://" in chunk
                    or
                    "https://" in chunk
                ),

            })

            yield DocumentChunk(

                chunk_id=(
                    f"{page_metadata['file_hash']}"
                    f"_{page_metadata['page']}"
                    f"_{index}"
                ),

                text=chunk,

                file_hash=page_metadata["file_hash"],

                file_name=page_metadata["file_name"],

                page_number=page_metadata["page"],

                page_type=page_metadata["page_type"],

                chunk_index=index,

                metadata=metadata,
            )

        # --------------------------------------------------------
    # Stream Chunking
    # --------------------------------------------------------

    def chunk_pages(
        self,
        processed_pages,
    ) -> Generator[DocumentChunk, None, None]:

        for page in processed_pages:

            yield from self.chunk_page(

                page_text=page["text"],

                page_metadata=page["metadata"],
            )

        # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def chunk_count(
        self,
        text: str,
    ) -> int:

        return len(
            self.split_text(text)
        )

        # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def valid_chunk(
        self,
        text: str,
    ) -> bool:

        return bool(
            text and text.strip()
        )

        # --------------------------------------------------------
    # Token Estimation
    # --------------------------------------------------------

    def estimate_tokens(
        self,
        text: str,
    ) -> int:

        if not text:
            return 0

        # Approximation: ~4 characters/token for English
        return max(1, len(text) // 4)

        # --------------------------------------------------------
    # Heading Detection
    # --------------------------------------------------------

    def is_heading(
        self,
        line: str,
    ) -> bool:

        line = line.strip()

        if not line:
            return False

        if len(line) > 120:
            return False

        if line.endswith(":"):
            return True

        if re.match(r"^\d+(\.\d+)*\s+", line):
            return True

        words = line.split()

        if len(words) <= 10 and line == line.title():
            return True

        return False

        # --------------------------------------------------------
    # Heading Merge
    # --------------------------------------------------------

    def merge_headings(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()

        merged = []

        index = 0

        while index < len(lines):

            current = lines[index].strip()

            if (
                self.is_heading(current)
                and index + 1 < len(lines)
            ):

                merged.append(
                    current
                    + "\n"
                    + lines[index + 1].strip()
                )

                index += 2

                continue

            merged.append(current)

            index += 1

        return "\n".join(merged)

        # --------------------------------------------------------
    # Chunk Cleanup
    # --------------------------------------------------------

    def clean_chunks(
        self,
        chunks: Iterable[str],
    ) -> list[str]:

        cleaned = []

        seen = set()

        for chunk in chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            key = chunk.lower()

            if key in seen:
                continue

            seen.add(key)

            cleaned.append(chunk)

        return cleaned


        # --------------------------------------------------------
    # Minimum Size Filter
    # --------------------------------------------------------

    def should_embed(
        self,
        chunk: str,
    ) -> bool:

        return len(chunk.strip()) >= 30