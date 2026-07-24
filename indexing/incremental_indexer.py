"""
Enterprise Incremental Indexer

Pipeline

PDF
 ↓
Loader
 ↓
OCR
 ↓
Chunker
 ↓
Embeddings
 ↓
FAISS
 ↓
Metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.logger import get_logger

logger = get_logger(__name__)

from document_processing.pdf_loader import PDFLoader
from document_processing.ocr_engine import OCREngine
from document_processing.chunker import DocumentChunker

from embedding.embedding_model import EmbeddingModel

from database.faiss_repository import FaissRepository
from database.metadata_repository import MetadataRepository
import numpy as np

class IncrementalIndexer:

    def __init__(

        self,

        pdf_loader: PDFLoader,

        ocr_engine: OCREngine,

        chunker: DocumentChunker,

        embedding_model: EmbeddingModel,

        faiss_repository: FaissRepository,

        metadata_repository: MetadataRepository,

    ):

        self.loader = pdf_loader

        self.ocr = ocr_engine

        self.chunker = chunker

        self.embedding = embedding_model

        self.faiss = faiss_repository

        self.metadata = metadata_repository


        # ------------------------------------------------------
    # Index PDF
    # ------------------------------------------------------

    def index_pdf(
        self,
        pdf_path: Path,
    ) -> bool:
        """
        Index a single PDF incrementally.

        Returns
        -------
        bool
            True if the document was indexed.
            False if it was already indexed.
        """

        logger.info(
            "Starting indexing: %s",
            pdf_path.name,
        )

        file_hash = self.loader.file_hash(pdf_path)

        # Skip already indexed documents
        if self.metadata.document_exists(file_hash):

            logger.info(
                "Document already indexed: %s",
                pdf_path.name,
            )

            return False

        document = self.loader.open(pdf_path)

        try:

            # Register document
            self.metadata.add_document(
                file_hash=file_hash,
                file_name=pdf_path.name,
                total_pages=document.page_count,
            )

            total_chunks = 0

            # Process page batches
            for pages in self.loader.stream_page_batches(document):

                try:

                    batch_chunks = self.process_batch(
                        pages=pages,
                        file_name=pdf_path.name,
                        file_hash=file_hash,
                    )

                    if batch_chunks:
                        total_chunks += batch_chunks

                except Exception:

                    logger.exception(
                        "Failed processing batch in %s",
                        pdf_path.name,
                    )

                    raise

            # Persist FAISS index
            self.checkpoint()

            logger.info(
                "Successfully indexed %s | Pages=%d | Chunks=%d",
                pdf_path.name,
                document.page_count,
                total_chunks,
            )

            return True

        except Exception:

            logger.exception(
                "Failed indexing document: %s",
                pdf_path.name,
            )

            # Optional (recommended):
            # If MetadataRepository supports rollback,
            # uncomment the following:
            #
            # self.metadata.delete_document(file_hash)

            raise

        finally:

            self.loader.close(document)
    # ------------------------------------------------------
    # Folder
    # ------------------------------------------------------

    def index_folder(

        self,

        folder: Path,

    ):

        for pdf in folder.glob(
            "*.pdf"
        ):

            self.index_pdf(
                pdf
            )

        # ------------------------------------------------------
    # Status
    # ------------------------------------------------------

    def statistics(self):

        return {

            "documents":

                self.metadata.total_documents(),

            "chunks":

                self.metadata.total_chunks(),

            "vectors":

                self.faiss.index.ntotal,
        }

        # ------------------------------------------------------
    # Process Batch
    # ------------------------------------------------------

    def process_batch(
        self,
        pages,
        file_name: str,
        file_hash: str,
    ):

        chunks = []

        texts = []

        for page_number, page in pages:

            page_metadata = self.loader.page_metadata(
                page_number=page_number,
                page=page,
                file_hash=file_hash,
                file_name=file_name,
            )

            page_type = self.loader.classify_page(
                page
            )

            native_text = self.loader.extract_native_text(
                page
            )

            text = self.ocr.process_page(
                page=page,
                page_type=page_type,
                native_text=native_text,
                file_hash=file_hash,
                page_number=page_number,
            )

            for chunk in self.chunker.chunk_page(
                page_text=text,
                page_metadata=page_metadata,
            ):

                chunks.append(chunk)

                texts.append(chunk.text)

        if not chunks:
            return

        embeddings = self.embedding.embed_batch(
            texts
        )

        self.store_chunks(
            chunks,
            embeddings,
        )

        # ------------------------------------------------------
    # Store Chunks
    # ------------------------------------------------------

    def store_chunks(
        self,
        chunks,
        embeddings: np.ndarray,
    ):

        vector_ids = self.metadata.reserve_vector_ids(
            len(chunks)
        )

        self.faiss.add_vectors(
            embeddings=embeddings,
            vector_ids=vector_ids,
        )

        for vector_id, chunk in zip(
            vector_ids,
            chunks,
        ):

            self.metadata.add_chunk(

                vector_id=vector_id,

                chunk_id=chunk.chunk_id,

                file_hash=chunk.file_hash,

                page_number=chunk.page_number,

                chunk_index=chunk.chunk_index,

                text=chunk.text,

                metadata=chunk.metadata,
            )

        # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    def checkpoint(self):

        self.faiss.save()