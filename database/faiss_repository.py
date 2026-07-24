"""
Enterprise FAISS Repository

Features
--------
✓ IVF-PQ
✓ GPU support
✓ Persistent storage
✓ Incremental indexing
✓ Explicit vector IDs
✓ Thread-safe
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import (
    DEVICE,
    EMBEDDING_MODEL,
    FAISS_ADD_BATCH_SIZE,
    FAISS_INDEX_FILE,
    FAISS_M,
    FAISS_METADATA_FILE,
    FAISS_MIN_TRAINING_VECTORS,
    FAISS_NBITS,
    FAISS_NLIST,
    FAISS_NPROBE,
)

from core.constants import EMBEDDING_DIMENSIONS
from core.logger import get_logger
from database.metadata_repository import MetadataRepository

logger = get_logger(__name__)


class FaissRepository:
    """
    Enterprise IVF-PQ Repository
    """

    def __init__(
        self,
        metadata_repository: MetadataRepository,
    ):

        self.metadata_repository = metadata_repository

        self.dimension = EMBEDDING_DIMENSIONS[
            EMBEDDING_MODEL
        ]

        self.lock = threading.RLock()

        self.cpu_index = None
        self.gpu_index = None
        self.index = None

        self.next_vector_id = 0

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):

        if Path(FAISS_INDEX_FILE).exists():

            logger.info("Loading existing FAISS index")

            self._load_index()

        else:

            logger.info("Creating new IVF-PQ index")

            self._create_index()

        self.index.nprobe = FAISS_NPROBE

    # ------------------------------------------------------------------
    # Create IVF-PQ
    # ------------------------------------------------------------------

    def _create_index(self):

        quantizer = faiss.IndexFlatIP(
            self.dimension
        )

        ivfpq = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            FAISS_NLIST,
            FAISS_M,
            FAISS_NBITS,
        )

        id_map = faiss.IndexIDMap2(ivfpq)

        self.cpu_index = id_map

        if DEVICE == "cuda":

            self._cpu_to_gpu()

        else:

            self.index = self.cpu_index

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

    def _cpu_to_gpu(self):

        resources = faiss.StandardGpuResources()

        self.gpu_resources = resources

        self.gpu_index = faiss.index_cpu_to_gpu(
            resources,
            0,
            self.cpu_index,
        )

        self.index = self.gpu_index

        logger.info("FAISS running on GPU")

    def _gpu_to_cpu(self):

        if self.gpu_index is None:
            return

        self.cpu_index = faiss.index_gpu_to_cpu(
            self.gpu_index
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_index(self):

        self.cpu_index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )

        if DEVICE == "cuda":

            self._cpu_to_gpu()

        else:

            self.index = self.cpu_index

        self.next_vector_id = self.index.ntotal

        logger.info(
            "Loaded %d vectors",
            self.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:

        return self.index.is_trained

    def train(
        self,
        vectors: np.ndarray,
    ):

        if self.is_trained:

            logger.info(
                "IVF-PQ already trained"
            )
            return

        if len(vectors) < FAISS_MIN_TRAINING_VECTORS:

            raise ValueError(
                f"Need at least "
                f"{FAISS_MIN_TRAINING_VECTORS} "
                f"training vectors."
            )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        logger.info(
            "Training IVF-PQ on %d vectors",
            len(vectors),
        )

        with self.lock:

            self.index.train(vectors)

        logger.info("Training completed")

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    @property
    def total_vectors(self):

        return self.index.ntotal

    def __len__(self):

        return self.index.ntotal

    def __repr__(self):

        return (
            "FaissRepository("
            f"vectors={self.index.ntotal}, "
            f"trained={self.index.is_trained})"
        )

        # ------------------------------------------------------------------
    # Vector ID
    # ------------------------------------------------------------------

    def _generate_vector_ids(
        self,
        count: int,
    ) -> np.ndarray:
        """
        Generate sequential FAISS vector IDs.
        """

        ids = np.arange(
            self.next_vector_id,
            self.next_vector_id + count,
            dtype=np.int64,
        )

        self.next_vector_id += count

        return ids

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def _normalize(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        faiss.normalize_L2(vectors)

        return vectors

    # ------------------------------------------------------------------
    # Add vectors
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: list[dict],
    ) -> np.ndarray:
        """
        Add vectors into FAISS.

        Returns
        -------
        ndarray[int64]

        Assigned vector IDs.
        """

        if not self.index.is_trained:

            raise RuntimeError(
                "IVF-PQ index has not been trained."
            )

        vectors = self._normalize(vectors)

        ids = self._generate_vector_ids(
            len(vectors)
        )

        with self.lock:

            self.index.add_with_ids(
                vectors,
                ids,
            )

        logger.info(
            "Added %d vectors",
            len(ids),
        )

        self._save_metadata(
            ids,
            metadata,
        )

        return ids


        # ------------------------------------------------------------------
    # Batch insertion
    # ------------------------------------------------------------------

    def add_vectors_batch(
        self,
        vectors: np.ndarray,
        metadata: list[dict],
    ) -> list[int]:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        all_ids = []

        total = len(vectors)

        start = 0

        while start < total:

            end = min(
                start + FAISS_ADD_BATCH_SIZE,
                total,
            )

            batch_vectors = vectors[
                start:end
            ]

            batch_metadata = metadata[
                start:end
            ]

            ids = self.add_vectors(
                batch_vectors,
                batch_metadata,
            )

            all_ids.extend(
                ids.tolist()
            )

            logger.info(
                "Inserted %d/%d vectors",
                end,
                total,
            )

            start = end

        return all_ids


        # ------------------------------------------------------------------
    # SQLite synchronization
    # ------------------------------------------------------------------

    def _save_metadata(
        self,
        ids: np.ndarray,
        metadata: list[dict],
    ):

        for vector_id, meta in zip(
            ids,
            metadata,
        ):

            self.metadata_repository.add_chunk(

                vector_id=int(vector_id),

                file_hash=meta["file_hash"],

                page=meta["page"],

                chunk_index=meta["chunk_index"],

                text=meta["text"],

                image_paths=meta.get(
                    "image_paths",
                    "",
                ),

                metadata=str(meta),
            )

        # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self):

        return {

            "trained": self.index.is_trained,

            "vectors": self.index.ntotal,

            "dimension": self.dimension,

            "nlist": FAISS_NLIST,

            "m": FAISS_M,

            "nbits": FAISS_NBITS,

            "nprobe": self.index.nprobe,
        }

        # ------------------------------------------------------------------
    # nprobe
    # ------------------------------------------------------------------

    def set_nprobe(
        self,
        nprobe: int,
    ):
        """
        Update IVF search probes.
        """

        if nprobe <= 0:
            raise ValueError("nprobe must be positive.")

        self.index.nprobe = nprobe

        logger.info(
            "FAISS nprobe set to %d",
            nprobe,
        )

        # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search nearest vectors.

        Returns metadata enriched results.
        """

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = self._normalize(query_vector)

        distances, ids = self.index.search(
            query_vector,
            top_k,
        )

        results = []

        for score, vector_id in zip(
            distances[0],
            ids[0],
        ):

            if vector_id == -1:
                continue

            if (
                score_threshold is not None
                and score < score_threshold
            ):
                continue

            row = self.metadata_repository.get_chunk(
                int(vector_id)
            )

            if row is None:
                continue

            results.append(
                {
                    "vector_id": int(vector_id),
                    "score": float(score),
                    "text": row["text"],
                    "page": row["page"],
                    "file_hash": row["file_hash"],
                    "chunk_index": row["chunk_index"],
                    "image_paths": row["image_paths"],
                    "metadata": row["metadata"],
                }
            )

        return results

        # ------------------------------------------------------------------
    # Search IDs
    # ------------------------------------------------------------------

    def search_ids(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ):

        query_vector = self._normalize(
            query_vector
        )

        scores, ids = self.index.search(
            query_vector,
            top_k,
        )

        return scores[0], ids[0]

        # ------------------------------------------------------------------
    # Batch Search
    # ------------------------------------------------------------------

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ):

        query_vectors = np.asarray(
            query_vectors,
            dtype=np.float32,
        )

        query_vectors = self._normalize(
            query_vectors
        )

        scores, ids = self.index.search(
            query_vectors,
            top_k,
        )

        return scores, ids


        # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self):

        return {

            "trained": self.index.is_trained,

            "vectors": self.index.ntotal,

            "gpu": self.gpu_index is not None,

            "dimension": self.dimension,

            "nprobe": self.index.nprobe,

            "type": type(self.index).__name__,
        }

        # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):

        logger.warning(
            "Resetting FAISS index."
        )

        self._create_index()

        self.next_vector_id = 0


        # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self):
        """
        Persist FAISS index to disk.
        """

        with self.lock:

            if self.gpu_index is not None:

                logger.info(
                    "Copying GPU index to CPU..."
                )

                self._gpu_to_cpu()

            logger.info(
                "Saving FAISS index..."
            )

            faiss.write_index(
                self.cpu_index,
                str(FAISS_INDEX_FILE),
            )

            logger.info(
                "Index saved successfully."
            )

            if DEVICE == "cuda":

                self._cpu_to_gpu()


        # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def checkpoint(self):

        logger.info(
            "Checkpointing FAISS..."
        )

        self.save()


        # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def reload(self):

        logger.info(
            "Reloading FAISS..."
        )

        self._load_index()

        # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self):

        self.save()

        # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):

        logger.info(
            "Closing FAISS repository."
        )

        self.save()

        self.index = None

        self.cpu_index = None

        self.gpu_index = None


        # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self):

        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        self.close()

        # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __str__(self):

        return (
            f"IVFPQ("
            f"vectors={self.index.ntotal}, "
            f"trained={self.index.is_trained}, "
            f"dimension={self.dimension})"
        )