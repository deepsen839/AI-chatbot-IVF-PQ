"""
Enterprise Embedding Model

Features
--------
✓ BGE embeddings
✓ GPU/CPU support
✓ Batch encoding
✓ L2 normalization
✓ Streaming support
✓ Memory efficient
"""

from __future__ import annotations

from typing import Generator, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import (
    DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)
import hashlib
import time
from pathlib import Path

import orjson
from core.logger import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    """
    Enterprise embedding service.
    """

    def __init__(self):

        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL,
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
        )

        self.dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        logger.info(
            "Embedding dimension: %d",
            self.dimension,
        )
        self.cache_dir = Path("embedding_cache")
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.total_embeddings = 0
        self.total_time = 0.0




    # -------------------------------------------------------
    # Normalize
    # -------------------------------------------------------

    @staticmethod
    def normalize(
        embeddings: np.ndarray,
    ) -> np.ndarray:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1

        return embeddings / norms


        # -------------------------------------------------------
    # Single
    # -------------------------------------------------------
    # -------------------------------------------------------
    # Cached Embedding
    # -------------------------------------------------------

    def embed(
        self,
        text: str,
    ) -> np.ndarray:

        cached = self.load_cache(text)

        if cached is not None:

            return cached

        start = time.perf_counter()

        embedding = self.model.encode(

            text,

            convert_to_numpy=True,

            normalize_embeddings=False,

            show_progress_bar=False,
        )

        embedding = embedding.reshape(1, -1)

        embedding = self.normalize(
            embedding
        )[0]

        self.save_cache(
            text,
            embedding,
        )

        self.total_embeddings += 1

        self.total_time += (
            time.perf_counter() - start
        )

        return embedding

        # -------------------------------------------------------
    # Batch
    # -------------------------------------------------------

    def embed_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:

        embeddings = self.model.encode(

            texts,

            batch_size=EMBEDDING_BATCH_SIZE,

            convert_to_numpy=True,

            normalize_embeddings=False,

            show_progress_bar=False,
        )

        return self.normalize(
            embeddings
        )

        # -------------------------------------------------------
    # Streaming
    # -------------------------------------------------------

    def stream_embeddings(
        self,
        texts: Iterable[str],
    ) -> Generator[np.ndarray, None, None]:

        batch = []

        for text in texts:

            batch.append(text)

            if len(batch) >= EMBEDDING_BATCH_SIZE:

                yield self.embed_batch(
                    batch
                )

                batch.clear()

        if batch:

            yield self.embed_batch(
                batch
            )

        # -------------------------------------------------------
    # Device
    # -------------------------------------------------------

    @property
    def device(self):

        return str(
            next(
                self.model.parameters()
            ).device
        )

        # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    def info(
        self,
    ) -> dict:

        return {

            "model": EMBEDDING_MODEL,

            "dimension": self.dimension,

            "device": self.device,

            "cuda": torch.cuda.is_available(),
        }

        # -------------------------------------------------------
    # Close
    # -------------------------------------------------------

    def close(self):

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        logger.info(
            "Embedding model released."
        )

        # -------------------------------------------------------
    # Cache Key
    # -------------------------------------------------------

    def cache_key(
        self,
        text: str,
    ) -> str:

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------------
    # Cache File
    # -------------------------------------------------------

    def cache_file(
        self,
        key: str,
    ) -> Path:

        return self.cache_dir / f"{key}.json"


        # -------------------------------------------------------
    # Load Cache
    # -------------------------------------------------------

    def load_cache(
        self,
        text: str,
    ):

        key = self.cache_key(text)

        file = self.cache_file(key)

        if not file.exists():
            return None

        with open(file, "rb") as fp:

            data = orjson.loads(
                fp.read()
            )

        return np.asarray(
            data,
            dtype=np.float32,
        )

        # -------------------------------------------------------
    # Save Cache
    # -------------------------------------------------------

    def save_cache(
        self,
        text: str,
        embedding: np.ndarray,
    ):

        key = self.cache_key(text)

        file = self.cache_file(key)

        with open(file, "wb") as fp:

            fp.write(
                orjson.dumps(
                    embedding.tolist()
                )
            )


        # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    def validate(
        self,
        embedding: np.ndarray,
    ) -> bool:

        if embedding is None:
            return False

        if embedding.ndim != 1:
            return False

        if embedding.shape[0] != self.dimension:
            return False

        if np.isnan(embedding).any():
            return False

        if np.isinf(embedding).any():
            return False

        return True

        # -------------------------------------------------------
    # Adaptive Batch
    # -------------------------------------------------------

    def embed_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:

        batch_size = EMBEDDING_BATCH_SIZE

        while True:

            try:

                embeddings = self.model.encode(

                    texts,

                    batch_size=batch_size,

                    convert_to_numpy=True,

                    normalize_embeddings=False,

                    show_progress_bar=False,
                )

                return self.normalize(
                    embeddings
                )

            except RuntimeError as exc:

                if (
                    "out of memory"
                    not in str(exc).lower()
                    or batch_size == 1
                ):
                    raise

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_size //= 2

                logger.warning(
                    "Reducing embedding batch size to %d",
                    batch_size,
                )

        # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------

    def metrics(
        self,
    ) -> dict:

        average = 0.0

        if self.total_embeddings:

            average = (
                self.total_time
                / self.total_embeddings
            )

        return {

            "model": EMBEDDING_MODEL,

            "dimension": self.dimension,

            "embeddings_created":
                self.total_embeddings,

            "total_seconds":
                round(
                    self.total_time,
                    3,
                ),

            "average_seconds":
                round(
                    average,
                    5,
                ),
        }