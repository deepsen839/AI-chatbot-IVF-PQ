"""
Enterprise Retriever

Pipeline

Query
  ↓
Embedding
  ↓
FAISS Search
  ↓
Metadata Lookup
  ↓
Results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from embedding.embedding_model import EmbeddingModel
from database.faiss_repository import FaissRepository
from database.metadata_repository import MetadataRepository

from config import TOP_K

from core.logger import get_logger

logger = get_logger(__name__)
import time
from collections import defaultdict

@dataclass(slots=True)
class SearchResult:

    vector_id: int

    score: float

    text: str

    file_name: str

    file_hash: str

    page_number: int

    chunk_index: int

    metadata: dict


class Retriever:

    def __init__(

        self,

        embedding_model: EmbeddingModel,

        faiss_repository: FaissRepository,

        metadata_repository: MetadataRepository,

    ):

        self.embedding = embedding_model

        self.faiss = faiss_repository

        self.metadata = metadata_repository


        # -----------------------------------------------------
    # Query Embedding
    # -----------------------------------------------------

    def embed_query(
        self,
        query: str,
    ):

        return self.embedding.embed(query)


        # -----------------------------------------------------
    # Vector Search
    # -----------------------------------------------------

    def vector_search(

        self,

        query: str,

        top_k: int = TOP_K,

    ):

        embedding = self.embed_query(
            query
        )

        return self.faiss.search(

            embedding,

            top_k=top_k,
        )

        # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    def metadata_lookup(
        self,
        vector_id: int,
    ):

        return self.metadata.get_chunk_by_vector_id(
            vector_id
        )


        # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[SearchResult]:

        start = time.perf_counter()

        scores, ids = self.vector_search(
            query,
            top_k,
        )

        normalized = self.normalize_scores(scores)

        results = []

        for score, vector_id in zip(
            normalized,
            ids,
        ):

            if vector_id == -1:
                continue

            chunk = self.metadata_lookup(
                vector_id
            )

            if chunk is None:
                continue

            results.append(

                SearchResult(

                    vector_id=vector_id,

                    score=float(score),

                    text=chunk["text"],

                    file_name=chunk["file_name"],

                    file_hash=chunk["file_hash"],

                    page_number=chunk["page_number"],

                    chunk_index=chunk["chunk_index"],

                    metadata=chunk["metadata"],
                )

            )

        results = self.deduplicate(
            results
        )

        results = self.diversify(
            results
        )

        latency = (
            time.perf_counter()
            - start
        )

        logger.info(

            "Retrieved %d chunks in %.3f sec",

            len(results),

            latency,

        )

        return results

        # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    def statistics(self):

        return {

            "documents":
                self.metadata.total_documents(),

            "chunks":
                self.metadata.total_chunks(),

            "vectors":
                self.faiss.index.ntotal,
        }

        # -----------------------------------------------------
    # Normalize Scores
    # -----------------------------------------------------

    @staticmethod
    def normalize_scores(
        scores,
    ):

        if len(scores) == 0:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:

            return [1.0] * len(scores)

        return [

            (score - minimum)
            / (maximum - minimum)

            for score in scores

        ]

        # -----------------------------------------------------
    # Remove Duplicates
    # -----------------------------------------------------

    def deduplicate(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        unique = {}

        for result in results:

            key = (

                result.file_hash,

                result.page_number,

                result.chunk_index,

            )

            if key not in unique:

                unique[key] = result

        return list(unique.values())


        # -----------------------------------------------------
    # Diversify
    # -----------------------------------------------------

    def diversify(
        self,
        results: list[SearchResult],
        max_per_document: int = 3,
    ) -> list[SearchResult]:

        grouped = defaultdict(int)

        diversified = []

        for result in sorted(
            results,
            key=lambda r: r.score,
            reverse=True,
        ):

            if grouped[result.file_hash] >= max_per_document:
                continue

            diversified.append(result)

            grouped[result.file_hash] += 1

        return diversified


        # -----------------------------------------------------
    # File Distribution
    # -----------------------------------------------------

    def top_files(
        self,
        results: list[SearchResult],
    ) -> dict:

        distribution = defaultdict(int)

        for result in results:

            distribution[result.file_name] += 1

        return dict(distribution)


        # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    def health(self):

        return {

            "documents":

                self.metadata.total_documents(),

            "chunks":

                self.metadata.total_chunks(),

            "vectors":

                self.faiss.index.ntotal,

            "embedding_dimension":

                self.embedding.dimension,
        }