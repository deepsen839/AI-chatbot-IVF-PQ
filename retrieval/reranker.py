"""
Enterprise Cross Encoder Reranker

Pipeline

Query
    ↓
Retrieved Chunks
    ↓
Cross Encoder
    ↓
Sorted Results
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from config import (
    DEVICE,
    RERANKER_MODEL,
)

from core.logger import get_logger

from retrieval.retriever import SearchResult

logger = get_logger(__name__)

@dataclass(slots=True)
class RerankedResult:

    vector_id: int

    rerank_score: float

    retrieval_score: float

    text: str

    file_name: str

    file_hash: str

    page_number: int

    chunk_index: int

    metadata: dict


class Reranker:

    def __init__(self):

        logger.info(
            "Loading reranker: %s",
            RERANKER_MODEL,
        )

        self.model = CrossEncoder(

            RERANKER_MODEL,

            device=DEVICE,
        )

        logger.info(
            "Reranker ready."
        )

        # --------------------------------------------------------
    # Pair Construction
    # --------------------------------------------------------

    def build_pairs(

        self,

        query: str,

        results: list[SearchResult],

    ):

        return [

            (

                query,

                result.text,

            )

            for result in results

        ]


        # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    def score(

        self,

        query: str,

        results: list[SearchResult],

    ):

        if not results:

            return []

        pairs = self.build_pairs(

            query,

            results,
        )

        scores = self.model.predict(

            pairs,

            show_progress_bar=False,
        )

        return scores.tolist()


        # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    def rerank(

        self,

        query: str,

        results: list[SearchResult],

        top_k: int,

    ) -> list[RerankedResult]:

        if not results:

            return []

        start = time.perf_counter()

        scores = self.score(

            query,

            results,
        )

        ranked = []

        for result, score in zip(

            results,

            scores,
        ):

            ranked.append(

                RerankedResult(

                    vector_id=result.vector_id,

                    rerank_score=float(score),

                    retrieval_score=result.score,

                    text=result.text,

                    file_name=result.file_name,

                    file_hash=result.file_hash,

                    page_number=result.page_number,

                    chunk_index=result.chunk_index,

                    metadata=result.metadata,
                )

            )

        ranked.sort(

            key=lambda x: x.rerank_score,

            reverse=True,
        )

        latency = (
            time.perf_counter()
            - start
        )

        logger.info(

            "Reranked %d chunks in %.3f sec",

            len(ranked),

            latency,
        )

        return ranked[:top_k]


        # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def statistics(self):

        return {

            "model":

                RERANKER_MODEL,

            "device":

                DEVICE,

            "cuda":

                torch.cuda.is_available(),
        }

        # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def close(self):

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        logger.info(
            "Reranker released."
        )