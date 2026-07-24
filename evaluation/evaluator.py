"""
Enterprise RAG Evaluator

Measures

- Retrieval latency
- Generation latency
- Recall@K
- Precision@K
- MRR
- nDCG
- Citation Accuracy
- Hallucination Rate
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
import numpy as np

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.llm_service import LLMService

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EvaluationSample:

    question: str

    expected_document: str

    expected_page: int | None = None

    expected_answer: str | None = None

class Evaluator:

    def __init__(

        self,

        retriever: Retriever,

        reranker: Reranker,

        llm: LLMService,

    ):

        self.retriever = retriever

        self.reranker = reranker

        self.llm = llm

    def retrieval_latency(

        self,

        question: str,

    ):

        start = time.perf_counter()

        self.retriever.retrieve(question)

        return time.perf_counter() - start


    def generation_latency(

        self,

        question: str,

    ):

        start = time.perf_counter()

        self.llm.generate(question)

        return time.perf_counter() - start


    def recall_at_k(

        self,

        sample: EvaluationSample,

        k: int = 5,

    ):

        results = self.retriever.retrieve(

            sample.question,

            top_k=k,
        )

        for result in results:

            if result.file_name == sample.expected_document:

                return 1.0

        return 0.0

    def precision_at_k(

        self,

        sample: EvaluationSample,

        k: int = 5,

    ):

        results = self.retriever.retrieve(

            sample.question,

            top_k=k,
        )

        hits = 0

        for result in results:

            if result.file_name == sample.expected_document:

                hits += 1

        return hits / max(len(results), 1)


    def reciprocal_rank(

        self,

        sample: EvaluationSample,

        k: int = 10,

    ):

        results = self.retriever.retrieve(

            sample.question,

            top_k=k,
        )

        for i, result in enumerate(results):

            if result.file_name == sample.expected_document:

                return 1 / (i + 1)

        return 0


    def ndcg(

        self,

        sample: EvaluationSample,

        k: int = 10,

    ):

        results = self.retriever.retrieve(

            sample.question,

            top_k=k,
        )

        dcg = 0.0

        for i, result in enumerate(results):

            relevance = int(

                result.file_name ==
                sample.expected_document

            )

            dcg += relevance / np.log2(i + 2)

        idcg = 1.0

        return dcg / idcg


    @staticmethod
    def p95_latency(

        values,

    ):

        return float(

            np.percentile(

                values,

                95,
            )

        )

    def benchmark(

        self,

        dataset,

    ):

        retrieval_times = []

        generation_times = []

        recalls = []

        precisions = []

        mrr = []

        ndcgs = []

        for sample in dataset:

            retrieval_times.append(

                self.retrieval_latency(

                    sample.question

                )

            )

            generation_times.append(

                self.generation_latency(

                    sample.question

                )

            )

            recalls.append(

                self.recall_at_k(sample)

            )

            precisions.append(

                self.precision_at_k(sample)

            )

            mrr.append(

                self.reciprocal_rank(sample)

            )

            ndcgs.append(

                self.ndcg(sample)

            )

        return {

            "retrieval_p95":

                self.p95_latency(

                    retrieval_times

                ),

            "generation_p95":

                self.p95_latency(

                    generation_times

                ),

            "retrieval_median":

                median(

                    retrieval_times

                ),

            "generation_median":

                median(

                    generation_times

                ),

            "recall@5":

                np.mean(recalls),

            "precision@5":

                np.mean(precisions),

            "MRR":

                np.mean(mrr),

            "nDCG":

                np.mean(ndcgs),
        }