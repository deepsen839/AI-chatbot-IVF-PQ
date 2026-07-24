"""
Enterprise LLM Service

Pipeline

Question
    ↓
Retriever
    ↓
Reranker
    ↓
Context
    ↓
LLM
    ↓
Grounded Answer
"""

from __future__ import annotations

import time
from typing import List

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_CONTEXT_CHUNKS,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from retrieval.retriever import Retriever
from retrieval.reranker import (
    Reranker,
    RerankedResult,
)

from core.logger import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(

        self,

        retriever: Retriever,

        reranker: Reranker,

    ):

        self.retriever = retriever

        self.reranker = reranker

        self.client = Groq(
                api_key=GROQ_API_KEY,
        )


        # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    def retrieve_context(

        self,

        question: str,

    ) -> list[RerankedResult]:

        retrieved = self.retriever.retrieve(
            question
        )

        reranked = self.reranker.rerank(

            query=question,

            results=retrieved,

            top_k=MAX_CONTEXT_CHUNKS,
        )

        return reranked

        # --------------------------------------------------------
    # System Prompt
    # --------------------------------------------------------

    @staticmethod
    def system_prompt():

        return """
You are an enterprise knowledge assistant.

Answer ONLY using the supplied context.

If the answer cannot be found in the context, say:

"I couldn't find that information in the indexed documents."

Never invent facts.

Always cite the document name and page number.
""".strip()


        # --------------------------------------------------------
    # Context Builder
    # --------------------------------------------------------

    def build_context(

        self,

        results: List[RerankedResult],

    ) -> str:

        context = []

        for chunk in results:

            context.append(

                f"""
Document : {chunk.file_name}
Page     : {chunk.page_number}

{chunk.text}
""".strip()

            )

        return "\n\n----------------------\n\n".join(
            context
        )

        # --------------------------------------------------------
    # User Prompt
    # --------------------------------------------------------

    def build_prompt(

        self,

        question: str,

        context: str,

    ):

        return f"""
                Context

                {context}

                ------------------------------------

                Question

                {question}

                Answer only from the supplied context.
                """.strip()


        # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    def generate(

        self,

        question: str,

    ):

        start = time.perf_counter()

        context_chunks = self.retrieve_context(
            question
        )

        context = self.build_context(
            context_chunks
        )

        prompt = self.build_prompt(

            question,

            context,
        )

        response = self.client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": self.system_prompt(),
                },

                {
                    "role": "user",
                    "content": prompt,
                },

            ],

            temperature=TEMPERATURE,

            max_tokens=MAX_OUTPUT_TOKENS,
        )

        answer = response.choices[0].message.content

        latency = (
            time.perf_counter()
            - start
        )

        logger.info(

            "LLM response generated in %.2f sec",

            latency,
        )

        return {

            "answer": answer,

            "sources": context_chunks,

            "latency": latency,
        }


        # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    def health(self):

        return {

            "model": OPENAI_MODEL,

            "provider": OPENAI_BASE_URL,
        }