"""
Enterprise RAG Chatbot

Pipeline

Upload PDFs
      ↓
Incremental Indexing
      ↓
FAISS + SQLite
      ↓
Retriever
      ↓
Reranker
      ↓
LLM
      ↓
Answer + Citations
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import (
    DOCUMENT_DIRECTORY,
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
)

from core.logger import get_logger

from document_processing.pdf_loader import PDFLoader
from document_processing.ocr_engine import OCREngine
from document_processing.chunker import DocumentChunker

from embedding.embedding_model import EmbeddingModel

from indexing.incremental_indexer import IncrementalIndexer

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

from llm.llm_service import LLMService

from database.faiss_repository import FaissRepository
from database.metadata_repository import MetadataRepository

logger = get_logger(__name__)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)

st.title("📚 Enterprise RAG Chatbot")


if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource(show_spinner=False)
def build_pipeline():

    metadata = MetadataRepository()

    faiss = FaissRepository()

    loader = PDFLoader()

    ocr = OCREngine()

    chunker = DocumentChunker()

    embedding = EmbeddingModel()

    indexer = IncrementalIndexer(

        pdf_loader=loader,

        ocr_engine=ocr,

        chunker=chunker,

        embedding_model=embedding,

        faiss_repository=faiss,

        metadata_repository=metadata,
    )

    retriever = Retriever(

        embedding,

        faiss,

        metadata,
    )

    reranker = Reranker()

    llm = LLMService(

        retriever,

        reranker,
    )

    return {

        "metadata": metadata,

        "faiss": faiss,

        "loader": loader,

        "ocr": ocr,

        "chunker": chunker,

        "embedding": embedding,

        "indexer": indexer,

        "retriever": retriever,

        "reranker": reranker,

        "llm": llm,
    }

pipeline = build_pipeline()

with st.sidebar:

    st.header("Documents")

    stats = pipeline["indexer"].statistics()

    st.metric(
        "Documents",
        stats["documents"],
    )

    st.metric(
        "Chunks",
        stats["chunks"],
    )

    st.metric(
        "Vectors",
        stats["vectors"],
    )


uploaded = st.file_uploader(

    "Upload PDFs",

    type=["pdf"],

    accept_multiple_files=True,
)

if uploaded:

    document_directory = Path(
        DOCUMENT_DIRECTORY
    )

    document_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in uploaded:

        destination = (
            document_directory / file.name
        )

        with open(destination, "wb") as f:

            f.write(
                file.read()
            )

    st.success(
        f"{len(uploaded)} PDF(s) uploaded."
    )


if st.button(
    "Build / Update Index",
    type="primary",
):

    with st.spinner(
        "Indexing documents..."
    ):

        pipeline["indexer"].index_folder(
            Path(DOCUMENT_DIRECTORY)
        )

    st.success(
        "Index updated successfully."
    )

    st.rerun()


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    st.session_state.chat_history.append(

        {
            "role": "user",
            "content": question,
        }

    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = pipeline["llm"].generate(
                question
            )

        st.markdown(
            response["answer"]
        )

        with st.expander(
            "Sources"
        ):

            for source in response["sources"]:

                st.markdown(

                    f"""
**{source.file_name}**

Page: {source.page_number}

Score: {source.rerank_score:.3f}
"""
                )

        st.caption(
            f"Latency: {response['latency']:.2f} sec"
        )

    st.session_state.chat_history.append(

        {
            "role": "assistant",
            "content": response["answer"],
        }

    )

st.divider()

st.caption(
    "Enterprise RAG • Hybrid Retrieval • Cross-Encoder Reranking • Grounded Answers"
)