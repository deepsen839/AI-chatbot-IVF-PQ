"""
Central configuration for Enterprise RAG
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# -----------------------------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# Project Directories
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
IMAGE_DIR = STORAGE_DIR / "images"
INDEX_DIR = STORAGE_DIR / "index"
CACHE_DIR = STORAGE_DIR / "cache"

LOG_DIR = BASE_DIR / "logs"

# Metadata Database
METADATA_DB = STORAGE_DIR / "metadata.db"

# Processed files registry
PROCESSED_FILES = STORAGE_DIR / "processed_files.json"

# -----------------------------------------------------------------------------
# Create Directories
# -----------------------------------------------------------------------------

DIRECTORIES = [
    STORAGE_DIR,
    DOCUMENTS_DIR,
    IMAGE_DIR,
    INDEX_DIR,
    CACHE_DIR,
    LOG_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# FAISS
# -----------------------------------------------------------------------------

FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"

# Stores metadata corresponding to vector ids
FAISS_METADATA_FILE = INDEX_DIR / "index_metadata.pkl"

# IVF-PQ Parameters
FAISS_NLIST = 4096
FAISS_M = 32
FAISS_NBITS = 8

# Search parameter
FAISS_NPROBE = 32

# Train IVF after these many vectors
FAISS_MIN_TRAINING_VECTORS = 160000

# Batch size while adding vectors
FAISS_ADD_BATCH_SIZE = 1024

# -----------------------------------------------------------------------------
# Embedding Model
# -----------------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EMBEDDING_DEVICE = "cuda"

EMBEDDING_BATCH_SIZE = 128

EMBEDDING_NORMALIZE = True

# -----------------------------------------------------------------------------
# Reranker
# -----------------------------------------------------------------------------

RERANKER_MODEL = "BAAI/bge-reranker-base"

RERANKER_DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

MIN_CHUNK_LENGTH = 100

# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

OCR_LANGUAGE = "en"

# Render PDF pages at this DPI before OCR
OCR_DPI = 300

OCR_USE_GPU = True

OCR_BATCH_SIZE = 8

OCR_MIN_TEXT_THRESHOLD = 40

# Cache rendered images and OCR text
OCR_CACHE_DIR = CACHE_DIR / "ocr"

OCR_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# If extracted text from a page is below this,
# OCR will be executed.

# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------

EXTRACT_IMAGES = True

MIN_IMAGE_WIDTH = 100

MIN_IMAGE_HEIGHT = 100

IMAGE_FORMAT = "png"

# -----------------------------------------------------------------------------
# PDF
# -----------------------------------------------------------------------------

MAX_UPLOAD_SIZE_MB = 500

SUPPORTED_FILE_TYPES = [".pdf"]

PDF_BATCH_PAGES = 10

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

TOP_K = 50

FINAL_TOP_K = 5

SIMILARITY_THRESHOLD = 0.30

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_PROVIDER = "groq"

GROQ_MODEL = "llama-3.3-70b-versatile"

OPENAI_MODEL = "gpt-4.1-mini"

TEMPERATURE = 0.0

MAX_TOKENS = 2048

# -----------------------------------------------------------------------------
# Parallel Processing
# -----------------------------------------------------------------------------

MAX_WORKERS = 8

ENABLE_MULTIPROCESSING = True

# -----------------------------------------------------------------------------
# Duplicate Detection
# -----------------------------------------------------------------------------

HASH_ALGORITHM = "sha256"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "enterprise_rag.log"

# -----------------------------------------------------------------------------
# Streamlit
# -----------------------------------------------------------------------------

PAGE_TITLE = "Enterprise RAG"

PAGE_ICON = "📚"

LAYOUT = "wide"

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

RANDOM_SEED = 42

DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Citation
# -----------------------------------------------------------------------------

ENABLE_CITATIONS = True

CITATION_TEMPLATE = (
    "{file} | Page {page} | Chunk {chunk}"
)
DOCUMENT_DIRECTORY=DOCUMENTS_DIR
MAX_CONTEXT_CHUNKS = FINAL_TOP_K
MAX_OUTPUT_TOKENS=MAX_TOKENS

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_DEVICE = DEVICE

RERANKER_DEVICE = DEVICE

OCR_USE_GPU = DEVICE == "cuda"


ENABLE_RERANKING = True

ENABLE_HYBRID_SEARCH = True

ENABLE_QUERY_EXPANSION = False

ENABLE_MULTI_QUERY = False

MAX_CONTEXT_CHARACTERS = 25000

LLM_MAX_RETRIES = 3

LLM_TIMEOUT = 60

ENABLE_RERANKING = True

ENABLE_HYBRID_SEARCH = True

ENABLE_QUERY_EXPANSION = False

ENABLE_MULTI_QUERY = False

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

MAX_OUTPUT_TOKENS = 2048