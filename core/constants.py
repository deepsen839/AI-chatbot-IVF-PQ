"""
Enterprise RAG Constants

Shared constants used across the entire project.
"""

from enum import Enum


# =============================================================================
# Supported Files
# =============================================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
}


# =============================================================================
# PDF Page Types
# =============================================================================

class PageType(str, Enum):
    TEXT = "text"
    SCANNED = "scanned"
    MIXED = "mixed"
    EMPTY = "empty"


# =============================================================================
# OCR Status
# =============================================================================

class OCRStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Processing Status
# =============================================================================

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Metadata Keys
# =============================================================================

FILE_NAME = "file_name"

FILE_HASH = "file_hash"

FILE_SIZE = "file_size"

PAGE_NUMBER = "page"

PAGE_TYPE = "page_type"

CHUNK_ID = "chunk_id"

CHUNK_INDEX = "chunk_index"

TEXT = "text"

SOURCE = "source"

VECTOR_ID = "vector_id"

IMAGE_PATHS = "image_paths"

OCR_STATUS = "ocr_status"

CREATED_AT = "created_at"

UPDATED_AT = "updated_at"


# =============================================================================
# Image Formats
# =============================================================================

SUPPORTED_IMAGE_FORMATS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tiff",
    "webp",
}


# =============================================================================
# Retrieval Defaults
# =============================================================================

DEFAULT_TOP_K = 50

DEFAULT_FINAL_TOP_K = 5


# =============================================================================
# Similarity
# =============================================================================

MIN_SIMILARITY_SCORE = 0.30


# =============================================================================
# Chunk Metadata
# =============================================================================

CHUNK_METADATA_FIELDS = [
    FILE_NAME,
    PAGE_NUMBER,
    CHUNK_ID,
    CHUNK_INDEX,
    SOURCE,
    VECTOR_ID,
]


# =============================================================================
# Metadata Table Names
# =============================================================================

DOCUMENT_TABLE = "documents"

CHUNK_TABLE = "chunks"

VECTOR_TABLE = "vectors"


# =============================================================================
# OCR Languages
# =============================================================================

SUPPORTED_OCR_LANGUAGES = {
    "en",
}


# =============================================================================
# Embedding
# =============================================================================

EMBEDDING_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


# =============================================================================
# Search Types
# =============================================================================

class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"


# =============================================================================
# LLM Providers
# =============================================================================

class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"


# =============================================================================
# Image Extraction
# =============================================================================

MIN_IMAGE_AREA = 100 * 100


# =============================================================================
# Logging Messages
# =============================================================================

LOG_PDF_LOADING = "Loading PDF"

LOG_PDF_COMPLETED = "PDF Loaded Successfully"

LOG_OCR_STARTED = "OCR Started"

LOG_OCR_COMPLETED = "OCR Completed"

LOG_EMBEDDING_STARTED = "Generating Embeddings"

LOG_EMBEDDING_COMPLETED = "Embedding Generation Finished"

LOG_FAISS_SAVE = "Saving FAISS Index"

LOG_FAISS_LOAD = "Loading FAISS Index"

LOG_QUERY = "User Query"

LOG_RETRIEVAL = "Retrieving Documents"

LOG_RESPONSE = "Generating LLM Response"


# =============================================================================
# Default Messages
# =============================================================================

NO_DOCUMENT_MESSAGE = (
    "Please upload one or more PDF documents."
)

NO_RESULTS_MESSAGE = (
    "No relevant information was found."
)

INDEX_EMPTY_MESSAGE = (
    "Vector database is empty."
)

UPLOAD_SUCCESS = (
    "Documents uploaded successfully."
)

UPLOAD_FAILED = (
    "Failed to upload document."
)