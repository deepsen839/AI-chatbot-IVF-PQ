"""
Enterprise RAG Utility Functions
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Directory
# =============================================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Create directory if it doesn't exist.
    """
    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


# =============================================================================
# Timestamp
# =============================================================================

def current_timestamp() -> str:
    """
    Current UTC timestamp.
    """
    return datetime.utcnow().isoformat()


# =============================================================================
# UUID
# =============================================================================

def generate_uuid() -> str:
    """
    Generate UUID4 string.
    """
    return str(uuid.uuid4())


# =============================================================================
# SHA256
# =============================================================================

def calculate_file_hash(file_path: str | Path) -> str:
    """
    SHA256 hash of a file.
    """
    sha = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha.update(data)

    return sha.hexdigest()


# =============================================================================
# File Size
# =============================================================================

def get_file_size(file_path: str | Path) -> int:
    """
    File size in bytes.
    """
    return Path(file_path).stat().st_size


def human_readable_size(size: int) -> str:
    """
    Convert bytes to KB/MB/GB.
    """

    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:

        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


# =============================================================================
# Filename
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Safe filename.
    """

    keep = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    return "".join(c if c in keep else "_" for c in filename)


# =============================================================================
# JSON
# =============================================================================

def save_json(data: Any, file_path: str | Path):
    """
    Save JSON using orjson.
    """

    file_path = Path(file_path)

    with open(file_path, "wb") as file:
        file.write(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2,
            )
        )


def load_json(file_path: str | Path, default=None):
    """
    Load JSON.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return default

    with open(file_path, "rb") as file:
        return orjson.loads(file.read())


# =============================================================================
# Standard JSON
# =============================================================================

def save_json_std(data: Any, file_path: str |Path):

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_json_std(file_path: str |Path, default=None):

    if not Path(file_path).exists():
        return default

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# =============================================================================
# File Validation
# =============================================================================

def is_pdf(file_path: str | Path) -> bool:
    """
    Check if file is PDF.
    """

    return Path(file_path).suffix.lower() == ".pdf"


def file_exists(file_path: str | Path) -> bool:
    """
    Check file exists.
    """

    return Path(file_path).exists()


# =============================================================================
# Delete
# =============================================================================

def delete_file(file_path: str | Path):

    file_path = Path(file_path)

    if file_path.exists():

        file_path.unlink()


# =============================================================================
# Directory Size
# =============================================================================

def directory_size(directory: str | Path) -> int:

    total = 0

    directory = Path(directory)

    for file in directory.rglob("*"):

        if file.is_file():
            total += file.stat().st_size

    return total


# =============================================================================
# Count Files
# =============================================================================

def count_files(directory: str | Path) -> int:

    directory = Path(directory)

    return sum(
        1
        for item in directory.rglob("*")
        if item.is_file()
    )


# =============================================================================
# Flatten
# =============================================================================

def flatten(list_of_lists):

    return [
        item
        for sublist in list_of_lists
        for item in sublist
    ]


# =============================================================================
# Batch Generator
# =============================================================================

def batch_iterator(data, batch_size):

    for i in range(0, len(data), batch_size):

        yield data[i:i + batch_size]


# =============================================================================
# Timer Decorator
# =============================================================================

def timer(func):

    from time import perf_counter

    def wrapper(*args, **kwargs):

        start = perf_counter()

        result = func(*args, **kwargs)

        elapsed = perf_counter() - start

        logger.info(
            "%s completed in %.2f sec",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper