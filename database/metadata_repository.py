"""
SQLite Metadata Repository

Stores:
- Document metadata
- Chunk metadata
- Vector mapping
- Duplicate detection
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from config import METADATA_DB
from core.logger import get_logger

logger = get_logger(__name__)


class MetadataRepository:
    """
    SQLite metadata repository.
    """

    def __init__(self, db_path: Path = METADATA_DB):

        self.db_path = Path(db_path)

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                file_hash TEXT UNIQUE,

                file_name TEXT,

                file_path TEXT,

                file_size INTEGER,

                total_pages INTEGER,

                processed_at TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                vector_id INTEGER,

                file_hash TEXT,

                page INTEGER,

                chunk_index INTEGER,

                text TEXT,

                image_paths TEXT,

                metadata TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_vector
            ON chunks(vector_id)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_filehash
            ON chunks(file_hash)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_page
            ON chunks(page)
            """
        )

        self.connection.commit()

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    def document_exists(
        self,
        file_hash: str,
    ) -> bool:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT 1
            FROM documents
            WHERE file_hash=?
            LIMIT 1
            """,
            (file_hash,),
        )

        return cursor.fetchone() is not None

    def add_document(
        self,
        file_hash: str,
        file_name: str,
        file_path: str,
        file_size: int,
        total_pages: int,
        processed_at: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO documents(

                file_hash,

                file_name,

                file_path,

                file_size,

                total_pages,

                processed_at

            )
            VALUES(?,?,?,?,?,?)
            """,
            (
                file_hash,
                file_name,
                file_path,
                file_size,
                total_pages,
                processed_at,
            ),
        )

        self.connection.commit()

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        vector_id: int,
        file_hash: str,
        page: int,
        chunk_index: int,
        text: str,
        image_paths: str,
        metadata: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO chunks(

                vector_id,

                file_hash,

                page,

                chunk_index,

                text,

                image_paths,

                metadata

            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                vector_id,
                file_hash,
                page,
                chunk_index,
                text,
                image_paths,
                metadata,
            ),
        )

        self.connection.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_chunk(
        self,
        vector_id: int,
    ) -> Optional[sqlite3.Row]:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM chunks
            WHERE vector_id=?
            """,
            (vector_id,),
        )

        return cursor.fetchone()

    def get_chunks_by_file(
        self,
        file_hash: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM chunks
            WHERE file_hash=?
            ORDER BY page, chunk_index
            """,
            (file_hash,),
        )

        return cursor.fetchall()

    def get_document(
        self,
        file_hash: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE file_hash=?
            """,
            (file_hash,),
        )

        return cursor.fetchone()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def total_documents(self) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM documents
            """
        )

        return cursor.fetchone()[0]

    def total_chunks(self) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        )

        return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(
        self,
        file_hash: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            DELETE FROM chunks
            WHERE file_hash=?
            """,
            (file_hash,),
        )

        cursor.execute(
            """
            DELETE FROM documents
            WHERE file_hash=?
            """,
            (file_hash,),
        )



        cursor.execute(
            """CREATE TABLE IF NOT EXISTS index_info (

    id INTEGER PRIMARY KEY CHECK(id = 1),

    embedding_dimension INTEGER,

    total_vectors INTEGER,

    trained INTEGER,

    last_vector_id INTEGER,

    created_at TEXT,

    updated_at TEXT"""
)
        self.connection.commit()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):

        self.connection.close()