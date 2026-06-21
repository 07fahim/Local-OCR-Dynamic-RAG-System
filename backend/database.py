"""
SQLite ingestion log.

Tracks one row per uploaded document: provenance, counts, detected languages,
the user-supplied manual metadata, and processing status. This is the
operational record that powers GET /documents; the chunk vectors themselves
live in ChromaDB (see vectorstore.py).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    document_id        TEXT PRIMARY KEY,
    filename           TEXT NOT NULL,
    upload_timestamp   TEXT NOT NULL,
    page_count         INTEGER,
    chunk_count        INTEGER,
    languages_detected TEXT,          -- JSON array, e.g. ["bn","en"]
    document_date      TEXT,          -- YYYY-MM-DD or NULL
    document_type      TEXT,
    status             TEXT NOT NULL  -- 'pending' | 'processing' | 'done' | 'error'
);
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(os.path.abspath(settings.sqlite_path)), exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def log_ingestion(
    document_id: str,
    filename: str,
    upload_timestamp: str,
    page_count: int,
    chunk_count: int,
    languages_detected: list[str],
    document_date: str | None,
    document_type: str,
    status: str,
) -> None:
    """Insert or replace the row for a document (idempotent on document_id)."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_log
                (document_id, filename, upload_timestamp, page_count, chunk_count,
                 languages_detected, document_date, document_type, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename=excluded.filename,
                upload_timestamp=excluded.upload_timestamp,
                page_count=excluded.page_count,
                chunk_count=excluded.chunk_count,
                languages_detected=excluded.languages_detected,
                document_date=excluded.document_date,
                document_type=excluded.document_type,
                status=excluded.status
            """,
            (
                document_id,
                filename,
                upload_timestamp,
                page_count,
                chunk_count,
                json.dumps(languages_detected),
                document_date,
                document_type,
                status,
            ),
        )


def set_status(document_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE ingestion_log SET status=? WHERE document_id=?",
            (status, document_id),
        )


def get_all_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ingestion_log ORDER BY upload_timestamp DESC"
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        try:
            d["languages_detected"] = json.loads(d.get("languages_detected") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["languages_detected"] = []
        out.append(d)
    return out
