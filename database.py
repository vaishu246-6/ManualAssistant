"""
EquipAI — Phase 1: Database
Sets up SQLite schema for documents, chunks, and chat history.
"""

import sqlite3
import logging
from pathlib import Path
from config import DB_FILE

logger = logging.getLogger("equipai.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory enabled."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist yet."""
    conn = get_connection()
    cur = conn.cursor()

    # ── Documents ─────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL UNIQUE,
            filepath    TEXT    NOT NULL,
            page_count  INTEGER,
            chunk_count INTEGER,
            status      TEXT    DEFAULT 'pending',   -- pending | done | error
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Chunks ────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL REFERENCES documents(id),
            chunk_index INTEGER NOT NULL,
            page        INTEGER NOT NULL,
            section     TEXT,
            content     TEXT    NOT NULL,
            image_refs  TEXT,   -- JSON array of image filenames
            is_table    INTEGER DEFAULT 0,
            word_count  INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Chat history (used from Phase 3 onwards) ──────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            role         TEXT    NOT NULL,   -- 'user' | 'assistant'
            content      TEXT    NOT NULL,
            retrieved_pages TEXT,            -- JSON: pages used for RAG
            latency_ms   INTEGER,
            feedback     INTEGER DEFAULT 0,  -- 0=none, 1=thumbs up, -1=thumbs down
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page  ON chunks(page)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_FILE)


def insert_document(filename: str, filepath: str) -> int:
    """Insert a document record and return its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO documents (filename, filepath) VALUES (?, ?)",
        (filename, str(filepath))
    )
    conn.commit()
    doc_id = cur.execute(
        "SELECT id FROM documents WHERE filename = ?", (filename,)
    ).fetchone()["id"]
    conn.close()
    return doc_id


def update_document_status(doc_id: int, status: str, page_count: int = 0, chunk_count: int = 0):
    conn = get_connection()
    conn.execute(
        "UPDATE documents SET status=?, page_count=?, chunk_count=? WHERE id=?",
        (status, page_count, chunk_count, doc_id)
    )
    conn.commit()
    conn.close()


def insert_chunks_bulk(chunks: list[dict]):
    """Bulk-insert chunks into the database."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO chunks
           (doc_id, chunk_index, page, section, content, image_refs, is_table, word_count)
           VALUES (:doc_id, :chunk_index, :page, :section, :content,
                   :image_refs, :is_table, :word_count)""",
        chunks
    )
    conn.commit()
    conn.close()


def get_all_chunks() -> list[dict]:
    """Fetch all chunks as dicts (used when re-building the FAISS index)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, doc_id, chunk_index, page, section, content, image_refs FROM chunks ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("✓ Database ready at", DB_FILE)