"""
EquipAI — Phase 1: Ingestion Pipeline
Main entry point. Run this to process a PDF manual end-to-end.

Usage:
    python ingest.py data/0-5691AB_1.pdf
    python ingest.py data/0-5691AB_1.pdf --skip-embed   (extract only, no FAISS)
    python ingest.py --test                               (run a test query)

What it does:
    1. Validates the PDF
    2. Extracts text, tables, and images from every page
    3. Detects section headings and tags each chunk
    4. Chunks text with sliding-window overlap
    5. Saves chunks to SQLite + chunks.json
    6. Embeds all chunks with sentence-transformers
    7. Builds and saves a FAISS index
"""

import sys
import time
import json
import logging
from pathlib import Path

from config import (
    OUTPUT_DIR, IMAGES_DIR, INDEX_FILE, CHUNKS_FILE, DB_FILE,
    LOG_FILE, SUPPORTED_EXTS
)
from database  import init_db, insert_document, update_document_status, insert_chunks_bulk
from extractor import extract_pdf, build_chunks
from embedder  import embed_chunks, build_faiss_index, save_chunks_metadata, retrieve, build_rag_prompt


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
        handlers= handlers,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pdf(path: Path) -> None:
    """Raise ValueError if the file is missing or unsupported."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {SUPPORTED_EXTS}")
    if path.stat().st_size < 1024:
        raise ValueError(f"File too small to be a valid PDF: {path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_ingestion(pdf_path: Path, skip_embed: bool = False):
    logger = logging.getLogger("equipai.ingest")
    t_start = time.time()

    print("\n" + "="*60)
    print("  EquipAI — Phase 1 Ingestion")
    print("="*60)

    # ── Step 1: Validate ──────────────────────────────────────────────────────
    print(f"\n[1/5] Validating: {pdf_path.name}")
    validate_pdf(pdf_path)
    print(f"      ✓ File OK ({pdf_path.stat().st_size / 1024:.0f} KB)")

    # ── Step 2: Init database ─────────────────────────────────────────────────
    print("\n[2/5] Initialising database...")
    init_db()
    doc_id = insert_document(pdf_path.name, str(pdf_path))
    print(f"      ✓ Document registered (id={doc_id})")

    # ── Step 3: Extract content ───────────────────────────────────────────────
    print(f"\n[3/5] Extracting content from PDF...")
    try:
        pages = extract_pdf(pdf_path)
        chunks_obj = build_chunks(pages)
    except Exception as exc:
        update_document_status(doc_id, "error")
        logger.error("Extraction failed: %s", exc, exc_info=True)
        raise

    n_pages  = len(pages)
    n_chunks = len(chunks_obj)
    n_images = len(list(IMAGES_DIR.iterdir())) if IMAGES_DIR.exists() else 0
    n_tables = sum(1 for c in chunks_obj if c.is_table)

    print(f"      ✓ Pages processed : {n_pages}")
    print(f"      ✓ Chunks created  : {n_chunks}")
    print(f"        - Text chunks   : {n_chunks - n_tables}")
    print(f"        - Table chunks  : {n_tables}")
    print(f"      ✓ Images saved    : {n_images}")

    # ── Step 4: Save to DB + JSON ─────────────────────────────────────────────
    print("\n[4/5] Saving chunks to database and JSON...")
    db_rows = [
        {
            "doc_id":      doc_id,
            "chunk_index": c.chunk_index,
            "page":        c.page,
            "section":     c.section,
            "content":     c.content,
            "image_refs":  json.dumps(c.image_refs),
            "is_table":    int(c.is_table),
            "word_count":  c.word_count,
        }
        for c in chunks_obj
    ]
    insert_chunks_bulk(db_rows)
    save_chunks_metadata(chunks_obj)
    update_document_status(doc_id, "done", page_count=n_pages, chunk_count=n_chunks)
    print(f"      ✓ SQLite DB  : {DB_FILE}")
    print(f"      ✓ Chunks JSON: {CHUNKS_FILE}")

    # ── Step 5: Embed + FAISS ─────────────────────────────────────────────────
    if skip_embed:
        print("\n[5/5] Skipping embedding (--skip-embed flag set)")
        print("\n      Run without --skip-embed to build the FAISS index.")
    else:
        print(f"\n[5/5] Embedding {n_chunks} chunks and building FAISS index...")
        try:
            embeddings = embed_chunks(chunks_obj)
            build_faiss_index(embeddings)
            print(f"      ✓ FAISS index: {INDEX_FILE}")
        except ImportError:
            print("      ✗ sentence-transformers or faiss-cpu not installed.")
            print("        Run: pip install sentence-transformers faiss-cpu")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✓ Ingestion complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"    faiss.index  — vector search index")
    print(f"    chunks.json  — chunk text + metadata")
    print(f"    equipai.db   — SQLite (documents, chunks, chat history)")
    print(f"    images/      — {n_images} embedded images")
    print(f"    ingestion.log\n")

    return chunks_obj


# ── Test query ────────────────────────────────────────────────────────────────

def run_test_queries():
    """Run a few test queries to verify the pipeline works end-to-end."""
    test_questions = [
        "What is the maximum output current of the Precision 130?",
        "How do I replace the torch consumables?",
        "What gas pressure should I use for cutting stainless steel?",
        "How do I install the pilot lead?",
        "What does the red LED indicate on the front panel?",
    ]

    print("\n" + "="*60)
    print("  EquipAI — Phase 1 Test Queries")
    print("="*60)

    for q in test_questions:
        print(f"\n  Q: {q}")
        results = retrieve(q, top_k=3)
        if results:
            top = results[0]
            print(f"  → Page {top['page']} | {top['section']} | score={top['score']:.3f}")
            print(f"  → {top['content'][:200].strip()}...")
            if top["image_refs"]:
                print(f"  → Images: {top['image_refs']}")
        else:
            print("  → No results found.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()

    if "--test" in sys.argv:
        run_test_queries()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/manual.pdf> [--skip-embed]")
        print("       python ingest.py --test")
        sys.exit(1)

    pdf_path   = Path(sys.argv[1])
    skip_embed = "--skip-embed" in sys.argv

    run_ingestion(pdf_path, skip_embed=skip_embed)