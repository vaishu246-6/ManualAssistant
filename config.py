"""
EquipAI — Phase 1 Configuration
All settings in one place. Edit this file to change behaviour.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
OUTPUT_DIR      = BASE_DIR / "output"
IMAGES_DIR      = OUTPUT_DIR / "images"
INDEX_FILE      = OUTPUT_DIR / "faiss.index"
CHUNKS_FILE     = OUTPUT_DIR / "chunks.json"
DB_FILE         = OUTPUT_DIR / "equipai.db"
LOG_FILE        = OUTPUT_DIR / "ingestion.log"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 400       # words per chunk
CHUNK_OVERLAP   = 80        # words of overlap between adjacent chunks
MIN_CHUNK_WORDS = 40        # skip chunks shorter than this (headers, footers)

# ── Images ────────────────────────────────────────────────────────────────────
IMAGE_MIN_AREA  = 8000      # px² — skip tiny decorative images below this size
IMAGE_DPI       = 150       # DPI when rasterizing pages for thumbnails

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH     = 64        # chunks per embedding batch

# ── Section detection regex ───────────────────────────────────────────────────
# Matches: "3.12 Gas Connections", "SECTION 4", "APPENDIX 2:", "SAFETY PRECAUTIONS"
SECTION_PATTERN = (
    r"^("
    r"\d+\.\d+[\s\-–—]+[A-Z][A-Za-z\s/\-&()]{4,60}"   # 3.12 Gas Connections
    r"|SECTION\s+\d+[:\s—].*"                             # SECTION 4: ...
    r"|APPENDIX\s+\d+[:\s:—].*"                           # APPENDIX 2: CPU PCB
    r"|[A-Z][A-Z\s]{8,50}"                                # ALL CAPS headings
    r")$"
)

# ── Pages to skip ─────────────────────────────────────────────────────────────
# These are confirmed "This Page Intentionally Blank" pages in manual 0-5691AB
BLANK_PAGES     = { 15, 68, 91}   # 1-indexed

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTS  = {".pdf", ".docx"}