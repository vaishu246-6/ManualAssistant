"""
EquipAI — Phase 1: PDF Extractor
Extracts text, tables, and images from every page of the manual.

Handles:
  - Digital-native text (PyMuPDF, reading-order sorted)
  - Tables (pdfplumber — cut charts, spec sheets, parts lists)
  - Embedded images (raster photos, PCB diagrams)
  - Section heading detection
  - Blank page filtering
"""

import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber

from config import (
    IMAGES_DIR, BLANK_PAGES, SECTION_PATTERN,
    IMAGE_MIN_AREA, MIN_CHUNK_WORDS, CHUNK_SIZE, CHUNK_OVERLAP
)

logger = logging.getLogger("equipai.extractor")
SECTION_RE = re.compile(SECTION_PATTERN, re.MULTILINE)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PageData:
    """All extracted data for a single PDF page."""
    page_num:   int
    text:       str
    tables:     list[str]          = field(default_factory=list)
    image_refs: list[str]          = field(default_factory=list)
    section:    str                = "General"
    word_count: int                = 0


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""
    chunk_index: int
    page:        int
    section:     str
    content:     str
    image_refs:  list[str]         = field(default_factory=list)
    is_table:    bool              = False
    word_count:  int               = 0


# ── Text utilities ────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Remove form-feeds, collapse whitespace, strip footers/headers."""
    text = raw.replace("\f", "\n")
    # Remove running headers like "PRECISION 130" and footers "0-5691  INSTALLATION  42"
    text = re.sub(r"^PRECISION\s+130\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d[\d\-]+\s+(INSTALLATION|OPERATION|MAINTENANCE|APPENDIX|SAFETY)\s+\d+\s*$",
                  "", text, flags=re.MULTILINE)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def detect_section(text: str, previous: str) -> str:
    """Find the last section heading in a page's text."""
    matches = SECTION_RE.findall(text)
    if matches:
        raw = matches[-1].strip()
        # Collapse multi-line headings (rotated sidebar text creates newlines)
        cleaned = " ".join(raw.split())
        # Remove TOC dot-leaders like "SECTION 6: REPLACEMENT......"
        cleaned = re.sub(r"\.{3,}.*$", "", cleaned).strip()
        return cleaned[:80]
    return previous


def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-count windows."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


# ── Image extraction ──────────────────────────────────────────────────────────

def extract_images_from_page(fitz_doc: fitz.Document, page_num: int) -> list[str]:
    """
    Save all meaningful embedded images from a page.
    Returns a list of saved filenames (relative to IMAGES_DIR).
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    page = fitz_doc[page_num - 1]

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(fitz_doc, xref)

            # Skip tiny decorative images
            if pix.width * pix.height < IMAGE_MIN_AREA:
                pix = None
                continue

            # Convert CMYK / exotic colour spaces to RGB
            if pix.n > 4:
                try:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                except Exception:
                    # Some exotic colour spaces can't convert — save as JPEG instead
                    pass

            fname = f"page{page_num:03d}_img{xref}.png"
            fpath = IMAGES_DIR / fname
            if not fpath.exists():
                try:
                    pix.save(str(fpath))
                except Exception:
                    # Fall back to JPEG for any colorspace PNG can't handle
                    fname = f"page{page_num:03d}_img{xref}.jpg"
                    fpath = IMAGES_DIR / fname
                    if not fpath.exists():
                        pix.save(str(fpath))
                logger.debug("Saved image: %s", fname)
            saved.append(fname)
            pix = None

        except Exception as exc:
            logger.warning("Could not extract image xref=%d page=%d: %s", xref, page_num, exc)

    return saved


# ── Table extraction ──────────────────────────────────────────────────────────

def extract_tables_from_page(plumb_page) -> list[str]:
    """
    Extract tables using pdfplumber. Returns each table as a pipe-delimited string.
    """
    result = []
    try:
        tables = plumb_page.extract_tables()
        for tbl in (tables or []):
            if not tbl:
                continue
            rows = []
            for row in tbl:
                cells = [str(c or "").strip().replace("\n", " ") for c in row]
                rows.append(" | ".join(cells))
            table_str = "\n".join(rows)
            if len(table_str.split()) >= 6:   # skip trivial 1-cell tables
                result.append(table_str)
    except Exception as exc:
        logger.warning("Table extraction failed: %s", exc)
    return result


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str | Path) -> list[PageData]:
    """
    Extract all content from a PDF.
    Returns a list of PageData objects, one per meaningful page.
    """
    pdf_path = Path(pdf_path)
    logger.info("Opening: %s", pdf_path.name)

    fitz_doc = fitz.open(str(pdf_path))
    total_pages = len(fitz_doc)
    pages: list[PageData] = []
    current_section = "General"

    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for i in range(total_pages):
            page_num = i + 1

            # Skip confirmed blank pages
            if page_num in BLANK_PAGES:
                logger.debug("Skipping blank page %d", page_num)
                continue

            fitz_page  = fitz_doc[i]
            plumb_page = plumber_pdf.pages[i]

            # ── Text ──────────────────────────────────────────────────────────
            blocks   = fitz_page.get_text("blocks", sort=True)
            raw_text = "\n".join(b[4].strip() for b in blocks if len(b) > 4 and b[4].strip())
            text     = clean_text(raw_text)

            # Auto-skip pages with almost no text and no images (section dividers)
            has_images = bool(fitz_page.get_images())
            if len(text.split()) < 5 and not has_images:
                logger.debug("Skipping near-empty page %d", page_num)
                continue

            # ── Section ───────────────────────────────────────────────────────
            current_section = detect_section(text, current_section)

            # ── Tables ────────────────────────────────────────────────────────
            tables = extract_tables_from_page(plumb_page)

            # ── Images ────────────────────────────────────────────────────────
            image_refs = extract_images_from_page(fitz_doc, page_num)

            pages.append(PageData(
                page_num   = page_num,
                text       = text,
                tables     = tables,
                image_refs = image_refs,
                section    = current_section,
                word_count = len(text.split()),
            ))

            if page_num % 25 == 0:
                logger.info("  Extracted %d / %d pages...", page_num, total_pages)

    fitz_doc.close()
    logger.info("Extracted %d usable pages (of %d total)", len(pages), total_pages)
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def build_chunks(pages: list[PageData]) -> list[Chunk]:
    """
    Convert page-level data into overlapping text chunks.
    Each chunk preserves: section, page number, image refs.
    Tables get their own dedicated chunk (they don't split well).
    """
    chunks: list[Chunk] = []
    idx = 0

    for page in pages:
        refs_json = page.image_refs   # pass through as list

        # ── Table chunks (one per table, no splitting) ─────────────────────
        for table_text in page.tables:
            words = table_text.split()
            if len(words) < 6:
                continue
            chunks.append(Chunk(
                chunk_index = idx,
                page        = page.page_num,
                section     = page.section,
                content     = f"[TABLE — {page.section}]\n{table_text}",
                image_refs  = refs_json,
                is_table    = True,
                word_count  = len(words),
            ))
            idx += 1

        # ── Text chunks (sliding window) ───────────────────────────────────
        for window in sliding_window(page.text, CHUNK_SIZE, CHUNK_OVERLAP):
            words = window.split()
            if len(words) < MIN_CHUNK_WORDS:
                continue
            # Prepend section context so every chunk is self-contained
            content = f"[{page.section}]\n{window}"
            chunks.append(Chunk(
                chunk_index = idx,
                page        = page.page_num,
                section     = page.section,
                content     = content,
                image_refs  = refs_json,
                is_table    = False,
                word_count  = len(words),
            ))
            idx += 1

    logger.info("Built %d chunks from %d pages", len(chunks), len(pages))
    return chunks


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/0-5691AB_1.pdf"
    pages = extract_pdf(pdf)
    chunks = build_chunks(pages)

    print(f"\nPages extracted : {len(pages)}")
    print(f"Total chunks    : {len(chunks)}")
    print(f"  - Text chunks : {sum(1 for c in chunks if not c.is_table)}")
    print(f"  - Table chunks: {sum(1 for c in chunks if c.is_table)}")
    print(f"Images saved    : {len(list(IMAGES_DIR.iterdir()))}")
    print(f"\nSample chunk (index 5):")
    if len(chunks) > 5:
        c = chunks[5]
        print(f"  Page    : {c.page}")
        print(f"  Section : {c.section}")
        print(f"  Words   : {c.word_count}")
        print(f"  Preview : {c.content[:200]}...")