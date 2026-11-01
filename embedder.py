"""
EquipAI — Phase 1: Embedder & FAISS Index
Embeds text chunks using sentence-transformers and builds a FAISS index.

Uses cosine similarity (IndexFlatIP on L2-normalised vectors).
The index is saved to disk — rebuilding is not needed on every run.
"""

import json
import logging
import numpy as np
from pathlib import Path

from config import EMBED_MODEL, EMBED_BATCH, INDEX_FILE, CHUNKS_FILE, OUTPUT_DIR
from extractor import Chunk

logger = logging.getLogger("equipai.embedder")


def _get_model():
    """Lazy-load the embedding model (avoids slow import at module level)."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """
    Embed all chunks and return L2-normalised vectors as float32 array.
    Shape: (N, embedding_dim)
    """
    model  = _get_model()
    texts  = [c.content for c in chunks]

    logger.info("Embedding %d chunks in batches of %d...", len(texts), EMBED_BATCH)
    embeddings = model.encode(
        texts,
        batch_size        = EMBED_BATCH,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True,   # L2-normalise → cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build and save a FAISS IndexFlatIP (exact cosine search)."""
    import faiss

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    logger.info("FAISS index saved: %s  (%d vectors, dim=%d)", INDEX_FILE, index.ntotal, dim)


def save_chunks_metadata(chunks: list[Chunk]):
    """Serialise chunk metadata to JSON (keeps index and metadata in sync)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for c in chunks:
        records.append({
            "chunk_index": c.chunk_index,
            "page":        c.page,
            "section":     c.section,
            "content":     c.content,
            "image_refs":  c.image_refs,
            "is_table":    c.is_table,
            "word_count":  c.word_count,
        })
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("Chunk metadata saved: %s  (%d chunks)", CHUNKS_FILE, len(records))


def load_chunks_metadata() -> list[dict]:
    """Load chunk metadata from JSON (for retrieval at query time)."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"chunks.json not found at {CHUNKS_FILE}. Run ingest.py first.")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index():
    """Load the FAISS index from disk."""
    import faiss
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}. Run ingest.py first.")
    index = faiss.read_index(str(INDEX_FILE))
    logger.info("FAISS index loaded: %d vectors", index.ntotal)
    return index


# ── Retrieval (used in Phase 3 — defined here for convenience) ─────────────────

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed a query and return the top_k most relevant chunks.

    Returns list of dicts:
      { score, page, section, content, image_refs, is_table }
    """
    model  = _get_model()
    index  = load_faiss_index()
    chunks = load_chunks_metadata()

    q_vec = model.encode(
        [query],
        normalize_embeddings = True,
        convert_to_numpy     = True,
    ).astype(np.float32)

    scores, idxs = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append({
            "score":      float(score),
            "page":       c["page"],
            "section":    c["section"],
            "content":    c["content"],
            "image_refs": c["image_refs"],
            "is_table":   c["is_table"],
        })
    return results


def build_rag_prompt(query: str, results: list[dict]) -> str:
    """
    Build the full RAG prompt to send to your LLM (Mistral / Claude).
    Each context block is numbered and includes page + section for citation.
    """
    context_blocks = []
    for i, r in enumerate(results, 1):
        header = f"[Source {i} | Page {r['page']} | {r['section']}]"
        context_blocks.append(f"{header}\n{r['content']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are EquipAI, an expert technical assistant for the \
Thermal Dynamics Precision 130 Plasma Cutting System (Manual 0-5691).

RULES:
- Answer ONLY based on the manual excerpts provided below.
- If the answer is not in the excerpts, say: "I don't find that in the manual. \
Please check sections [list relevant sections] or contact Thermal Dynamics support."
- Always cite your source as "Page X, [Section name]".
- Use clear, numbered steps for procedures.
- Highlight safety warnings prominently.

MANUAL EXCERPTS:
{context}

TECHNICIAN'S QUESTION:
{query}

ANSWER:"""


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if "--test" in sys.argv:
        print("\nTest query: 'How do I replace the torch consumables?'")
        results = retrieve("How do I replace the torch consumables?", top_k=3)
        for r in results:
            print(f"\n  Score={r['score']:.3f} | Page {r['page']} | {r['section']}")
            print(f"  {r['content'][:250]}...")
            if r["image_refs"]:
                print(f"  Images: {r['image_refs']}")