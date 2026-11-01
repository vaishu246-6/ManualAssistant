"""
EquipAI — Phase 3: Hybrid Retriever
Combines dense semantic search (FAISS) with keyword search (BM25)
for better accuracy on part numbers, model codes, and exact terms.

Why hybrid?
  - FAISS alone misses: "J55 connector", "0558008880", "TP3 test point"
    (exact codes that semantic search doesn't handle well)
  - BM25 alone misses: paraphrased questions, synonyms, context
  - Combined: best of both worlds

Usage:
    from retriever import retrieve, build_prompt
    results = retrieve("gas pressure for stainless steel", top_k=5)
    prompt  = build_prompt("gas pressure for stainless steel", results)
"""

import json
import math
import logging
import re
from pathlib import Path
from collections import Counter

import numpy as np

from config import INDEX_FILE, CHUNKS_FILE

logger = logging.getLogger("equipai.retriever")

# ── Lazy-load heavy models once ───────────────────────────────────────────────
_faiss_index  = None
_chunks       = None
_embed_model  = None


def _load_resources():
    global _faiss_index, _chunks, _embed_model
    if _faiss_index is not None:
        return

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_FILE}.\n"
            "Run: python ingest.py data/0-5691AB_1.pdf"
        )
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"chunks.json not found at {CHUNKS_FILE}.")

    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info("Loading FAISS index...")
    _faiss_index = faiss.read_index(str(INDEX_FILE))

    logger.info("Loading chunks metadata...")
    _chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    logger.info("Loading embedding model...")
    _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info(
        "Resources loaded: %d vectors, %d chunks",
        _faiss_index.ntotal, len(_chunks)
    )


# ── BM25 implementation (no extra dependencies) ───────────────────────────────

class BM25:
    """Lightweight BM25 scorer — no external libraries needed."""
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size   = len(corpus)
        self.avgdl         = sum(len(d.split()) for d in corpus) / max(len(corpus), 1)
        self.doc_freqs     = []   # term → count per document
        self.idf           = {}   # term → IDF score
        self.doc_len       = []   # token count per document

        df: dict[str, int] = {}
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for term in freq:
                df[term] = df.get(term, 0) + 1

        # IDF with smoothing
        for term, n in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - n + 0.5) / (n + 0.5) + 1
            )

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, keep part-number tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        score  = 0.0
        dl     = self.doc_len[doc_idx]
        freq   = self.doc_freqs[doc_idx]
        for term in tokens:
            if term not in self.idf:
                continue
            tf   = freq.get(term, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * (tf * (self.k1 + 1)) / denom
        return score

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:k] if s > 0]


_bm25: BM25 | None = None

def _get_bm25() -> BM25:
    global _bm25
    if _bm25 is None:
        _load_resources()
        logger.info("Building BM25 index over %d chunks...", len(_chunks))
        corpus = [c.get("content", "") for c in _chunks]
        _bm25  = BM25(corpus)
    return _bm25


# ── Dense retrieval (FAISS) ───────────────────────────────────────────────────

def _dense_retrieve(query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_idx, cosine_score) pairs from FAISS."""
    _load_resources()
    q_vec = _embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, idxs = _faiss_index.search(q_vec, top_k * 2)
    return [(int(idx), float(score))
            for idx, score in zip(idxs[0], scores[0])
            if idx >= 0]


# ── Sparse retrieval (BM25) ───────────────────────────────────────────────────

def _sparse_retrieve(query: str, top_k: int) -> list[tuple[int, float]]:
    """Return (chunk_idx, bm25_score) pairs."""
    bm25 = _get_bm25()
    results = bm25.top_k(query, k=top_k * 2)
    # Normalise BM25 scores to [0, 1]
    if not results:
        return []
    max_score = max(s for _, s in results) or 1.0
    return [(idx, score / max_score) for idx, score in results]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_merge(
    dense_results:  list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    dense_weight:   float = 0.7,
    sparse_weight:  float = 0.3,
    k:              int   = 60,
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = dense_weight / (k + rank_dense) + sparse_weight / (k + rank_sparse)
    """
    scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + dense_weight / (k + rank + 1)

    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + sparse_weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5, use_hybrid: bool = True) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:      Natural language question from the technician
        top_k:      Number of chunks to return
        use_hybrid: True = FAISS + BM25 merged; False = FAISS only

    Returns:
        List of chunk dicts: {score, page, section, content, image_refs, is_table, source}
    """
    _load_resources()

    if use_hybrid:
        dense  = _dense_retrieve(query, top_k)
        sparse = _sparse_retrieve(query, top_k)
        merged = _rrf_merge(dense, sparse)
        top    = merged[:top_k]
    else:
        top = _dense_retrieve(query, top_k)

    results = []
    for idx, score in top:
        if idx < 0 or idx >= len(_chunks):
            continue
        c = _chunks[idx]
        results.append({
            "score":      float(score),
            "page":       c.get("page", "?"),
            "section":    c.get("section", ""),
            "content":    c.get("content", ""),
            "image_refs": c.get("image_refs", []),
            "is_table":   c.get("is_table", False),
            "source":     c.get("source", "text"),
        })

    logger.debug("Retrieved %d chunks for: %s", len(results), query[:60])
    return results


def build_prompt(query: str, chunks: list[dict], history: list[dict] = None) -> str:
    """
    Build the full RAG prompt for Mistral.

    Args:
        query:   Current question
        chunks:  Retrieved context chunks
        history: Recent chat turns [{role, content}] for multi-turn context
    """
    # Context blocks
    context_parts = []
    for i, c in enumerate(chunks, 1):
        tag = "[TABLE]" if c.get("is_table") else "[TEXT]"
        src = "Vision" if c.get("source") in ("llava_vision", "claude_vision") else "Manual"
        header = f"Source {i} | Page {c['page']} | {c['section']} | {src} {tag}"
        context_parts.append(f"[{header}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # Recent chat history (last 3 turns for context)
    history_text = ""
    if history:
        recent = history[-6:]   # last 3 Q&A pairs
        turns  = []
        for h in recent:
            role = "Technician" if h["role"] == "user" else "Assistant"
            turns.append(f"{role}: {h['content'][:300]}")
        history_text = "\n\nRECENT CONVERSATION:\n" + "\n".join(turns)

    return f"""You are EquipAI, an expert technical assistant for the Thermal Dynamics \
Precision 130 Plasma Cutting System (Manual 0-5691).

RULES:
- Answer ONLY from the manual excerpts below. Do not use outside knowledge.
- If the answer is not in the excerpts, say: "I don't find that in the manual. \
Please check [suggest relevant section] or contact Thermal Dynamics support."
- Always cite your source as "Page X" at the end of your answer.
- Use numbered steps for procedures.
- Start safety warnings with ⚠️ WARNING:.
- Keep answers concise and practical for a technician on the shop floor.
{history_text}

MANUAL EXCERPTS:
{context}

TECHNICIAN'S QUESTION: {query}

ANSWER:"""


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    test_queries = [
        "What is the maximum output current?",
        "How do I replace the torch consumables?",
        "What gas pressure for cutting stainless steel?",
        "Where is connector J55?",
        "What does TP3 test point measure?",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        results = retrieve(q, top_k=3)
        for r in results:
            print(f"  Page {r['page']:3} | score={r['score']:.3f} | {r['section'][:40]}")
            print(f"  {r['content'][:120].strip()}...")