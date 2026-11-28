"""Hybrid and vector retrieval over ingested insurance policy chunks.

Supports dense cosine similarity search via pgvector, hybrid search
combining pgvector with Norwegian full-text search (tsvector) via Reciprocal
Rank Fusion (RRF), BM25 word search, and hybrid_bm25 (pgvector and BM25
merged with RRF).
"""

from dataclasses import dataclass
from google import genai
from google.genai import types
from src.ingestion.db import get_conn
from src.retrieval.bm25 import bm25_search
from src.llm import EMBED_MODEL, EMBED_DIM

client = genai.Client()

MODES = ("vector", "hybrid", "bm25", "hybrid_bm25")
RRF_K = 60
CANDIDATES = 30  # chunks per side before fusion


def rrf_fuse(*rankings: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked lists of chunk ids by Reciprocal Rank Fusion.

    Each id scores the sum of 1 / (k + rank) over the lists it appears in.
    Returns (id, score) pairs, best first, ties broken by id.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass
class RetrievedChunk:
    """A retrieved policy chunk with source metadata and ranking score."""

    id: int
    doc_name: str
    page_number: int
    content: str
    score: float


def embed_query(query: str) -> list[float]:
    """Generate an EMBED_DIM-dim RETRIEVAL_QUERY embedding for a user question."""
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return res.embeddings[0].values


def search(
    query: str, mode: str = "hybrid", top_k: int = 5, query_vector: list[float] | None = None
) -> list[RetrievedChunk]:
    """Retrieve top_k relevant policy chunks using 'vector', 'hybrid' (RRF), 'bm25' or 'hybrid_bm25' mode.

    Pass query_vector to reuse a saved embedding instead of calling the API.
    'bm25' needs no embedding and can return fewer than top_k chunks.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown search mode {mode!r}. Use one of {', '.join(MODES)}.")
    if mode == "bm25":
        return [RetrievedChunk(*r) for r in bm25_search(query, top_k)]
    q_vec = embed_query(query) if query_vector is None else query_vector
    if mode == "hybrid_bm25":
        with get_conn() as conn:
            vec_rows = conn.execute(
                """
                SELECT id, doc_name, page_number, content
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (q_vec, CANDIDATES),
            ).fetchall()
        word_rows = bm25_search(query, CANDIDATES)
        rows_by_id = {r[0]: r[:4] for r in [*vec_rows, *word_rows]}
        fused = rrf_fuse([r[0] for r in vec_rows], [r[0] for r in word_rows])
        return [RetrievedChunk(*rows_by_id[i], score) for i, score in fused[:top_k]]
    with get_conn() as conn:
        if mode == "vector":
            rows = conn.execute(
                """
                SELECT id, doc_name, page_number, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (q_vec, q_vec, top_k),
            ).fetchall()
        else:  # hybrid via Reciprocal Rank Fusion (RRF)
            rows = conn.execute(
                """
                WITH vec AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS r
                    FROM chunks LIMIT 30
                ),
                fts AS (
                    SELECT id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('norwegian', %s)) DESC
                    ) AS r
                    FROM chunks
                    WHERE content_tsv @@ plainto_tsquery('norwegian', %s)
                    LIMIT 30
                ),
                rrf AS (
                    SELECT COALESCE(vec.id, fts.id) AS id,
                           COALESCE(1.0 / (60 + vec.r), 0.0) + COALESCE(1.0 / (60 + fts.r), 0.0) AS score
                    FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
                )
                SELECT c.id, c.doc_name, c.page_number, c.content, rrf.score
                FROM rrf JOIN chunks c ON c.id = rrf.id
                ORDER BY rrf.score DESC
                LIMIT %s;
                """,
                (q_vec, query, query, top_k),
            ).fetchall()
    return [RetrievedChunk(*r) for r in rows]