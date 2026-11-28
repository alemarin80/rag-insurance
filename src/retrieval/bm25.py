"""BM25 word search over the policy chunks.

Chunks and queries are processed alike: lowercase, Norwegian stop words removed,
Norwegian Snowball stemming. The index is built in memory from the `chunks`
table on first use, once per process.

Spec: docs/specs/step1-bm25.md
"""

import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_NORWEGIAN

from src.ingestion.db import get_conn

SETTINGS = {"method": "lucene", "k1": 1.5, "b": 0.75}

_stemmer = Stemmer.Stemmer("norwegian")
_index: tuple[bm25s.BM25, list[tuple]] | None = None


def tokenize(texts: list[str]) -> list[list[str]]:
    """Split texts into lowercase, stemmed words without stop words."""
    return bm25s.tokenize(
        texts, stopwords=list(STOPWORDS_NORWEGIAN), stemmer=_stemmer, return_ids=False, show_progress=False
    )


def build_index(rows: list[tuple]) -> tuple[bm25s.BM25, list[tuple]]:
    """Index rows of (id, doc_name, page_number, content)."""
    retriever = bm25s.BM25(**SETTINGS)
    retriever.index(tokenize([r[3] for r in rows]), show_progress=False)
    return retriever, rows


def get_index() -> tuple[bm25s.BM25, list[tuple]]:
    global _index
    if _index is None:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, doc_name, page_number, content FROM chunks ORDER BY id;"
            ).fetchall()
        _index = build_index(rows)
    return _index


def bm25_search(query: str, top_k: int = 5, index: tuple[bm25s.BM25, list[tuple]] | None = None) -> list[tuple]:
    """Return up to top_k rows of (id, doc_name, page_number, content, score) with a score above 0."""
    retriever, rows = index or get_index()
    words = tokenize([query])[0]
    if not words:
        return []
    scores = retriever.get_scores(words)
    ranked = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i][0]))
    return [(*rows[i], float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]
