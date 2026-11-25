"""Ingests insurance policy PDFs into Postgres with pgvector embeddings.

Reads all `.pdf` files from `data/policies/`, extracts text page-by-page using
PyMuPDF, splits text into overlapping character chunks, generates
`RETRIEVAL_DOCUMENT` embeddings (EMBED_DIM dims) via the Gemini API, and stores chunks, page
metadata, and embeddings in the `chunks` table (with a generated Norwegian
`tsvector` column for hybrid search).

Usage:
    python -m src.ingestion.ingest
"""

from pathlib import Path
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from src.ingestion.db import get_conn, init_schema
from src.llm import EMBED_MODEL, EMBED_DIM

client = genai.Client()
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_batch(texts: list[str]) -> list[list[float]]:
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in res.embeddings]

def ingest_policies(pdf_dir: Path = Path("data/policies")):
    init_schema()
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    assert pdfs, f"No PDFs found in {pdf_dir}"

    with get_conn() as conn:
        conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")
        for pdf in pdfs:
            doc = fitz.open(pdf)
            rows = []
            for page_idx, page in enumerate(doc, start=1):
                for c_idx, chunk in enumerate(chunk_text(page.get_text("text"))):
                    rows.append((pdf.name, page_idx, c_idx, chunk))

            for i in range(0, len(rows), 50):
                batch = rows[i : i + 50]
                vecs = embed_batch([r[3] for r in batch])
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (doc_name, page_number, chunk_index, content, embedding) VALUES (%s, %s, %s, %s, %s)",
                        [(r[0], r[1], r[2], r[3], v) for r, v in zip(batch, vecs)],
                    )
            print(f"Ingested {pdf.name}: {len(rows)} chunks")

if __name__ == "__main__":
    ingest_policies()
