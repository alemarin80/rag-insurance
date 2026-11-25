"""PostgreSQL connection and schema setup for the insurance RAG pipeline.

Initializes the `pgvector` extension and creates two tables:
- `chunks`: Stores policy text, metadata, a vector embedding (EMBED_DIM dims), and a
  generated Norwegian `tsvector` (stemmed keywords). Uses two indexes for
  hybrid search (keyword search + semantic search + their ranking):
    - GIN (Generalized Inverted Index): Maps each Norwegian word to the chunk
      IDs containing it for fast exact/keyword matching.
    - HNSW (Hierarchical Navigable Small World): Graph-based approximate nearest
      neighbor index for fast semantic similarity search over vector embeddings.
- `eval_logs`: Stores evaluation runs, citations, latency, and LLM-judge scores.
"""

import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from src.llm import EMBED_DIM

load_dotenv()

def get_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    return conn

def init_schema():
    with get_conn() as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,
            page_number INT NOT NULL,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('norwegian', content)) STORED,
            embedding vector({EMBED_DIM}) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (content_tsv);
        CREATE INDEX IF NOT EXISTS idx_chunks_vec ON chunks USING hnsw (embedding vector_cosine_ops);

        CREATE TABLE IF NOT EXISTS eval_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT now(),
            pipeline_version TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_source TEXT,
            answer TEXT NOT NULL,
            citations JSONB,
            faithfulness_score FLOAT,
            relevance_score FLOAT,
            judge_reason TEXT,
            latency_ms INT
        );
        """)
    print("Schema initialized.")

if __name__ == "__main__":
    init_schema()