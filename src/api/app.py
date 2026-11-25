"""FastAPI service exposing the insurance RAG pipeline and user feedback logging.

Endpoints:
    GET  /health   - Verifies Postgres connectivity.
    POST /ask      - Runs hybrid/vector RAG and returns grounded answer + citations.
    POST /feedback - Logs thumbs-up/down user feedback to Postgres for eval loops.

Usage:
    uvicorn src.api.app:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.ingestion.db import get_conn, init_schema
from src.generation.rag import ask, Citation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure DB schema and feedback table exist on startup."""
    init_schema()
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT now(),
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INT NOT NULL CHECK (rating IN (-1, 1)),
            comment TEXT
        );
        """)
    yield


app = FastAPI(title="Insurance RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    mode: str = Field(default="hybrid", pattern="^(vector|hybrid)$")
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    is_covered_in_context: bool
    answer: str
    citations: list[Citation]
    latency_ms: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., description="1 for thumbs-up, -1 for thumbs-down")
    comment: str | None = None


@app.get("/health")
def health():
    """Verify database connectivity."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1;")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """Answer an insurance policy question with grounded citations."""
    ans, _, ms = ask(req.question, mode=req.mode, top_k=req.top_k)
    return AskResponse(
        is_covered_in_context=ans.is_covered_in_context,
        answer=ans.answer,
        citations=ans.citations,
        latency_ms=ms,
    )


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """Store user feedback in Postgres for offline error analysis."""
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating must be 1 or -1")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_feedback (question, answer, rating, comment) VALUES (%s, %s, %s, %s);",
            (req.question, req.answer, req.rating, req.comment),
        )
    return {"status": "saved"}