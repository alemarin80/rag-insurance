"""Fast unit tests for chunking, schemas, and API validation (no API credits used)."""

from fastapi.testclient import TestClient
from src.ingestion.ingest import chunk_text
from src.generation.rag import RagAnswer, Citation
from src.api.app import app

client = TestClient(app)


def test_chunk_text_splits_with_overlap():
    text = " ".join(["forsikring"] * 300)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)


def test_rag_answer_schema_refusal():
    ans = RagAnswer(
        is_covered_in_context=False,
        answer="Ikke funnet i vilkårene.",
        citations=[],
    )
    assert ans.is_covered_in_context is False
    assert ans.citations == []


def test_rag_answer_schema_with_citation():
    ans = RagAnswer(
        is_covered_in_context=True,
        answer="Egenandelen er 4000 kr.",
        citations=[Citation(doc_name="bil.pdf", page_number=2, quote="Egenandel 4000 kr")],
    )
    assert ans.citations[0].page_number == 2


def test_feedback_validation_rejects_bad_rating():
    res = client.post("/feedback", json={"question": "q", "answer": "a", "rating": 5})
    assert res.status_code == 422