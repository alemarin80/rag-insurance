"""Grounded RAG answer generation with structured citations and refusal handling.

Retrieves relevant policy chunks and prompts the configured LLM (see src/llm.py) to produce a Pydantic-validated
response with exact document/page citations, refusing when context is insufficient.
"""

import time
from pydantic import BaseModel, Field
from src.llm import generate_structured
from src.retrieval.search import search, RetrievedChunk


class Citation(BaseModel):
    """Source reference pointing to a specific policy document and page."""

    doc_name: str
    page_number: int
    quote: str = Field(description="Short exact phrase from the chunk supporting the answer")


class RagAnswer(BaseModel):
    """Structured RAG output enforcing grounding and explicit refusal."""

    is_covered_in_context: bool = Field(
        description="False if the policy context does not answer the question"
    )
    answer: str = Field(
        description="Concise answer in the same language as the question, or explicit refusal"
    )
    citations: list[Citation]


def ask(
    question: str, mode: str = "hybrid", top_k: int = 5
) -> tuple[RagAnswer, list[RetrievedChunk], int]:
    """Run retrieval + grounded generation, returning (answer, chunks, latency_ms)."""
    t0 = time.perf_counter()
    chunks = search(question, mode=mode, top_k=top_k)
    context = "\n\n".join(
        f"[{c.doc_name} | side {c.page_number}]\n{c.content}" for c in chunks
    )

    prompt = f"""You are an insurance policy assistant.
Answer ONLY using the policy excerpts below. If the excerpts do not contain enough
information to answer reliably, set is_covered_in_context=false and state that it is not found in the policy excerpts.

Policy excerpts:
{context}

Question: {question}"""

    # fast=True: low thinking to speed up generation (Gemini only)
    answer = generate_structured(prompt, RagAnswer, role="llm", temperature=0.0, fast=True)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return answer, chunks, latency_ms


if __name__ == "__main__":
    ans, _, ms = ask("Hvem gjelder bilforsikringen for?")
    print(f"[{ms}ms] Covered={ans.is_covered_in_context}\n{ans.answer}\nCitations: {ans.citations}")