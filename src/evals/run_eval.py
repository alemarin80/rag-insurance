"""Automated RAG evaluation using a User Simulator Agent and an LLM Judge.

1. User Simulator: Samples random policy chunks from Postgres and generates
   realistic Norwegian policyholder questions (plus unanswerable trap questions).
2. Benchmark: Runs the same eval set against both `vector` and `hybrid` modes.
3. Judge Agent: Scores faithfulness (no hallucinated coverage) and relevance,
   then logs every run to the `eval_logs` table and prints a summary table.

Usage:
    python -m src.evals.run_eval
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
import time
from src.ingestion.db import get_conn
from src.generation.rag import ask
from src.llm import generate_structured


class SimulatedQuery(BaseModel):
    """A synthetic policyholder question generated from a policy chunk."""

    question: str = Field(description="Natural Norwegian question from a policyholder")


class JudgeScore(BaseModel):
    """Structured LLM-judge evaluation of a RAG response."""

    faithfulness_score: float = Field(ge=0.0, le=1.0, description="1.0 if strictly grounded in retrieved chunks")
    relevance_score: float = Field(ge=0.0, le=1.0, description="1.0 if it accurately answers (or properly refuses)")
    reason: str = Field(description="Brief 1-sentence justification")


EVAL_PATH = Path("data/eval_set.json")


def generate_eval_set(n_per_type: int = 5, force_regen: bool = False) -> list[dict]:
    """Load or generate a fixed, challenging Norwegian eval set in data/eval_set.json."""
    if EVAL_PATH.exists() and not force_regen:
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    styles = [
        ("exact_lookup", "Ask a specific question targeting an exact number, age, day limit, kroner amount, or clause condition in the excerpt—without copying the surrounding sentence."),
        ("messy_user", "Write an informal, slightly vague real-life scenario from a stressed Norwegian customer (no insurance jargon) that this excerpt resolves."),
        ("exclusion_edge", "Ask an edge-case question testing whether a specific exception or limitation in the excerpt applies."),
    ]

    eval_set = []
    with get_conn() as conn:
        for style_name, instruction in styles:
            rows = conn.execute(
                "SELECT doc_name, page_number, content FROM chunks ORDER BY RANDOM() LIMIT %s;",
                (n_per_type,),
            ).fetchall()
            for doc_name, page_num, content in rows:
                prompt = (
                    f"You are a Norwegian insurance customer. {instruction}\n"
                    f"Write ONE question in Norwegian.\n\nExcerpt:\n{content}"
                )
                q = generate_structured(
                    prompt, SimulatedQuery, role="judge", temperature=0.7, fast=True
                ).question
                eval_set.append({
                    "type": style_name,
                    "question": q,
                    "expected_source": f"{doc_name}#p{page_num}",
                })

    eval_set.extend([
        {"type": "trap", "question": "Dekker forsikringen tap av kryptovaluta ved nettsvindel?", "expected_source": "NONE"},
        {"type": "trap", "question": "Hva er egenandelen for tannlegebehandling av hund?", "expected_source": "NONE"},
        {"type": "trap", "question": "Gjelder bilforsikringen for elektrisk sparkesykkel på ferie i Spania?", "expected_source": "NONE"},
    ])

    EVAL_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    return eval_set


def judge_answer(question: str, expected_source: str, answer_text: str, is_covered: bool, context_chunks: list) -> JudgeScore:
    """Grade faithfulness and relevance using the configured judge LLM (JUDGE_PROVIDER / JUDGE_MODEL)."""
    context = "\n\n".join(f"[{c.doc_name} p{c.page_number}] {c.content}" for c in context_chunks)
    if expected_source == "NONE":
        criteria = "This is an UNANSWERABLE trap question. Score relevance=1.0 ONLY if the assistant refused / set is_covered_in_context=False."
    else:
        criteria = "Score faithfulness=1.0 if no claims go beyond the retrieved chunks, and relevance=1.0 if it answers the question."

    prompt = (
        f"Evaluate this insurance RAG response.\n{criteria}\n\n"
        f"Question: {question}\n"
        f"Model is_covered_in_context: {is_covered}\n"
        f"Model Answer: {answer_text}\n\n"
        f"Retrieved Context:\n{context}"
    )
    return generate_structured(prompt, JudgeScore, role="judge", temperature=0.0)


def run_benchmark(modes: tuple[str, ...] = ("vector", "hybrid")) -> None:
    """Run eval_set across retrieval modes, log to Postgres, and print summary."""
    eval_set = generate_eval_set(n_per_type=5)
    print(f"Running {len(eval_set)} eval questions.\n")

    for mode in modes:
        recalls, faiths, rels, latencies = [], [], [], []
        with get_conn() as conn:
            for item in eval_set:
                q, exp = item["question"], item["expected_source"]
                ans, chunks, ms = ask(q, mode=mode, top_k=5)
                score = judge_answer(q, exp, ans.answer, ans.is_covered_in_context, chunks)

                if exp != "NONE":
                    hit = any(f"{c.doc_name}#p{c.page_number}" == exp for c in chunks)
                    recalls.append(1.0 if hit else 0.0)
                faiths.append(score.faithfulness_score)
                rels.append(score.relevance_score)
                latencies.append(ms)

                conn.execute(
                    """
                    INSERT INTO eval_logs
                    (pipeline_version, question, expected_source, answer, citations,
                     faithfulness_score, relevance_score, judge_reason, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        mode, q, exp, ans.answer,
                        json.dumps([c.model_dump() for c in ans.citations]),
                        score.faithfulness_score, score.relevance_score, score.reason, ms,
                    ),
                )
                time.sleep(1.0) # avoid Gemini API bursts limits
        print(
            f"[{mode.upper():6}] Recall@5: {sum(recalls)/len(recalls):.2f} | "
            f"Faithfulness: {sum(faiths)/len(faiths):.2f} | "
            f"Relevance: {sum(rels)/len(rels):.2f} | "
            f"Avg Latency: {sum(latencies)//len(latencies)}ms"
        )


if __name__ == "__main__":
    run_benchmark()