"""Multi-turn synthetic user simulator for end-to-end RAG stress testing.

Simulates Norwegian policyholder personas having 2-turn conversations with the
RAG system (including follow-ups and pushback), then logs simulated user
satisfaction ratings (+1/-1) and comments into the `user_feedback` table with source='simulated'.

Usage:
    python -m src.evals.simulate_users
"""

import time
from pydantic import BaseModel, Field
from src.ingestion.db import get_conn, init_schema
from src.generation.rag import ask
from src.llm import generate_structured, check_roles_differ

# The simulated user runs on the simulator model (SIM_PROVIDER / SIM_MODEL in .env).

PERSONAS = [
    ("stressed_driver", "Du er en stresset bileier som nettopp har hatt et lite uhell, og er usikker på egenandel og hvem som dekkes."),
    ("pushy_traveler", "Du prøver å få forsikringen til å dekke noe som kanskje er unntatt, og stiller et kritisk oppfølgingsspørsmål."),
    ("off_topic_user", "Du spør om noe helt utenfor bil/innbo/reise (f.eks. kryptovaluta eller hund), og tester om boten dikter opp svar."),
]


class UserTurn(BaseModel):
    """Next follow-up question from the simulated policyholder."""

    message: str = Field(description="Natural Norwegian follow-up question")


class UserFeedback(BaseModel):
    """Simulated end-of-chat user satisfaction rating."""

    rating: int = Field(description="1 if helpful, honest, and clear; -1 if unhelpful or hallucinated")
    comment: str = Field(description="Short 1-sentence Norwegian user feedback")


def simulate_conversation(persona_name: str, persona_prompt: str, turns: int = 2) -> None:
    """Run a multi-turn simulated user session and save feedback to Postgres."""
    history = []
    # Turn 1: initial question
    q = generate_structured(
        f"{persona_prompt}\nStill ditt første korte spørsmål på norsk til forsikringsboten.",
        UserTurn, role="sim", temperature=0.8, fast=True,
    ).message

    for t in range(turns):
        ans, _, ms = ask(q, mode="hybrid", top_k=5)
        history.append((q, ans.answer))
        print(f"[{persona_name} | Turn {t+1} | {ms}ms]\n  User: {q}\n  Bot:  {ans.answer}\n")

        if t < turns - 1:
            transcript = "\n".join(f"Bruker: {u}\nBot: {b}" for u, b in history)
            q = generate_structured(
                f"{persona_prompt}\nSamtale hittil:\n{transcript}\nStill ett kort oppfølgingsspørsmål på norsk.",
                UserTurn, role="sim", temperature=0.8, fast=True,
            ).message
            time.sleep(1.0)

    # Simulated user leaves thumbs-up/down feedback
    transcript = "\n".join(f"Bruker: {u}\nBot: {b}" for u, b in history)
    fb = generate_structured(
        f"{persona_prompt}\nVurder hele samtalen:\n{transcript}\n"
        "Gi rating=1 hvis boten var ærlig (inkl. å si nei når den ikke vet) og nyttig, ellers -1.",
        UserFeedback, role="sim", temperature=0.0, fast=True,
    )
    rating = 1 if fb.rating >= 0 else -1

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_feedback (question, answer, rating, comment, source) VALUES (%s, %s, %s, %s, 'simulated');",
            (history[-1][0], history[-1][1], rating, f"[{persona_name}] {fb.comment}"),
        )
    print(f"-> Logged feedback: {rating} | {fb.comment}\n" + "-" * 50)


if __name__ == "__main__":
    check_roles_differ()
    init_schema()  # creates user_feedback on a fresh database
    for name, prompt in PERSONAS:
        simulate_conversation(name, prompt, turns=2)