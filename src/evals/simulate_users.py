"""Multi-turn synthetic user simulator for end-to-end RAG stress testing.

Simulates Norwegian policyholder personas having 2-turn conversations with the
RAG system (including follow-ups and pushback), then logs simulated user
satisfaction ratings (+1/-1) and comments into the `user_feedback` table.

Usage:
    python -m src.evals.simulate_users
"""

import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.ingestion.db import get_conn
from src.generation.rag import ask

client = genai.Client()
SIM_MODEL = "gemini-4-hydrogen"

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
    init_res = client.models.generate_content(
        model=SIM_MODEL,
        contents=f"{persona_prompt}\nStill ditt første korte spørsmål på norsk til forsikringsboten.",
        config=types.GenerateContentConfig(
            temperature=0.8,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json",
            response_schema=UserTurn,
        ),
    )
    q = UserTurn.model_validate_json(init_res.text).message

    for t in range(turns):
        ans, _, ms = ask(q, mode="hybrid", top_k=5)
        history.append((q, ans.answer))
        print(f"[{persona_name} | Turn {t+1} | {ms}ms]\n  User: {q}\n  Bot:  {ans.answer}\n")

        if t < turns - 1:
            transcript = "\n".join(f"Bruker: {u}\nBot: {b}" for u, b in history)
            follow_res = client.models.generate_content(
                model=SIM_MODEL,
                contents=f"{persona_prompt}\nSamtale hittil:\n{transcript}\nStill ett kort oppfølgingsspørsmål på norsk.",
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json",
                    response_schema=UserTurn,
                ),
            )
            q = UserTurn.model_validate_json(follow_res.text).message
            time.sleep(1.0)

    # Simulated user leaves thumbs-up/down feedback
    transcript = "\n".join(f"Bruker: {u}\nBot: {b}" for u, b in history)
    fb_res = client.models.generate_content(
        model=SIM_MODEL,
        contents=(
            f"{persona_prompt}\nVurder hele samtalen:\n{transcript}\n"
            "Gi rating=1 hvis boten var ærlig (inkl. å si nei når den ikke vet) og nyttig, ellers -1."
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json",
            response_schema=UserFeedback,
        ),
    )
    fb = UserFeedback.model_validate_json(fb_res.text)
    rating = 1 if fb.rating >= 0 else -1

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_feedback (question, answer, rating, comment) VALUES (%s, %s, %s, %s);",
            (history[-1][0], history[-1][1], rating, f"[{persona_name}] {fb.comment}"),
        )
    print(f"-> Logged feedback: {rating} | {fb.comment}\n" + "-" * 50)


if __name__ == "__main__":
    for name, prompt in PERSONAS:
        simulate_conversation(name, prompt, turns=2)