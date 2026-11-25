"""Live smoke test for every provider/model in the matrix. Costs a few cents.

    python -m tests.smoke_llm            # tiny structured call per model
    python -m tests.smoke_llm --e2e      # plus 1 real RAG question + judge, using .env settings

A model is SKIPPED when its API key is missing. Exit code 1 if any model FAILS.
Not named test_*.py on purpose: pytest must not spend money by default.
"""

import os
import sys
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.llm import generate_structured

load_dotenv()

# (provider, model) pairs: cheapest useful model per provider, plus the .env choices.
MATRIX = [
    ("gemini", "gemini-4-hydrogen"),
    ("openai", "gpt-5-nano"),
    ("openai", "gpt-5-mini"),
    ("anthropic", "claude-haiku-5-5"),
]
KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class Probe(BaseModel):
    """Tiny schema: checks JSON-schema output, constraints and Norwegian text."""

    answer: int = Field(description="The numeric result")
    word: str = Field(description="The Norwegian word for 'insurance'")


PROMPT = "What is 17 + 25? Also give the Norwegian word for 'insurance'."


def probe(provider: str, model: str) -> tuple[str, str]:
    if not any(os.getenv(k) for k in KEYS[provider]):
        return "SKIP", f"no {KEYS[provider][0]}"
    t0 = time.perf_counter()
    try:
        out = generate_structured(PROMPT, Probe, provider=provider, model=model, fast=True)
    except Exception as e:  # report every failure, keep going
        return "FAIL", f"{type(e).__name__}: {str(e)[:120]}"
    ms = int((time.perf_counter() - t0) * 1000)
    ok = out.answer == 42 and "forsikring" in out.word.lower()
    return ("OK" if ok else "FAIL"), f"{ms}ms answer={out.answer} word={out.word!r}"


def e2e() -> tuple[str, str]:
    from src.evals.run_eval import judge_answer
    from src.generation.rag import ask
    from src.llm import resolve

    q = "Hvem gjelder bilforsikringen for?"
    try:
        ans, chunks, ms = ask(q)
        score = judge_answer(q, "x", ans.answer, ans.is_covered_in_context, chunks)
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:120]}"
    return "OK", (
        f"gen={resolve('llm')} judge={resolve('judge')} {ms}ms "
        f"covered={ans.is_covered_in_context} faith={score.faithfulness_score} rel={score.relevance_score}"
    )


def main() -> int:
    rows = [(f"{p}/{m}", *probe(p, m)) for p, m in MATRIX]
    if "--e2e" in sys.argv:
        rows.append(("e2e RAG+judge", *e2e()))
    for name, status, detail in rows:
        print(f"{status:5} {name:32} {detail}")
    return 1 if any(r[1] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
