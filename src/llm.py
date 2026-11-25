"""Provider-neutral structured LLM calls (Gemini, OpenAI, Anthropic).

Select the provider and model with env vars (or function arguments):

    LLM_PROVIDER=gemini|openai|anthropic   LLM_MODEL=<model id>      # answer generation
    JUDGE_PROVIDER=...                     JUDGE_MODEL=...           # eval LLM judge
    SIM_PROVIDER=...                       SIM_MODEL=...             # eval question + user simulator
                                                                     # (all fall back to LLM_*)

API keys are read by each SDK: GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY,
ANTHROPIC_API_KEY. Embeddings are NOT handled here and stay on Gemini.

# Dictionary with suggested models:
SUGGESTED_MODELS = {
    "gemini": [
        "gemini-4-hydrogen",  # tested; used for generation and judging so far
    ],
    "openai": [
        "gpt-5-nano",  # tested; cheapest and fastest
        "gpt-5-mini",  # tested; good default for generation
        "gpt-5.5",  # untested; DEFAULT_MODELS value
    ],
    "anthropic": [
        "claude-haiku-5-5",  # untested; cheap and fast
        "claude-sonnet-5-5",  # untested; DEFAULT_MODELS value, good judge
        "claude-opus-5-5",  # untested; strongest, most expensive
    ],
}
"""
import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-4-hydrogen",
    "openai": "gpt-5.5",
}

# Embeddings: one source of truth, used by ingest.py and search.py (Gemini only).
# Changing the model or EMBED_DIM requires re-ingesting all chunks, and recreating
# the `chunks` table (db.py builds `vector(EMBED_DIM)` only when the table is new).
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768  # via Matryoshka truncation

_clients: dict = {}


def resolve(role: str = "llm", provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Return (provider, model). role is 'llm' (generation), 'judge' or 'sim' (evals)."""
    prefix = role.upper()
    provider = (
        provider
        or os.getenv(f"{prefix}_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or DEFAULT_PROVIDER
    ).lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider {provider!r}. Use one of {list(DEFAULT_MODELS)}.")
    # A judge on a different provider than LLM_* must not inherit LLM_MODEL.
    env_model = os.getenv(f"{prefix}_MODEL")
    if not env_model and provider == (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower():
        env_model = os.getenv("LLM_MODEL")
    return provider, model or env_model or DEFAULT_MODELS[provider]


ROLES = ("llm", "judge", "sim")  # generator, LLM judge, question/user simulator


def check_roles_differ() -> None:
    """Raise if any two roles (generator, judge, simulator) use the same provider and model.

    A model grading its own answers, or writing the questions it later answers or
    grades, is biased toward them. Set ALLOW_SAME_MODELS=1 to override on purpose,
    for example for a cheap local run.
    """
    if os.getenv("ALLOW_SAME_MODELS") == "1":
        return
    chosen = {role: resolve(role) for role in ROLES}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            if chosen[a] == chosen[b]:
                raise ValueError(
                    f"Roles '{a}' and '{b}' both use {chosen[a][0]}/{chosen[a][1]}. Set "
                    f"{b.upper()}_PROVIDER and/or {b.upper()}_MODEL to a different model, "
                    "or ALLOW_SAME_MODELS=1 to allow it."
                )


def _client(provider: str):
    """Create each SDK client lazily so unused SDKs need not be installed."""
    if provider not in _clients:
        if provider == "gemini":
            from google import genai

            _clients[provider] = genai.Client()
        elif provider == "openai":
            from openai import OpenAI

            _clients[provider] = OpenAI()
        else:
            from anthropic import Anthropic

            _clients[provider] = Anthropic()
    return _clients[provider]


def generate_structured(
    prompt: str,
    schema: type[T],
    *,
    role: str = "llm",
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    fast: bool = False,
) -> T:
    """Send prompt, return a validated instance of schema.

    fast=True asks for low reasoning effort where the provider supports it (Gemini only).
    """
    provider, model = resolve(role, provider, model)
    client = _client(provider)

    if provider == "gemini":
        from google.genai import types

        cfg = dict(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if fast:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        res = client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
        )
        return schema.model_validate_json(res.text)

    if provider == "openai":
        kwargs = {}
        # Reasoning models (o-series, gpt-5*) reject temperature.
        if model.startswith("gpt-4"):
            kwargs["temperature"] = temperature
        res = client.responses.parse(
            model=model, input=prompt, text_format=schema, **kwargs
        )
        if res.output_parsed is None:
            raise RuntimeError(f"OpenAI returned no parsed output (model={model}).")
        return res.output_parsed

    # anthropic: force a single tool call whose input schema is the Pydantic schema
    res = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": "respond",
                "description": "Return the final structured response.",
                "input_schema": schema.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "respond"},
    )
    for block in res.content:
        if block.type == "tool_use":
            return schema.model_validate(block.input)
    raise RuntimeError(f"Anthropic returned no tool_use block (model={model}).")
