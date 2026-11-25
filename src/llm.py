"""Provider-neutral structured LLM calls (Gemini, OpenAI, Anthropic).

Select the provider and model with env vars (or function arguments):

    LLM_PROVIDER=gemini|openai|anthropic   LLM_MODEL=<model id>      # answer generation
    JUDGE_PROVIDER=...                     JUDGE_MODEL=...           # eval simulator + judge
                                                                     # (fall back to LLM_*)

API keys are read by each SDK: GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY,
ANTHROPIC_API_KEY. Embeddings are NOT handled here and stay on Gemini.
"""

import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODELS = {
    "gemini": "gemini-4-hydrogen",
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-5-5",
}

_clients: dict = {}


def resolve(role: str = "llm", provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Return (provider, model). role is 'llm' (generation) or 'judge' (evals)."""
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
