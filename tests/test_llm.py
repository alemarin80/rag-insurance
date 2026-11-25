"""Unit tests for provider/model resolution and the distinct-roles check (no API calls)."""

import pytest
from src.llm import check_roles_differ, resolve

ENV = [
    "LLM_PROVIDER", "LLM_MODEL", "JUDGE_PROVIDER", "JUDGE_MODEL",
    "SIM_PROVIDER", "SIM_MODEL", "ALLOW_SAME_MODELS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def set_roles(monkeypatch, gen, judge, sim):
    for prefix, (provider, model) in {"LLM": gen, "JUDGE": judge, "SIM": sim}.items():
        monkeypatch.setenv(f"{prefix}_PROVIDER", provider)
        monkeypatch.setenv(f"{prefix}_MODEL", model)


def test_unset_roles_fall_back_and_raise(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-5-mini")
    with pytest.raises(ValueError, match="'llm' and 'judge'"):
        check_roles_differ()  # judge and sim fall back to LLM_*


def test_judge_equals_generator_raises(monkeypatch):
    set_roles(monkeypatch, ("openai", "gpt-5-mini"), ("openai", "gpt-5-mini"), ("gemini", "gemini-4-hydrogen"))
    with pytest.raises(ValueError, match="'llm' and 'judge'"):
        check_roles_differ()


def test_simulator_equals_generator_raises(monkeypatch):
    set_roles(monkeypatch, ("openai", "gpt-5-mini"), ("gemini", "gemini-4-hydrogen"), ("openai", "gpt-5-mini"))
    with pytest.raises(ValueError, match="'llm' and 'sim'"):
        check_roles_differ()


def test_simulator_equals_judge_raises(monkeypatch):
    set_roles(monkeypatch, ("openai", "gpt-5-mini"), ("gemini", "gemini-4-hydrogen"), ("gemini", "gemini-4-hydrogen"))
    with pytest.raises(ValueError, match="'judge' and 'sim'"):
        check_roles_differ()


def test_three_distinct_models_pass(monkeypatch):
    set_roles(monkeypatch, ("openai", "gpt-5-mini"), ("gemini", "gemini-4-hydrogen"), ("openai", "gpt-5-nano"))
    check_roles_differ()  # same provider, different model is allowed
    assert resolve("sim") == ("openai", "gpt-5-nano")


def test_override_allows_same(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("ALLOW_SAME_MODELS", "1")
    check_roles_differ()


# --- reasoning effort and token usage (fake clients, no API calls) ---

from types import SimpleNamespace

from pydantic import BaseModel

from src import llm


class Answer(BaseModel):
    text: str


def fake_openai(captured):
    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_parsed=Answer(text="ok"),
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))


@pytest.fixture
def openai_calls(monkeypatch):
    captured = {}
    monkeypatch.setitem(llm._clients, "openai", fake_openai(captured))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    for k in ("LLM_EFFORT", "JUDGE_EFFORT", "SIM_EFFORT"):
        monkeypatch.delenv(k, raising=False)
    return captured


def test_effort_defaults_and_env(monkeypatch):
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    monkeypatch.delenv("JUDGE_EFFORT", raising=False)
    assert llm.resolve_effort("llm") == "low"
    assert llm.resolve_effort("judge") is None  # provider default
    monkeypatch.setenv("JUDGE_EFFORT", "HIGH")
    assert llm.resolve_effort("judge") == "high"
    assert llm.resolve_effort("judge", "minimal") == "minimal"  # argument wins


def test_effort_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LLM_EFFORT", "turbo")
    with pytest.raises(ValueError, match="Unknown effort"):
        llm.resolve_effort("llm")


def test_openai_reasoning_model_gets_effort_and_usage(openai_calls):
    out, tokens_in, tokens_out = llm.generate_structured_usage(
        "hi", Answer, model="gpt-5-mini", effort="low"
    )
    assert out.text == "ok" and (tokens_in, tokens_out) == (11, 7)
    assert openai_calls["reasoning"] == {"effort": "low"}
    assert "temperature" not in openai_calls


def test_openai_gpt4_gets_temperature_not_effort(openai_calls):
    llm.generate_structured("hi", Answer, model="gpt-4o", effort="low", temperature=0.3)
    assert openai_calls["temperature"] == 0.3
    assert "reasoning" not in openai_calls


def test_openai_default_effort_sends_no_reasoning(openai_calls, monkeypatch):
    monkeypatch.setenv("JUDGE_EFFORT", "default")
    llm.generate_structured("hi", Answer, role="judge", model="gpt-5-mini")
    assert "reasoning" not in openai_calls
