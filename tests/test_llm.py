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
